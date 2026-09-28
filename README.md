# EgoPose Project Scripts Readme

本项目的 `scripts` 目录是一个“上游检测产物 -> 对齐/配准 -> 估计 -> 生成轨迹与人体网格” 的完整流水线。
主入口为 `scripts/run_body_mesh.py`，其它脚本基本都是它的子步骤。

## 一、入口与输入

### 1.1 主入口

```bash
python /home/xieyongling/slam_project/ego-pose-project/scripts/run_body_mesh.py \
  --config /home/xieyongling/slam_project/ego-pose-project/config/body_mesh.yaml \
  --frames-dir /path/to/frames \
  --cam-space-dir /path/to/cam_space \
  --hawor-hands-vggt-world /path/to/hawor_hands.jsonl \
  --hawor-to-vggt-alignment /path/to/hawor_to_vggt_alignment.json \
  --vggt-head-pose /path/to/vggt_head_pose.jsonl \
  --scene-points-ply /path/to/scene_points.ply \
  --output-dir /path/to/output
```

必填输入：
- `frames-dir`：原视频抽帧图像目录
- `cam-space-dir`：双目/多目 `cam_space` 目录
- `hawor-hands-vggt-world`：HaWoR 手关键点
- `hawor-to-vggt-alignment`：对齐矩阵 json
- `vggt-head-pose`：VGGT 头部姿态 jsonl
- `scene-points-ply`：场景点云
- `config/body_mesh.yaml`：全局参数

### 1.2 插件入口

如果通过服务框架调用，插件在
- [scripts/model_plugin.py](/home/xieyongling/slam_project/ego-pose-project/scripts/model_plugin.py)
会直接调用上述 `run_body_mesh.py`。

## 二、执行流程（run_body_mesh）

`run_body_mesh.py` 串联以下脚本（依次执行）：

1. `01_convert_jsonl_to_camera_poses.py`
   从 VGGT head jsonl 生成：
   - `tmp/camera_poses.txt`
   - `egoallo_inputs/timestamps.txt`
   - `tmp/intrinsics.txt`

2. `02_extract_hawor_world_hands.py`
   HaWoR 手检测预处理，加入权重/软起止策略。

3. `03_stabilize_hawor_hands_for_egoallo.py`
   稳定手部轨迹（去抖/去抖动跳变）。

4. `extract_hawor_to_vggt_rotation.py`
   读对齐 json，产出 `tmp/R_hawor_to_vggt.npy`。

5. `04_build_egoallo_hamer_from_hawor_world.py`
   产出 `tmp/hamer_outputs.pkl`（EgoAllo 输入约束）。

6. `run_geocalib.py`
   用样本帧跑 GeoCalib，产出 `tmp/geocalib_gravity.npz`。
   **有真实 IMU 时这一步整个跳过**（`04_prepare_real_sensor_inputs.py` 已产出 `tmp/imu_gravity.txt`）。

7. `01_estimate_gravity_da3_geocalib.py` + `02_apply_alignment_da3_geocalib.py`
   估计/应用重力和场景主方向，产出：
   - `tmp/R_align.npy`
   - `tmp/alignment_transform.npz`
   - `tmp/head_trajectory.npy`

   这一步有两条互斥的路径：

   - **有真实 IMU（`--real-sensor-input` 的 episode.json 里有 IMU 流）**：
     1. 重力方向取**最静止的一段**（加速度方向抖动 + 模长偏离 + 陀螺仪三项判定），
        并按时间匹配到对应帧号写进 `tmp/imu_gravity.txt` 第一列，下游用**那一帧**的位姿
        把相机系向量转到世界系（不是固定用第 0 帧）。
     2. 先按设备定义的 `imu_vector_type` 得到初始正负号；若有可匹配手部数据，则用
        “相机/头部高度应高于手部高度”的比例校验正负号，比例小于
        `real_sensor.imu_min_camera_above_hand_ratio` 时翻转；没有手部数据时保持该配置。
     3. **只用该方向在最低相机高度以下搜地面候选**：按 2/5/8/12/18/25 分位顺序沿轴向上扫；
        某个分位 cutoff 或拟合 level 不低于最低相机高度时停止继续向上找。
     4. 每个候选由 `refine_floor_coarse_to_fine` 从点云重新估计法向。
     5. 重估法向与 IMU 方向夹角 ≤ `imu_max_refine_angle_deg`，则接受该 refine 后的地面方向。
     6. 角度超限就继续下一个候选；所有候选角度都超限则本次 IMU 对齐失败，但仍正常写
        `R_align.npy` / `alignment_transform.npz`、前端输入、`final_scene.ply` 和空
        `merged_smoothed_vis.npz`，主流程随后只跳过 EgoGVAE 推理。
     7. 头部高度/地面低包络校验仍写日志和元数据，但在真实 IMU 模式下只做诊断。
     8. 有真实 IMU 时不跑全局纯点云 RANSAC，也不回退到纯点云 RANSAC。
     参数见 `config/body_mesh.yaml` 的 `real_sensor.imu_*`，日志统一带 `[IMU]` 前缀，
     结论同时写进 `tmp/alignment_transform.npz`（`imu_stats_json` 等）和
     `tmp/pipeline_run_log.json` 的 `real_sensor.imu_alignment`。

   - **无 IMU**：先跑纯点云 RANSAC，再把 GeoCalib 当作先验（超过 `geocalib_switch_angle_deg`
     才启用）。候选和最终结果必须通过相机/头部高度强校验；有可匹配手部数据时，还必须满足
     头/相机高度高于多数手部点，否则该候选被拒或最终失败。参数见 `alignment.*`。

8. `04_estimate_foundationgeo_metric_scale.py`（可选）
   如启用 FoundationGeo，则估计 FoundationGeo 米制尺度。

   **有真实深度时改走 `04_prepare_real_sensor_inputs.py` 的重力约束地面尺度估计**，
   产出 `tmp/real_depth_scale_estimate.json`。流程：

   1. 上游传入的 RGB/head trajectory 已经是 5FPS；这里只按
      `real_sensor.depth_scale_num_samples` 均匀抽 pose（默认 64 帧），再按
      `timestamps.txt` 的 **`src_frame_id`**（不是采样后的 `frame_id`）通过 depth sample map
      找对应深度帧。
   2. 用真实标定内参将深度反投影为米制点，并通过 `depth_to_color` 外参转到 RGB
      相机系；把对齐阶段确定的世界 up 方向旋转到该帧相机系。
   3. 沿 up 方向寻找水平面候选，以点数、覆盖面积、残差和低处位置过滤单帧候选；
      对每个候选计算
      `implied_scale = real_camera_height / aligned_vggt_head_height`。
   4. 跨帧按 `implied_scale` 聚类，而不是直接聚类相机系 plane level（相机移动时
      后者本来就会变化）。每个 pose 最多投一票，簇必须有多帧独立支撑；优先选
      median scale 最大的物理平面，再用支撑帧数和相对 MAD 作为决胜项。
      因此真地面只在少量帧出现时不会单纯输给更密的桌面，同时单帧深度异常也不能
      定义尺度。
   5. 对选中簇做 MAD 异常值剔除与 median 汇总。该尺度只回答“1 个 head trajectory
      单位等于多少米”，不会替换轨迹旋转或运动形状。

   时间覆盖和聚类门槛由 `real_sensor.depth_scale_num_samples`、
   `depth_scale_cluster_rel_tol`、`depth_scale_min_support_frames` 配置；本地验证会从采样帧映射
   到原始深度帧后向前/向后连续搜索，同一 scale 的高质量水平面前后合并达到
   `depth_scale_local_min_support_frames` 即通过。候选、所有尺度簇、支撑帧和最终选择都会写入
   `real_depth_scale_estimate.json` 便于排查。

9. `05_apply_metric_scale.py`（关键）
   同时尝试 FoundationGeo 与 hands 尺度，取有效尺度最大值并做 p90 头高上限裁剪；两者都失败时 fallback 到 height，并输出：
   - `egoallo_outputs` 前端 jsonl（可配置）
   - `egoallo_outputs/final_scene.ply`
   - `tmp/metric_scale_metadata.json`
   - `tmp/hand_scale_bones.json`
   - `tmp/foundationgeo_metric_scale/foundationgeo_scale_estimate.json`（如启用 FoundationGeo）

10. `05_run_egogvae_from_head.py`
   生成 `egogvae_motion.npz`，供接地置信度检查使用，不向 EgoAllo 注入姿态或体型。

11. `07_run_egoallo_refine_with_prior.py` + `egoallo_stability.py` + `egoallo_hand_refinement.py`
   EgoAllo 使用自身预测的全身体型和姿态，采样窗口 128、重叠 32，软引导内循环 5 次、末尾 20 次。
   可选的 EgoGVAE 下肢姿态引导会在每个 DDIM 步把 EgoAllo 的 clean-pose 预测沿 SO(3) 测地线小幅推向
   EgoGVAE 的八个腿部局部旋转（双髋、膝、踝、脚趾），并在最后若干步平滑释放。它不复制 EgoGVAE
   的根节点、体型、躯干、双臂、腕部或手指；默认强度为 0.10，设为 0 可在同一个扩展 sampler 中关闭姿态引导。
   主配置使用官方腕部位置权重 20，其余引导权重直接使用源码官方默认值。
   保存未做后处理的 `egoallo_hawor_motion.npz`；再做五帧对称姿态平滑（每关节改变量最多 5 度，手指不变），
   随后只优化八个腿部关节的接地、防滑和时序一致性；接地阶段固定根节点、躯干、双臂、腕点和手指。
   最后做一次限幅的肩肘线性化 IK：每关节最多 2 度，目标残差截断为 3 cm，修正量按 0.2 秒高斯核平滑。
   保持根节点、躯干、下肢、手指局部姿势及手掌世界朝向；不补造手部输入，修正仅在约 0.4 秒邻域传播（按中位帧率）。
   每侧只有腕点误差的均值/P95 都不增加才接受；有全手观测时还要求 16 个手部关节误差的均值/P95 不增加。
   不通过的侧整段保持原姿势，避免逐帧开关引入跳变。该检查只衡量观测一致性，不代表真实姿态精度。
   输出 `merged_smoothed_vis.npz`，不再生成 `fused_init.npz`。原始 EgoAllo 文件不含上述后处理。
   EgoGVAE 的接地后处理仍只使用可靠接地概率：置信度 >0.7、足底距地面 <15 cm、足部水平速度 <20 cm/s 时有效；姿态软引导是否启用由独立的 strength 控制。
   这里假设已对齐的平地；不包含桌面碰撞、自碰撞或人体平衡保证。

12. 写总结日志
   `tmp/pipeline_run_log.json` 记录每个子命令耗时与完整 stdout/stderr。

## 三、目录结构（输出）

默认输出结构（可在 `output_layout` 调整）：

- `tmp/`：中间产物（日志、临时轨迹/点云/对齐参数等）
- `egoallo_inputs/`：EgoAllo 输入文件
  - `head_trajectory.npy`
  - `timestamps.txt`
  - `hamer_outputs.pkl`
- `ego_tmp/`：每段推理输出
  - `xx_aa-bb.npz`
  - `xx_aa-bb_args.yaml`
- `egoallo_outputs/`：最终结果
  - `merged_smoothed_vis.npz`
  - `final_scene.ply`
  - `head_trajectory_egoallo.jsonl`（若启用 frontend）
  - `hand_keypoints_egoallo.jsonl`（若启用 frontend）

`tmp`、`ego_tmp`、`egoallo_inputs` 可按配置清理。

## 四、关键参数（`config/body_mesh.yaml`）

### 4.1 资源与路径
- `model_sources.*`：源码 fallback 目录（当前优先使用 conda/site-packages，导入失败时才 fallback 到 `EGOALLO_ROOT` / `GEOCALIB_ROOT` / `FOUNDATIONGEO_ROOT`）。
- `model_weights.*`：`egoallo_checkpoint_dir`、`foundationgeo_model_pt`、`geocalib_pinhole_tar`、`mano_*.pkl` 等。
- `output_layout.*`：各目录名与清理开关。

### 4.2 手部预处理
- `hawor_extract.*`：提取阈值与平滑规则
- `hawor_stabilize.*`：去抖动/分段边界参数
- `hamer_build.*`：`cam_space` 子目录与 MANO 关键点 key 名

### 4.3 对齐与重力
- `geocalib.*`：GeoCalib 抽样与采样策略
- `alignment.*`：是否使用 geocalib prior、RANSAC fallback 及高度验收阈值

### 4.4 量纲缩放（重点）
- `metric_scale.hand_fallback.*`：手部尺度估计参数与 p90 头高上限
- `metric_scale.foundationgeo.*`：FoundationGeo 采样、最大支撑簇汇总、推理参数
- `metric_scale.height_fallback.*`：疑似直立阶段高度 fallback

### 4.5 EgoAllo
- `egoallo_refine.guidance_mode`：`hamer_wrist` 或 `hamer_reproj2`。
- `egoallo_refine.hamer_abspos_weight`：当前与官方均为 20。越大越追随腕点，也越容易牵动身体。
- `egoallo_refine.target_segment_length` / `context_frames`：长序列的分段和上下文长度。
- `egoallo_refine.window_size` / `overlap_size`：默认官方 128 / 32。
- 官方演示默认 `aria_hamer` 使用 Aria 腕点和 HaMeR 手指，不等同于只有 HaWoR 的当前输入。
  参考 [官方推理](https://github.com/brentyi/egoallo/blob/main/3_aria_inference.py) 和
  [官方引导](https://github.com/brentyi/egoallo/blob/main/src/egoallo/guidance_optimizer_jax.py)。

### 4.6 平滑
- `egoallo_refine.body_smoothing_max_deg`：对称五帧平滑的单关节修正上限，设 0 关闭。
- `egoallo_refine.leg_max_iters`：下肢接地优化次数，默认 250，设 0 关闭；不是手部 IK。
- `egoallo_refine.contact_source`：`egogvae` 加入可靠参考接地置信度，`egoallo` 只用自身预测。
- `egoallo_refine.egogvae_leg_pose_strength`：EgoGVAE 八个腿部局部旋转对每步 EgoAllo clean-pose 的软引导强度，范围 0～1，当前保守默认 0.10；设 0 只关闭姿态引导，仍强制使用新 sampler 接口。
- `egoallo_refine.egogvae_leg_pose_max_step_deg`：每个 DDIM 步、每个腿关节最多朝参考旋转的角度；不是最终姿态替换。
- `egoallo_refine.egogvae_leg_pose_release_last_steps`：最后若干 DDIM 步用 smoothstep 把引导降到 0，让 EgoAllo 自由完成末段微调。
- `egoallo_refine.hand_refinement_max_deg`：单次肩肘修正上限，默认 2 度，范围 0～5，设 0 关闭。不是反复强拉腕点；腕部世界朝向保持不变。
- 官方策略对照：腕部位置权重设 20，EgoGVAE 腿姿态强度、平滑上限、腿部迭代数、手部修正上限都设 0。
- 当前流水线强制依赖 `patches/egoallo_body_pose_reference.patch` 扩展的 EgoAllo sampler；即使强度为 0，实际导入路径或接口不匹配也会直接报错，不兼容旧 sampler。
- 没有动作真值时，“更接近 EgoGVAE”只表示先验确实生效，不等于更准确；应结合 HaWoR 手误差、穿地/脚滑、时序指标及固定相机并排视频验收。

### 4.7 EgoGVAE 下肢引导 A/B 验证

测试入口位于 `tests/run_egogvae_leg_pose_guidance_demos.py`。所有产物写入
`tests/outputs/egogvae_leg_pose_guidance/` 并被 git 忽略，不会覆盖 demo 输入或生产输出。
baseline 仍使用当前扩展 sampler，只是显式令 strength 为 0；它不是旧 EgoAllo 兼容路径。

```bash
python tests/run_egogvae_leg_pose_guidance_demos.py \
  --demos demo10 --strengths 0.10 0.25 0.50
```

候选只重跑 07 refine，已有 baseline/EgoGVAE/HaWoR 输入会断点复用。通用评价器分别检查
八关节参考误差、脚底穿地、可靠接触滑动、时序、HaWoR 手误差、CPF anchor 和候选相对
baseline 的上半身/根节点变化；固定相机三路视频由
`tests/render_egogvae_leg_pose_guidance.py` 生成。只有同一参数跨多个 demo 通过数值检查并完成人工
视频审阅后，才应把 YAML strength 从 0 改为非零。

## 五、常用排查点

1. 并行启动失败（`available_vram_gb`）
   - 降低 `target_segment_length`/`context_frames` 或提高 `available_vram_gb`。

2. 量纲缩放跳到 `scale=1.0`
   - 看 `tmp/metric_scale_metadata.json`，通常是 hand/foundationgeo 验收全部失败。

3. 运行中参数来源不一致
   - 所有真实执行参数都在 `run_body_mesh.py -> env` 中转给 EgoAllo；单独在代码默认值和 `egoallo.*` 不要混淆。

4. 输出异常或缺文件
   - 优先看 `tmp/pipeline_run_log.json` 每步 `stderr/stdout` + `return_code`。
   - 再看 `tmp/motion_quality_report.json`（平滑质量）和 `tmp/metric_scale_metadata.json`（缩放决策）。

## 六、独立脚本快速入口

- [scripts/esmateG/run_geocalib.py](/home/xieyongling/slam_project/ego-pose-project/scripts/esmateG/run_geocalib.py)
  `--img-dir ... --output ... --num-samples ...`

- [scripts/esmateG/01_estimate_gravity_da3_geocalib.py](/home/xieyongling/slam_project/ego-pose-project/scripts/esmateG/01_estimate_gravity_da3_geocalib.py)

- [scripts/esmateG/02_apply_alignment_da3_geocalib.py](/home/xieyongling/slam_project/ego-pose-project/scripts/esmateG/02_apply_alignment_da3_geocalib.py)

- [scripts/05_run_egogvae_from_head.py](scripts/05_run_egogvae_from_head.py)
- [scripts/07_run_egoallo_refine_with_prior.py](scripts/07_run_egoallo_refine_with_prior.py)
- [scripts/egoallo_stability.py](scripts/egoallo_stability.py)
- [scripts/convert_to_ego/04_estimate_foundationgeo_metric_scale.py](/home/xieyongling/slam_project/ego-pose-project/scripts/convert_to_ego/04_estimate_foundationgeo_metric_scale.py)
- [scripts/convert_to_ego/05_apply_metric_scale.py](/home/xieyongling/slam_project/ego-pose-project/scripts/convert_to_ego/05_apply_metric_scale.py)

## 七、输出产物清单（建议关注）
- `tmp/pipeline_run_log.json`
- `tmp/metric_scale_metadata.json`
- `tmp/motion_quality_report.json`
- `egoallo_outputs/final_scene.ply`
- `egoallo_outputs/merged_smoothed_vis.npz`
- `egoallo_outputs/head_trajectory_egoallo.jsonl`（如启用）
- `egoallo_outputs/hand_keypoints_egoallo.jsonl`（如启用）
- `tmp/foundationgeo_metric_scale/foundationgeo_scale_estimate.json`
