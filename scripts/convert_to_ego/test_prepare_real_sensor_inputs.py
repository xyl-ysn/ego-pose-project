"""Run with the pipeline Python: python scripts/convert_to_ego/test_prepare_real_sensor_inputs.py."""
import copy
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location(
    "prepare_real_sensor_inputs", Path(__file__).with_name("04_prepare_real_sensor_inputs.py")
)
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)


def test_intrinsics_fallback():
    calibration = {"cameras": [{
        "stream_id": "ego_rgb",
        "resolution": [1280, 720],
        "intrinsics": {"fx": 900, "fy": 901, "cx": 640, "cy": 360},
    }]}
    with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
        root = Path(directory)
        timestamps = root / "timestamps.txt"
        timestamps.write_text("frame_id timestamp src_frame_id\n0 0.0 0\n1 0.033333333 1\n")
        out = root / "intrinsics.txt"
        upstream = b"# fx fy cx cy source_width source_height\n439 441 256 148 518 294\n440 442 257 149 518 294\n"

        # Valid real calibration must still override the upstream estimate.
        result = sensor.write_intrinsics(calibration, "ego_rgb", out, timestamps)
        assert result["source"] == "camera"
        np.testing.assert_array_equal(np.loadtxt(out), [[900, 901, 640, 360, 1280, 720]] * 2)
        prepared = copy.deepcopy(calibration)
        prepared["cameras"][0]["prepared_image_geometry"] = {
            "intrinsics": {"fx": 450, "fy": 450.5, "cx": 320, "cy": 180},
            "resolution": {"width": 640, "height": 360},
        }
        result = sensor.write_intrinsics(prepared, "ego_rgb", out, timestamps)
        assert result["source"] == "prepared_image_geometry"
        np.testing.assert_array_equal(np.loadtxt(out), [[450, 450.5, 320, 180, 640, 360]] * 2)

        invalid_calibrations = [{"cameras": []}]
        for key, value in [("fx", None), ("fy", 0), ("fx", -1), ("cx", "bad"),
                           ("cy", float("nan")), ("fx", float("inf"))]:
            invalid = copy.deepcopy(calibration)
            invalid["cameras"][0]["intrinsics"][key] = value
            invalid_calibrations.append(invalid)
        missing = copy.deepcopy(calibration)
        del missing["cameras"][0]["intrinsics"]["fx"]
        invalid_calibrations.append(missing)
        for resolution in (None, [0, 720], [1280], {"width": 1280}, ["bad", 720]):
            invalid = copy.deepcopy(calibration)
            invalid["cameras"][0]["resolution"] = resolution
            invalid_calibrations.append(invalid)
        invalid = copy.deepcopy(prepared)
        invalid["cameras"][0]["prepared_image_geometry"]["intrinsics"]["fx"] = None
        invalid_calibrations.append(invalid)
        for invalid in invalid_calibrations:
            out.write_bytes(upstream)
            result = sensor.write_intrinsics(invalid, "ego_rgb", out, timestamps)
            assert result["status"] == "kept" and result["calibration_error"]
            assert out.read_bytes() == upstream

        placeholder = copy.deepcopy(calibration)
        placeholder["cameras"][0]["intrinsics"] = dict.fromkeys(("fx", "fy", "cx", "cy"))
        # Missing, malformed, mismatched, or invalid fallback data must fail
        # clearly without replacing the file with guessed calibration.
        for content in (None, b"bad\n", b"439 441 256 148 518 294\n",
                        upstream.replace(b"439", b"nan"), upstream.replace(b"439", b"0"),
                        upstream.replace(b"518", b"-1")):
            out.unlink(missing_ok=True)
            if content is not None:
                out.write_bytes(content)
            try:
                sensor.write_intrinsics(placeholder, "ego_rgb", out, timestamps)
            except ValueError as exc:
                assert "ego_rgb" in str(exc) and "fx" in str(exc) and "upstream intrinsics" in str(exc)
            else:
                raise AssertionError("Invalid fallback was accepted")
            assert (out.read_bytes() if out.exists() else None) == content

        # A single frame must keep its (1, 6) shape through validation.
        timestamps.write_text("frame_id timestamp src_frame_id\n0 0.0 0\n")
        out.write_bytes(b"439 441 256 148 518 294\n")
        assert sensor.write_intrinsics(placeholder, "ego_rgb", out, timestamps)["frame_count"] == 1

        # An intrinsics fallback must not skip subsequent IMU/depth work.
        (root / "calibration.json").write_text(json.dumps(placeholder))
        (root / "episode.json").write_text(json.dumps({
            "calibration_uri": "calibration.json",
            "streams": [{"stream_id": "ego_imu", "modality": "imu"}],
        }))
        argv = [str(spec.origin), "--episode-dir", str(root), "--camera-stream-id", "ego_rgb",
                "--intrinsics-out", str(out), "--timestamps-txt", str(timestamps),
                "--imu-gravity-out", str(root / "imu.txt"), "--scale-json", str(root / "scale.json")]
        before = out.read_bytes()
        with patch.object(sys, "argv", argv), \
                patch.object(sensor, "write_imu_gravity", return_value={"status": "ok"}) as imu, \
                patch.object(sensor, "write_real_depth_scale", return_value={"status": "ok"}) as depth:
            sensor.main()
            imu.assert_called_once()
            depth.assert_called_once()
        assert out.read_bytes() == before


if __name__ == "__main__":
    test_intrinsics_fallback()
    print("Intrinsics fallback regression checks passed.")
