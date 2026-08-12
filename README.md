<h1 align="center">AACV — Autonomous Arm with Computer Vision</h1>

<p align="center">
An autonomous 6-DOF robotic arm that uses an overhead camera and OpenCV to find colored cubes on a workspace mat, then picks them up and places them entirely on its own — no teleoperation, no pre-recorded motion paths. The arm detects the cubes, converts their positions into real-world coordinates, and solves its own inverse kinematics to reach and manipulate them.
</p>

<p align="center">
Python • OpenCV • ArUco • Inverse Kinematics • Servo Calibration • Raspberry Pi • Robotics
</p>

<p align="center">
<img src="docs/images/demo.gif" width="700" alt="AACV demo">
</p>
<!-- Swap docs/images/demo.gif for an actual sped-up demo clip once one is exported. -->

## README Contents
- [Overview](#overview)
- [Pipeline](#pipeline)
- [Hardware](#hardware)
- [Inverse Kinematics](#inverse-kinematics)
- [Calibration](#calibration)
- [Vision System](#vision-system)
- [Repo Structure](#repo-structure)
- [Running It](#running-it)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Credits](#credits)

## Overview

The project has two objectives, both built on the same core vision → correction → IK → servo pipeline:

1. **Rainbow Stack** — detect six 3D-printed, painted 30 mm cubes (red, orange, yellow, green, blue, purple) placed anywhere on the mat, and stack them on top of each other in rainbow order. The purple cube's real, detected position becomes the base of the tower — the tower isn't built at a fixed board coordinate, it's built wherever purple actually is.
2. **Color-to-Marker Matching** — cubes are placed anywhere on the mat next to ArUco markers, where each marker ID maps to a color (ID 4 = red, ID 5 = orange, and so on). The arm scans once, matches each visible cube to its marker by color, and sorts every matched pair — no stacking, one level only.

Every cycle, regardless of objective, follows the same shape: look at the workspace, convert what the camera sees into millimeters, correct for the arm's known physical inaccuracy, solve inverse kinematics, convert that into real servo commands, then move.

## Pipeline

1. **Camera capture** (`vision/vision.py`) — grabs a frame from the overhead webcam and crops it to the workspace.
2. **ArUco marker detection** (`vision/vision.py`) — finds the four corner markers (IDs 0–3) plus, depending on the task, the color-target markers (IDs 4–9).
3. **Perspective homography** (`vision/vision.py`) — warps the workspace into a flat, top-down 640×400 canonical view using `cv2.getPerspectiveTransform`, so pixel measurements correspond to real, undistorted positions instead of an angled camera view.
4. **Cleanup** (`vision/vision.py`) — ArUco marker regions and the bright claw are painted over with surrounding color so neither is mistaken for a cube.
5. **HSV color segmentation + center of mass** (`vision/vision.py`) — each color gets its own HSV threshold mask; the center of mass of the largest matching blob becomes that cube's detected pixel position.
6. **Pixel → mm conversion** (`vision/vision.py`) — marker ID 3's known physical edge length (24.75 mm) is used as a scale reference to convert every cube/marker pixel position into millimeters relative to the arm's origin.
7. **Physical correction interpolation** (`core/IK_solver.py`, `control/*.py`) — a 3×3 measured error grid is interpolated (`scipy.interpolate.RegularGridInterpolator`) to correct for the arm's position-dependent inaccuracy before any IK math runs.
8. **Inverse kinematics** (`core/IK_solver.py`) — converts the corrected (x, y) target into base/shoulder/elbow/wrist joint angles using the law of cosines.
9. **Servo calibration mapping** (`core/IK_solver.py`) — converts each mathematical joint angle into the actual hardware servo command, using per-servo calibration data.
10. **Pick / place sequence + safe retreat** (`control/final_rainbow_stack.py`, `control/color_marker_match.py`) — the high-level task logic: open gripper → move → close gripper → retreat while holding the cube, then move → open gripper → nudge away → fold back to rest.

Requirements are OpenCV (with the `opencv-contrib` ArUco module), NumPy, SciPy, and `pyserial`. Install with `pip install -r requirements.txt`, then run one of the scripts in `control/` (see [Running It](#running-it)).

## Hardware

| Component | Purpose |
|---|---|
| SO-101 follower arm (6 DOF) | Base, shoulder, elbow, wrist pitch, wrist roll, gripper |
| Hiwonder/Lewansoul-style serial bus servos | Actuate all 6 joints, driven via `lewansoul-servo-bus` |
| USB webcam | Overhead camera, ~737 mm above the mat, captures a cropped 640×400 px workspace |
| 3D-printed, painted cubes (30 mm) | Six colors: red, orange, yellow, green, blue, purple |
| ArUco markers (`DICT_4X4_50`) | IDs 0–3: board corners + scale reference. IDs 4–9: color-to-marker task targets |
| Raspberry Pi | Target deployment device (developed on a Mac, runs standalone on the Pi) |

<p align="center">
<img src="docs/images/arm_setup.jpg" width="700" alt="Arm and camera rig">
</p>
<!-- Swap in a real photo of the arm + overhead camera rig once available. -->

The arm itself wasn't designed by us — it's the open-source **SO-101 follower arm**, sourced and 3D-printed from its published STL files and reference control code rather than bought pre-built. We first verified it fully under its stock leader/follower teleoperation scripts, then wrote all of the autonomous vision + control logic in this repo completely from scratch, independent of that source code.

ArUco markers went through a few material iterations. Paper-printed markers glared badly under normal room lighting and caused missed detections; switching to 3D-printed markers with matte filament fixed it — see [Known Limitations](#known-limitations) for more on this.

## Inverse Kinematics

The arm is treated as a 3-link planar arm for IK purposes — base, shoulder, and elbow are solved directly; the wrist is derived, not solved independently.

**Link lengths:**
```
L1 = 115.48 mm   (upper arm)
L2 = 135.81 mm   (forearm)
L3 = 185.71 mm   (end-effector reach)
Z_offset = 127 mm
```

**Given a target (x, y):**
```
R    = sqrt(x² + y²)                # horizontal distance from base to target
R_w  = R - L3                       # distance to the wrist, after removing end-effector length
Z_w  = Z_target - Z_offset          # wrist height relative to the shoulder
D    = sqrt(R_w² + Z_w²)            # shoulder-to-wrist distance

base_angle     = atan2(y, x)
elbow_angle    = acos((L1² + L2² - D²) / (2 * L1 * L2))
shoulder_angle = acos((D² + L1² - L2²) / (2 * D * L1)) + atan2(Z_w, R_w)
wrist_angle    = 180 - (shoulder_angle + elbow_angle)
```

Both `shoulder_angle` and `elbow_angle` come straight out of the law of cosines applied to the shoulder–elbow–wrist triangle. Forcing `wrist_angle` to always sum with the other two to 180° keeps the gripper parallel to the ground at every target, so the claw always approaches a cube from the side rather than from directly above — this maximizes reach, at the cost of being a slightly awkward approach angle for cubes very close to the base. A "dead zone" close to the base is excluded entirely, since targets that close aren't reachable correctly with this geometry.

## Calibration

Getting the IK math right wasn't enough on its own — two separate layers of physical calibration sit on top of it.

**1. Servo calibration** (`calibration/calibration_results.json`, `calibration/motor_calibration.py`)
The joint angle IK produces doesn't map 1:1 onto the servo's own electrical angle — mounting offsets and mechanical load mean the two scales don't line up. Each servo stores a `reference_joint_angle` / `reference_servo_angle` pair (a known joint angle and the actual hardware angle that produces it), a scale factor, and an `inverted` flag. `map_angle_to_servo()` uses these to convert a desired joint angle into the real command to send, then clamps it to that servo's safe physical range:

```
hardware_angle = reference_servo + sign * (joint_angle - reference_joint) * scale
```

**2. Physical position-error correction** (`calibration/arm_corrections.json`, `calibration/manual_arm_cal.py`)
Even with servo calibration correct, the arm was consistently off by a few inches when reaching a specific (x, y) target — and critically, the error wasn't uniform across the workspace. The right side needed a substantially larger correction than the left side, so a single global offset would fix one side while making the other worse. Instead, actual landing error was measured at a 3×3 grid of known points:

```
X = -127.0, 0.0, 127.0 mm
Y = 228.6, 279.4, 330.2 mm
```

At each point, the difference between the commanded and actual position was recorded as a delta: `{"target": [x, y], "delta": [dx, dy]}`. Those 9 samples feed a `scipy.interpolate.RegularGridInterpolator` at runtime, so any arbitrary target between grid points gets a locally-interpolated correction instead of one flat number.

On top of both of these, a handful of small, deliberately *separate* task-specific biases handle physical quirks the general correction map can't capture:

- **Left-side pickup bias** — the claw only opens toward the right, so a cube on the left side is approached at a sharper angle to guarantee a clean grasp.
- **Placement bias (X/Y)** — placement was systematically shifted from pickup position; tuned separately so fixing one didn't break the other.
- **Safe base nudge** — after placing a cube, the base rotates a few degrees away *before* the rest of the arm folds back, so the claw clears the tower/marker vertically instead of dragging back through it. Only active after placement, not pickup — it wasn't needed there and risked introducing new error.
- **Red extra Z clearance** (rainbow stack only) — the topmost cube gets a little extra placement height so it doesn't clip the stack underneath it.

## Vision System

> Brief for now — a full writeup of this section is still in progress.

The vision pipeline converts a raw camera frame into cube and marker positions in millimeters:

1. **Reference marker tracking & boundary extraction** — detect the four corner ArUco markers to find the mat's boundary.
2. **Perspective homography** — warp the workspace into a flat, top-down canonical view so pixel measurements correspond to real, undistorted positions.
3. **Preprocessing / noise reduction** — mask out ArUco marker regions and the bright claw from the analysis image so neither is mistaken for cube color.
4. **HSV color segmentation & center-of-mass localization** — threshold each color in HSV space, then compute each cube's center of mass from image moments for a precise pickup point.

## Repo Structure

```
vision/          vision.py — camera capture, ArUco detection, HSV color segmentation
core/            IK_solver.py (inverse kinematics + servo angle mapping), hardware_bridge.py
control/         the two real pipelines: final_rainbow_stack.py, color_marker_match.py
calibration/     calibration scripts + calibration_results.json / arm_corrections.json
diagnostics/     low-level test utilities (single servo test, servo ID scan, IK test, etc.)
archive/         earlier prototype scripts, superseded by the ones in control/
lib/             vendored lewansoul-servo-bus library (third-party)
```

## Running It

```bash
pip install -r requirements.txt

# Rainbow stack
python control/final_rainbow_stack.py --dry-run

# Color-to-marker matching
python control/color_marker_match.py --dry-run
```

Serial port and camera index are fixed in-script (`/dev/ttyACM0` and webcam index `0`) rather than CLI flags, since these scripts only ever run against one Raspberry Pi + one webcam setup.

| Flag | What it does |
|---|---|
| `--dry-run` | Runs the full vision + correction + IK pipeline and prints what it *would* do, without moving any servos |
| `--auto` | Runs the pick/place sequence without waiting for a manual ENTER between steps |

## Known Limitations

- **Lighting sensitivity.** Color detection uses fixed HSV thresholds tuned for one lighting setup. Cooler colors (green, blue, purple) separated reliably; warmer colors (red, orange, yellow) were harder, since glare could wash them out or make them bleed into each other. A more robust fix would be adaptive thresholding rather than hardcoded bounds tuned to one room.
- **Left/right accuracy asymmetry.** Positioning error wasn't uniform across the workspace — the right side needed substantially larger corrections than the left. The 3×3 interpolated correction map handles most of this, but the pickup/placement biases layered on top are hand-tuned constants, not analytically derived, and need re-tuning if the arm, camera, or mat moves.
- **Single-frame vision, no averaging.** Each scan uses one camera frame — no multi-frame averaging or tracking to reduce noise in a detected cube position.
- **No automatic verification.** The arm doesn't re-check the camera after a placement to confirm it actually landed correctly — a bad pick or place isn't caught automatically.
- **ArUco marker material mattered more than expected.** Paper-printed markers glared under normal room lighting and caused missed detections; 3D-printed matte markers fixed it, but it was a real failure mode worth knowing about if you're replicating this setup.

## Future Improvements

- Multi-frame averaging/tracking for cube position to reduce vision noise.
- Re-calibrate the 3×3 correction grid after any physical change (new camera position, new mat, moving to different hardware) — current values are specific to the exact setup they were measured on.
- Replace the fixed placement biases with additional measured calibration points instead of hand-tuned constants.
- Tune HSV thresholds under the final, permanent lighting setup rather than whatever was available during development.
- Add a post-placement verification pass using the camera.
- Add collision/reachability checks before committing to a pick or place command.

## Credits

- Arm design and base hardware: the open-source **SO-101 follower arm**, built from its published STL files and reference code — not designed by us, but sourced, 3D-printed, and assembled ourselves. All autonomous vision and control logic in this repo was written from scratch, independent of the arm's stock scripts.
- Servo communication: [`lewansoul-servo-bus`](lib/lewansoul-servo-bus-master) (vendored third-party library — see its own README/LICENSE for details).

---

<p align="center">If you found this project interesting, feel free to star the repo!</p>
