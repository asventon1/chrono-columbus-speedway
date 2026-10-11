#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Record six synchronised roof cameras on a car driving round the speedway, for BEVFormer.

    python record_cameras.py --out runs/lap1

The car drives itself round a route in the right-hand lane while six cameras on its roof, laid
out as on the nuScenes car that BEVFormer is trained on, film it with Chrono::Sensor. Written to
--out:

    CAM_FRONT.mp4 ... CAM_BACK_RIGHT.mp4   one video per camera. Frame k of every video was
                                           taken at the same simulated instant
    calibration.json                       each camera's intrinsics and where it sits on the car
    frames.csv                             per frame: its time, where the car was (ego to
                                           global), its velocity, acceleration and turn rate,
                                           and the driver's inputs

The simulation does not run in real time: it takes as long as rendering takes, and every frame
is kept. Chrono::Sensor needs an NVIDIA GPU, and ffmpeg has to be on the path.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import speedway

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTE = os.path.join(HERE, "routes", "east_loop.csv")

# The nuScenes camera rig, which BEVFormer is trained on: name, position in the ego frame in
# metres, yaw in degrees (positive to the left), and focal length in pixels at 1600 x 900. The
# figures are nuScenes's calibration, rounded. The ego frame is nuScenes's too: its origin is
# the middle of the rear axle on the ground, x forward, y left, z up.
RIG = [
    ("CAM_FRONT", (1.70, 0.00, 1.51), 0.0, 1266.0),
    ("CAM_FRONT_RIGHT", (1.55, -0.49, 1.50), -55.0, 1260.0),
    ("CAM_FRONT_LEFT", (1.52, 0.49, 1.51), 55.0, 1257.0),
    ("CAM_BACK", (0.03, 0.00, 1.58), 180.0, 809.0),
    ("CAM_BACK_LEFT", (1.04, 0.48, 1.59), 110.0, 1256.0),
    ("CAM_BACK_RIGHT", (1.08, -0.49, 1.57), -110.0, 1259.0),
]

# A camera's axes as OpenCV and nuScenes have them, x right, y down, z forward, written in
# Chrono::Sensor's camera frame, which looks along x with z up.
OPENCV_IN_CHRONO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

SKY = "sensor/textures/sky_2_4k.hdr"

# The route's speed profile: no faster than --speed, and slow enough for each bend.
LATERAL_ACCEL = 2.5  # m/s^2 round a bend
DECEL = 2.0          # m/s^2 braking for one

STEP = 1e-3

# Where acceleration and turn rate sit in a row of the car's state: ax, ay, az, wx, wy, wz.
MOTION = slice(10, 16)


def main():
    parser = argparse.ArgumentParser(description="Record six synchronised roof cameras on a car driving round the speedway.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", metavar="DIR", required=True, help="directory for the videos, calibration.json and frames.csv")
    parser.add_argument("--route", metavar="CSV", default=ROUTE, help="closed loop to drive, x,y per line in the scene frame\n(default: routes/east_loop.csv, 620 m round the east end)")
    parser.add_argument("--laps", type=float, default=1.0, help="how far to drive, in laps of the route (default: 1)")
    parser.add_argument("--duration", metavar="S", type=float, default=None, help="stop after S seconds of recording instead")
    parser.add_argument("--speed", metavar="V", type=float, default=10.0, help="top speed, m/s. Bends are taken slower (default: 10)")
    parser.add_argument("--rate", metavar="HZ", type=float, default=12.0, help="frames per second, the same for every camera (default: 12, as nuScenes)")
    parser.add_argument("--width", type=int, default=1600, help="image width (default: 1600, as nuScenes)")
    parser.add_argument("--height", type=int, default=900, help="image height (default: 900)")
    parser.add_argument("--supersample", type=int, default=2, help="rays per pixel along each side, against jagged edges (default: 2)")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality, lower is better and bigger (default: 18)")
    parser.add_argument("--sun", nargs=2, type=float, metavar=("ELEV", "AZIM"), default=(50.0, -120.0),
                        help="sun elevation and azimuth in degrees, azimuth anticlockwise from east (default: 50 -120)")
    parser.add_argument("--warmup", metavar="S", type=float, default=1.0, help="seconds to let the car settle before recording (default: 1)")
    parser.add_argument("--preview", action="store_true", help="show the front camera in a window as it records")
    parser.add_argument("--data", metavar="DIR", default=speedway.SCENE_DIR, help="scene directory (default: scene/ beside speedway.py)")
    parser.add_argument("--textures", choices=list(speedway.TEXTURES), default="standard", help="ground photo detail (default: standard)")
    for flag, _, what in speedway.OPTIONAL:
        parser.add_argument(f"--no-{flag}", action="store_true", help=f"leave out {what}")
    parser.add_argument("--tire", choices=sorted(speedway.TIRES), default="pac02", help="tire model (default: pac02)")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on the path. It writes the videos.")
    try:
        import pychrono as chrono
        import pychrono.sensor as sens
        import pychrono.vehicle as veh
    except ImportError as err:
        raise SystemExit(f"This needs PyChrono with the vehicle and sensor modules ({err}).\n"
                         "  conda install projectchrono::pychrono -c conda-forge")

    scene = speedway.fetch(args.data)
    route = read_route(args.route)
    boot = time.perf_counter()

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(150)
    system.SetMaxPenetrationRecoverySpeed(4.0)

    leave_out = {group for flag, group, _ in speedway.OPTIONAL if getattr(args, "no_" + flag.replace("-", "_"))}
    speedway.add_scenery(system, scene, args.textures, groups=[g for g in speedway.GROUPS if g not in leave_out])
    terrain = speedway.add_ground(system, scene)

    # The route at ground height, so the path follower's 3D distances are measured near the car.
    heights = GroundHeights(scene)
    route_z = np.array([heights(x, y) for x, y in route])
    x0, y0 = route[0]
    yaw0 = math.atan2(route[1][1] - y0, route[1][0] - x0)

    audi = veh.WheeledVehicle(system, veh.GetVehicleDataFile("audi/json/audi_Vehicle.json"))
    audi.Initialize(chrono.ChCoordsysd(chrono.ChVector3d(x0, y0, route_z[0] + 0.5), chrono.QuatFromAngleZ(yaw0)))
    for set_type in (audi.SetChassisVisualizationType, audi.SetSuspensionVisualizationType,
                     audi.SetSteeringVisualizationType, audi.SetWheelVisualizationType):
        set_type(chrono.VisualizationType_MESH)
    engine = veh.ReadEngineJSON(veh.GetVehicleDataFile("audi/json/audi_EngineSimpleMap.json"))
    transmission = veh.ReadTransmissionJSON(veh.GetVehicleDataFile("audi/json/audi_AutomaticTransmissionSimpleMap.json"))
    audi.InitializePowertrain(veh.ChPowertrainAssembly(engine, transmission))
    for axle in audi.GetAxles():
        for wheel in axle.GetWheels():
            tire = veh.ReadTireJSON(veh.GetVehicleDataFile(speedway.TIRES[args.tire]))
            tire.SetStepsize(1e-4)
            audi.InitializeTire(tire, wheel, chrono.VisualizationType_MESH)

    ego = ego_offset(audi)

    path = chrono.ChBezierCurve(chrono.vector_ChVector3d([chrono.ChVector3d(x, y, z) for (x, y), z in zip(route, route_z)]), True)
    driver = veh.ChPathFollowerDriver(audi, path, "route", 0.0)
    driver.GetSteeringController().SetLookAheadDistance(5.0)
    driver.GetSteeringController().SetGains(0.8, 0.0, 0.0)
    driver.GetSpeedController().SetGains(0.4, 0.0, 0.0)
    driver.Initialize()
    speeds = speed_profile(route, args.speed)

    # The cameras ride on a body of their own, put at the ego frame every step. Their mounting
    # offsets are then exactly the sensor-to-ego transforms written to calibration.json.
    rig = chrono.ChBody()
    rig.SetFixed(True)
    rig.EnableCollision(False)
    system.Add(rig)

    manager = sens.ChSensorManager(system)
    elevation, azimuth = (math.radians(a) for a in args.sun)
    manager.scene.AddDirectionalLight(chrono.ChColor(1.0, 1.0, 1.0), elevation, azimuth)
    manager.scene.SetAmbientLight(chrono.ChVector3f(0.15, 0.15, 0.15))
    background = sens.Background()
    sky = chrono.GetChronoDataFile(SKY)
    if os.path.isfile(sky):
        background.mode = sens.BackgroundMode_ENVIRONMENT_MAP
        background.env_tex = sky
    else:
        background.mode = sens.BackgroundMode_GRADIENT
        background.color_zenith = chrono.ChVector3f(0.2, 0.4, 0.8)
        background.color_horizon = chrono.ChVector3f(0.7, 0.8, 0.95)
    manager.scene.SetBackground(background)

    cameras, calibration = [], {}
    for name, position, yaw, focal in RIG:
        hfov = 2.0 * math.atan(800.0 / focal)  # the field of view the nuScenes camera has
        mount = chrono.ChFramed(chrono.ChVector3d(*position), chrono.QuatFromAngleZ(math.radians(yaw)))
        camera = sens.ChCameraSensor(rig, args.rate, mount, args.width, args.height, hfov, args.supersample)
        camera.SetName(name)
        camera.PushFilter(sens.ChFilterRGBA8Access())
        if args.preview and name == "CAM_FRONT":
            camera.PushFilter(sens.ChFilterVisualize(args.width // 2, args.height // 2, name))
        manager.AddSensor(camera)
        cameras.append(camera)
        calibration[name] = camera_calibration(name, position, yaw, hfov, args.width, args.height)

    os.makedirs(args.out, exist_ok=True)
    doc = speedway.manifest(scene)
    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump({
            "rate_hz": args.rate,
            "width": args.width,
            "height": args.height,
            "ego_frame": "middle of the rear axle on the ground; x forward, y left, z up; metres",
            "camera_frame": "OpenCV and nuScenes: x right, y down, z forward",
            "global_frame": doc["frame"],
            "ego_in_chassis": list(ego),
            "cameras": calibration,
        }, f, indent=2)

    videos = {camera.GetName(): open_video(os.path.join(args.out, camera.GetName() + ".mp4"), args) for camera in cameras}
    table = open(os.path.join(args.out, "frames.csv"), "w", newline="")
    rows = csv.writer(table)
    rows.writerow(["frame", "time", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "ax", "ay", "az",
                   "wx", "wy", "wz", "speed", "steering", "throttle", "braking"])

    print(f"\n  [{time.perf_counter() - boot:.1f} s to build the scene]")
    print(f"  recording {len(cameras)} cameras at {args.width}x{args.height}, {args.rate:g} frames/s, into {args.out}")

    route_length = float(np.sum(np.hypot(*np.diff(np.vstack([route, route[:1]]), axis=0).T)))
    goal = args.laps * route_length
    index, travelled = 0, 0.0
    poses = {}        # launch time in steps -> the car's state when the cameras were launched
    pending = {}      # launch count -> {camera name: (time stamp, image)}
    last_seen = {camera.GetName(): -1 for camera in cameras}
    launches = 0
    motion, motion_steps = np.zeros(6), 0  # acceleration and turn rate summed since the last launch
    written = dropped = 0
    wall0 = time.perf_counter()
    next_report = 0.0
    start = None

    while True:
        now = system.GetChTime()
        if start is None and now >= args.warmup:
            start = now
        if start is not None:
            if args.duration is not None and now - start >= args.duration:
                break
            if args.duration is None and travelled >= goal:
                break

        # Where along the route the car is, so the speed it is asked for looks ahead to bends.
        pos = audi.GetPos()
        nearest = nearest_index(route, index, pos.x, pos.y)
        step_ahead = (nearest - index) % len(route)
        if step_ahead < len(route) // 2:
            travelled += step_ahead * route_length / len(route)
            index = nearest
        driver.SetDesiredSpeed(speeds[(index + 3) % len(route)])

        inputs = driver.GetInputs()
        if start is None:
            # Settling: brakes on, or the car rolls off down the grade.
            inputs.m_throttle, inputs.m_braking = 0.0, 1.0
        driver.Synchronize(now)
        terrain.Synchronize(now)
        audi.Synchronize(now, inputs, terrain)
        driver.Advance(STEP)
        terrain.Advance(STEP)
        audi.Advance(STEP)
        system.DoStepDynamics(STEP)

        ego_pos, ego_rot, state = ego_state(audi, ego, inputs)
        rig.SetPos(ego_pos)
        rig.SetRot(ego_rot)
        manager.Update()
        motion += state[MOTION]
        motion_steps += 1

        # A launch shows as a new count on the first camera. Keep the car's state at that
        # instant, which is what the frames launched now will show. Acceleration and turn rate
        # are averaged since the last frame, as an IMU would report them: from one step to the
        # next they are dominated by the suspension and tires chattering.
        count = cameras[0].GetNumLaunches()
        if count != launches:
            launches = count
            state[MOTION] = motion / motion_steps
            motion[:], motion_steps = 0.0, 0
            poses[round(system.GetChTime() / STEP)] = state

        # Each frame arrives one launch later, when the next one starts. Write a frame only
        # once all six cameras have delivered it, so the videos stay in step.
        for camera in cameras:
            buffer = camera.GetMostRecentRGBA8Buffer()
            name = camera.GetName()
            if not buffer.HasData() or buffer.LaunchedCount == last_seen[name]:
                continue
            last_seen[name] = buffer.LaunchedCount
            pending.setdefault(buffer.LaunchedCount, {})[name] = (buffer.TimeStamp, buffer.GetRGBA8Data())
        for launch in sorted(pending):
            frame = pending[launch]
            if len(frame) < len(cameras):
                break
            del pending[launch]
            stamp = next(iter(frame.values()))[0]
            key = round(stamp / STEP)
            if any(round(s / STEP) != key for s, _ in frame.values()):
                raise SystemExit(f"cameras disagree on the time of launch {launch}: {sorted(s for s, _ in frame.values())}")
            state = poses.pop(key, None)
            if start is None or stamp < start or state is None:
                continue
            for name, (_, image) in frame.items():
                videos[name].stdin.write(np.ascontiguousarray(image).data)
            rows.writerow([written, f"{stamp - start:.4f}"] + [f"{v:.6f}" for v in state])
            written += 1
        # A frame some camera never delivered is dropped from every video.
        for launch in [k for k in pending if k < max(last_seen.values()) - 1]:
            del pending[launch]
            dropped += 1
        poses = {k: v for k, v in poses.items() if k * STEP > now - 2.0}

        if now >= next_report:
            wall = time.perf_counter() - wall0
            where = f"{travelled / route_length:4.2f} laps" if start is not None else "settling"
            print(f"  t={now:6.1f} s   {where}   {written:5d} frames   {audi.GetSpeed():5.1f} m/s   {now / max(wall, 1e-9):.2f}x real time")
            next_report += 5.0

    table.close()
    for video in videos.values():
        video.stdin.close()
    for video in videos.values():
        video.wait()
    print(f"  {written} frames from each camera, {written / args.rate:.1f} s of video, in {args.out}")
    if dropped:
        print(f"  {dropped} frames were dropped from every video because some camera did not deliver them. "
              "frames.csv has the time of each frame that was kept.")

    sys.stdout.flush()
    os._exit(0)  # tearing Chrono down from Python can crash after everything has worked


def read_route(path):
    points = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line[0].isalpha():
                continue
            x, y = line.split(",")[:2]
            points.append((float(x), float(y)))
    if len(points) < 3:
        raise SystemExit(f"{path} holds no route: it wants x,y lines")
    return np.array(points)


class GroundHeights:
    """speedway.ground_height for many points: the ground mesh is read once."""

    def __init__(self, scene_dir):
        verts = []
        with open(os.path.join(scene_dir, speedway.GROUND)) as f:
            for line in f:
                if line.startswith("v "):
                    verts.append(line.split()[1:4])
        self.v = np.array(verts, float)

    def __call__(self, x, y, radius=2.0):
        near = (np.abs(self.v[:, 0] - x) < radius) & (np.abs(self.v[:, 1] - y) < radius)
        if not near.any():
            raise SystemExit(f"the route at ({x:.1f}, {y:.1f}) is off the ground mesh")
        return float(self.v[near, 2].max())


def speed_profile(route, top_speed):
    """Speed at each route point: at most top_speed, slow enough for the bend, braked for in time."""
    p = np.asarray(route)
    a, b, c = np.roll(p, 1, axis=0), p, np.roll(p, -1, axis=0)
    ab, bc, ca = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    curvature = 2.0 * cross / np.maximum(ab * bc * ca, 1e-9)
    # Smooth over a few points: a route sampled every couple of metres is noisy point to point.
    for _ in range(3):
        curvature = np.maximum(curvature, 0.5 * (np.roll(curvature, 1) + np.roll(curvature, -1)))
    v = np.minimum(top_speed, np.sqrt(LATERAL_ACCEL / np.maximum(curvature, 1e-6)))
    v = np.maximum(v, 3.0)
    # Braking: a point is no faster than the next allows. Twice round, since the loop is closed.
    for _ in range(2):
        for k in range(len(v) - 1, -1, -1):
            v[k] = min(v[k], math.sqrt(v[(k + 1) % len(v)] ** 2 + 2.0 * DECEL * bc[k]))
    return v


def nearest_index(route, index, x, y, back=5, ahead=40):
    """The route point nearest (x, y), searched a little behind and ahead of index."""
    ks = np.arange(index - back, index + ahead) % len(route)
    d = np.hypot(route[ks, 0] - x, route[ks, 1] - y)
    return int(ks[np.argmin(d)])


def ego_offset(vehicle):
    """The ego origin in the chassis reference frame: under the middle of the rear axle, on the ground."""
    ref = vehicle.GetRefFrame()
    axle = vehicle.GetAxles()[-1]
    centres = [ref.TransformPointParentToLocal(wheel.GetPos()) for wheel in axle.GetWheels()]
    radius = axle.GetWheels()[0].GetTire().GetRadius()
    x = sum(c.x for c in centres) / len(centres)
    z = sum(c.z for c in centres) / len(centres) - radius
    return (x, 0.0, z)


def ego_state(vehicle, ego, inputs):
    """The ego frame's pose in the scene frame, and its row of frames.csv after frame and time."""
    import pychrono as chrono

    ref = vehicle.GetRefFrame()
    local = chrono.ChVector3d(*ego)
    pos = ref.TransformPointLocalToParent(local)
    rot = ref.GetRot()
    vel = ref.TransformDirectionParentToLocal(vehicle.GetPointVelocity(local))
    acc = ref.TransformDirectionParentToLocal(vehicle.GetPointAcceleration(local))
    w = ref.GetAngVelLocal()
    state = np.array([pos.x, pos.y, pos.z, rot.e0, rot.e1, rot.e2, rot.e3, vel.x, vel.y, vel.z, acc.x, acc.y, acc.z,
                      w.x, w.y, w.z, vehicle.GetSpeed(), inputs.m_steering, inputs.m_throttle, inputs.m_braking])
    return pos, rot, state


def camera_calibration(name, position, yaw, hfov, width, height):
    """A camera's entry in calibration.json, in the conventions nuScenes uses."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    yaw_matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotation = yaw_matrix @ OPENCV_IN_CHRONO
    # Chrono::Sensor's pinhole has square pixels and its axis through the middle of the image.
    focal = 0.5 * width / math.tan(0.5 * hfov)
    return {
        "video": name + ".mp4",
        "intrinsic": [[focal, 0.0, 0.5 * (width - 1)], [0.0, focal, 0.5 * (height - 1)], [0.0, 0.0, 1.0]],
        "sensor2ego_translation": list(position),
        "sensor2ego_rotation": quaternion(rotation),
        "hfov_deg": math.degrees(hfov),
    }


def quaternion(m):
    """(w, x, y, z) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
    return [w, x, y, z]


def open_video(path, args):
    """An ffmpeg process that takes raw RGBA frames on stdin. Chrono's images are bottom row first."""
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{args.width}x{args.height}",
         "-framerate", f"{args.rate:g}", "-i", "-", "-vf", "vflip", "-c:v", "libx264", "-preset", "medium",
         "-crf", str(args.crf), "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE,
    )


if __name__ == "__main__":
    main()
