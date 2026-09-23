#!/usr/bin/env python3
"""
multi_duck_controller.py
=========================

Corre N patos de forma independiente sobre la escena generada por
build_multi_duck_scene.py, usando la MISMA política de caminar (walking
ONNX) para todos. Replica exactamente la lógica de observación/acción de
`infer_policy.py` (path legacy, 51D: ang_vel(3) + proj_gravity(3) +
joint_pos(14) + joint_vel(14) + last_action(14) + vel_cmd(3)) — si tus
políticas se entrenaron con `--new-cmd-obs` (61D), usa
`--new-cmd-obs` aquí también (ver NewCmdObsHelper más abajo).

Requiere que la escena se haya generado con build_multi_duck_scene.py,
que preserva los nombres `trunk_base`, `trunk_base_freejoint`,
`imu_ang_vel`, `imu_accel` con el prefijo `duck{i}_`.

Ejemplos
--------
Línea recta, 6 patos, todos caminando hacia +x:

    uv run python multi_duck_controller.py \
        --scene scene_line_6ducks.xml \
        --walking walk.onnx \
        --n 6 --lin-vel-x 0.25

Carrera, 8 patos, primero en cruzar x=3.0 gana:

    uv run python multi_duck_controller.py \
        --scene scene_race_8ducks.xml \
        --walking walk.onnx \
        --n 8 --lin-vel-x 0.3 --race --finish-x 3.0
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# --- constantes copiadas literalmente de infer_policy.py ---
DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)

TIMESTEP = 0.005
DECIMATION = 4  # -> 50 Hz de control, igual que infer_policy.py
CONTROL_DT = TIMESTEP * DECIMATION
INIT_Z = 0.125  # misma altura inicial que usa infer_policy.py (no-rollers)


def quat_rotate_inverse(quat, vec):
    """Idéntico a PolicyInference.quat_rotate_inverse."""
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


class Duck:
    """Todos los índices de un pato dentro del modelo multi-robot."""

    def __init__(self, model, index, new_cmd_obs):
        self.index = index
        self.new_cmd_obs = new_cmd_obs
        prefix = f"duck{index}_"

        def name2id(objtype, base_name):
            i = mujoco.mj_name2id(model, objtype, prefix + base_name)
            if i < 0:
                raise ValueError(
                    f"No se encontró '{prefix + base_name}' en el modelo. "
                    f"¿La escena se generó con build_multi_duck_scene.py?"
                )
            return i

        self.trunk_base_id = name2id(mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.trunk_freejoint_id = name2id(mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qpos_adr = int(model.jnt_qposadr[self.trunk_freejoint_id])
        self.qvel_adr = int(model.jnt_dofadr[self.trunk_freejoint_id])

        self.imu_ang_vel_id = name2id(mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")

        # actuadores de este pato, en el orden en que MuJoCo los numeró
        # (igual que policy.n_joints/actuator_trnid en infer_policy.py, pero
        # ya viene filtrado por pertenecer a este pato via el prefijo)
        self.actuator_ids = sorted(
            i for i in range(model.nu)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith(prefix)
        )
        self.n_joints = len(self.actuator_ids)
        if self.n_joints != len(DEFAULT_POSE):
            raise ValueError(
                f"duck{index}: {self.n_joints} actuadores, se esperaban "
                f"{len(DEFAULT_POSE)} (DEFAULT_POSE). ¿Es el robot estándar "
                f"de 14 servos?"
            )

        # índices qpos/qvel de cada joint actuado, vía actuator_trnid
        # (mismo truco que infer_policy.py: robusto a cualquier orden)
        self.joint_qpos_indices = [
            int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in self.actuator_ids
        ]
        self.joint_qvel_indices = [
            int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in self.actuator_ids
        ]

        self.default_pose = DEFAULT_POSE.copy()
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.vel_cmd = np.zeros(3, dtype=np.float32)

    def set_initial_pose(self, data, x0, y0):
        data.qpos[self.qpos_adr + 0] = x0
        data.qpos[self.qpos_adr + 1] = y0
        data.qpos[self.qpos_adr + 2] = INIT_Z
        data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = [1, 0, 0, 0]
        for qi, val in zip(self.joint_qpos_indices, self.default_pose):
            data.qpos[qi] = val

    def get_projected_gravity(self, data):
        quat = data.xquat[self.trunk_base_id].copy().astype(np.float32)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return quat_rotate_inverse(quat, world_gravity)

    def get_base_ang_vel(self, model, data):
        sensor_adr = model.sensor_adr[self.imu_ang_vel_id]
        return data.sensordata[sensor_adr:sensor_adr + 3].copy().astype(np.float32)

    def get_joint_pos_relative(self, data):
        current = data.qpos[self.joint_qpos_indices].copy().astype(np.float32)
        return current - self.default_pose

    def get_joint_vel(self, data):
        return data.qvel[self.joint_qvel_indices].copy().astype(np.float32)

    def get_observation(self, model, data):
        """51D legacy (walking): ang_vel(3)+proj_grav(3)+joint_pos(14)+
        joint_vel(14)+last_action(14)+vel_cmd(3). Cambia a 61D si tus
        políticas usan --new-cmd-obs (ver comentario abajo)."""
        obs = [
            self.get_base_ang_vel(model, data),
            self.get_projected_gravity(data),
            self.get_joint_pos_relative(data),
            self.get_joint_vel(data),
            self.last_action,
        ]
        if self.new_cmd_obs:
            cmd = np.zeros(13, dtype=np.float32)
            cmd[0:3] = self.vel_cmd
            obs.append(cmd)
        else:
            obs.append(self.vel_cmd)
        return np.concatenate(obs).astype(np.float32)

    def apply_action(self, data, action, action_scale):
        self.last_action = action.copy()
        target_positions = self.default_pose + action * action_scale
        for local_idx, act_id in enumerate(self.actuator_ids):
            data.ctrl[act_id] = target_positions[local_idx]


def cross_finish_x(model, data, duck, finish_x, start_x):
    """True si el pato ya cruzó la línea (funciona tanto si corre hacia +x
    como hacia -x, según el signo de finish_x - start_x)."""
    x = data.qpos[duck.qpos_adr]
    if finish_x >= start_x:
        return x >= finish_x
    return x <= finish_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--walking", required=True, help="ruta al walk.onnx")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--lin-vel-x", type=float, default=0.25)
    ap.add_argument("--lin-vel-y", type=float, default=0.0)
    ap.add_argument("--ang-vel-z", type=float, default=0.0)
    ap.add_argument("--action-scale", type=float, default=1.0)
    ap.add_argument("--new-cmd-obs", action="store_true",
                     help="usa el layout de comando 13D (61D obs) — pon esto"
                          " si entrenaste/exportaste tus políticas con "
                          "--new-cmd-obs en infer_policy.py")
    ap.add_argument("--race", action="store_true",
                     help="modo carrera: se detiene y anuncia ganador al "
                          "cruzar --finish-x")
    ap.add_argument("--finish-x", type=float, default=3.0)
    ap.add_argument("--start-x", type=float, default=0.0,
                     help="debe coincidir con --start-x usado al generar la escena")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)

    ducks = [Duck(model, i, args.new_cmd_obs) for i in range(args.n)]

    # posiciones iniciales: léelas del propio XML generado (pos x,y de cada
    # body duckN_trunk_base tal como las dejó build_multi_duck_scene.py)
    for duck in ducks:
        x0 = float(model.body_pos[duck.trunk_base_id][0])
        y0 = float(model.body_pos[duck.trunk_base_id][1])
        duck.set_initial_pose(data, x0, y0)
        duck.vel_cmd[:] = [args.lin_vel_x, args.lin_vel_y, args.ang_vel_z]

    mujoco.mj_forward(model, data)

    session = ort.InferenceSession(args.walking)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_obs = session.get_inputs()[0].shape[-1]
    got_obs = ducks[0].get_observation(model, data).size
    if expected_obs != got_obs:
        print(f"[AVISO] la política espera obs de {expected_obs}D pero "
              f"estamos construyendo {got_obs}D. Prueba con/sin --new-cmd-obs.")

    winners = []
    finished = set()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        t0 = time.time()
        while viewer.is_running():
            step_start = time.time()

            if step % DECIMATION == 0:
                # Una inferencia por pato: el ONNX exportado por
                # scripts/export.py trae el batch fijo en 1 (no dinámico),
                # así que no se puede apilar (N,obs) en una sola llamada —
                # hay que llamarlo N veces por paso de control, igual que
                # infer_policy.py hace para 1 robot.
                for duck in ducks:
                    obs = duck.get_observation(model, data)
                    action = session.run([output_name], {input_name: obs[None, :]})[0][0]
                    duck.apply_action(data, action.astype(np.float32), args.action_scale)

            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

            if args.race:
                for duck in ducks:
                    if duck.index in finished:
                        continue
                    if cross_finish_x(model, data, duck, args.finish_x, args.start_x):
                        finished.add(duck.index)
                        elapsed = time.time() - t0
                        place = len(winners) + 1
                        winners.append(duck.index)
                        print(f"#{place}: duck{duck.index}  (t={elapsed:.2f}s)")
                if len(finished) == args.n:
                    print("Carrera terminada. Orden:", winners)
                    break

            elapsed = time.time() - step_start
            sleep_time = TIMESTEP - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
