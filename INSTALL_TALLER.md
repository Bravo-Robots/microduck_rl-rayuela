# Instalación para el taller: Microduck RL + rayuela (ROS 2)

Guía paso a paso para dejar listo el proyecto antes del taller. Hay **dos
entornos Python separados a propósito** y conviene no mezclarlos:

| Entorno | Python | Qué contiene | Para qué |
|---|---|---|---|
| venv del proyecto (`.venv/`, gestionado por `uv`) | 3.12 | mjlab, MuJoCo Warp, torch, rsl_rl, onnxruntime, BAM | Entrenar, exportar a ONNX, simular con los actuadores BAM |
| ROS 2 Humble (Python del sistema) | 3.10 | rclpy, mensajes, colcon, OpenCV | Los nodos de la rayuela (visión, control, teleop, puente) |

`rclpy` solo existe para el Python 3.10 de Humble y `bam` exige Python ≥ 3.12,
por eso la simulación corre en el venv (`sim_worker.py`) y habla con ROS por un
socket Unix (`bridge_node.py`, `rayuela_ipc.py`).

**Regla de oro:** usa una terminal para `uv` y otra para ROS 2. No pongas
`source /opt/ros/humble/setup.bash` en tu `~/.bashrc`: el `PYTHONPATH` de ROS se
cuela en el venv y rompe las importaciones.

## 0. Requisitos

- Ubuntu 22.04 (es la plataforma oficial de ROS 2 Humble).
- GPU NVIDIA con driver reciente **solo para entrenar**. La demo de la rayuela y
  `scripts/infer_policy.py` corren en CPU.
- Unos 10 GB libres (torch + CUDA + ROS 2).

## 1. Clonar el repositorio

```bash
git clone https://github.com/dasanplaen-cmyk/microduck_rl
cd microduck_rl
git checkout rayuela
```

Las rutas de la rayuela están fijadas a `/home/robot/microduck_rl`
(`ros2_microduck/src/rayuela/launch/rayuela.launch.py`, `sim_node.py`,
`sim_worker.py`). Si tu clon está en otro sitio, crea un enlace:

```bash
sudo mkdir -p /home/robot
sudo ln -s "$PWD" /home/robot/microduck_rl
```

## 2. Entorno Python del proyecto (uv + venv 3.12)

Terminal A, **sin** ROS cargado:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # instala uv
exec $SHELL                                       # recarga el PATH

uv sync          # crea .venv/ con Python 3.12 y todas las dependencias
```

`uv` descarga Python 3.12 si no lo tienes. En máquinas ARM (DGX Spark, Jetson)
exporta antes `UV_HTTP_TIMEOUT=600`, porque la primera descarga de CUDA es
grande.

Comprueba que todo está bien:

```bash
uv run list-envs                                              # debe listar Mjlab-BallKickSpeed-*
uv run --with pytest pytest tests/test_ball_kick_speed_cfg.py # tests en CPU
```

Smoke test de entrenamiento (necesita GPU, tarda poco):

```bash
uv run train Mjlab-BallKickSpeed-Right-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5
```

## 3. ROS 2 Humble (Python del sistema)

Terminal B. Instalación estándar de Humble por apt
([guía oficial](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)):

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions \
    python3-numpy python3-opencv ros-humble-tf2-ros
```

`vision_node` usa OpenCV y numpy del sistema; con el modo puente
(`use_bam_bridge:=true`) no hace falta instalar MuJoCo ni onnxruntime en el
Python de ROS.

## 4. Compilar el workspace de la rayuela

Terminal B:

```bash
source /opt/ros/humble/setup.bash
cd microduck_rl/ros2_microduck
colcon build --symlink-install
source install/setup.bash
ros2 interface show rayuela_msgs/msg/TargetCasilla   # comprueba el mensaje propio
```

## 5. Políticas ONNX

Los `.onnx` no están en git (`.gitignore`). La simulación los busca en
`microduck_rl/policies-v1/`:

| Archivo | Tarea de origen |
|---|---|
| `alpha_walking.onnx` | `Mjlab-Velocity-Flat-MicroDuck` |
| `alpha_standing.onnx` | `Mjlab-StandUp-Flat-MicroDuck` |
| `alpha_sitstand.onnx` | `Mjlab-SitStand-Flat-MicroDuck` |
| `alpha_ground_pick.onnx` | `Mjlab-GroundPick-Flat-MicroDuck` |
| `roulade.onnx` | `Mjlab-Roulade-Flat-MicroDuck` |
| `ball_kick_speed_right.onnx`, `ball_kick_speed_left.onnx` | `Mjlab-BallKickSpeed-{Right,Left}-Flat-MicroDuck` |

Si no están las dos de BallKickSpeed, se usan `ball_kick_right.onnx` y
`ball_kick_left.onnx` (patada de fuerza fija). Para generar cualquiera desde un
run de wandb, en la terminal A:

```bash
uv run scripts/export.py <TASK_ID> --wandb-run-path <entidad/proyecto/run_id>
```

Exporta siempre con `scripts/export.py`: mete el normalizador de observaciones
dentro del ONNX. El organizador del taller compartirá un paquete con estas
políticas ya exportadas.

`scripts/mirror_policy.py` genera la pierna izquierda a partir de la derecha.

## 6. Lanzar la rayuela

Terminal B (con `install/setup.bash` cargado):

```bash
ros2 launch rayuela rayuela.launch.py use_bam_bridge:=true use_viewer:=true enable_teleop:=true
```

- `use_bam_bridge:=true` lanza `sim_worker.py` con el Python del venv
  (`/home/robot/microduck_rl/.venv/bin/python3`) y `bridge_node` en ROS.
- En la ventana de teleop: dígitos 1–9 patean a esa casilla, 0 al cielo.

En otra terminal con ROS cargado:

```bash
ros2 topic list | grep rayuela
ros2 topic echo /rayuela/target_casilla
ros2 topic pub --once /rayuela/behavior_cmd std_msgs/String "data: kick_casilla:7"
rqt_graph
```

## 7. Sin ROS 2 (plan B)

En la terminal A, la misma simulación con teclado y sin ROS:

```bash
uv run scripts/infer_policy.py --walking policies-v1/alpha_walking.onnx \
    --standing policies-v1/alpha_standing.onnx \
    --kick-right policies-v1/ball_kick_speed_right.onnx \
    --kick-speed-commanded --kick-speed 1.0 --new-cmd-obs
```

## Problemas frecuentes

| Síntoma | Causa | Arreglo |
|---|---|---|
| `ModuleNotFoundError: rclpy` dentro de `uv run` | Es lo esperado: rclpy no está en el venv | Usa la terminal B para ROS |
| Errores raros de import en `uv run` | ROS cargado en esa terminal | Abre una terminal limpia sin `source /opt/ros/...` |
| `IndexError` en `select_gpus()` al entrenar en ARM | Torch sin CUDA | Ver la sección aarch64 de `AGENTS.md` |
| `sim_worker.py` no arranca | No existe `/home/robot/microduck_rl/.venv` | Crea el enlace del paso 1 y ejecuta `uv sync` |
| El pato no se mueve en ROS | Faltan ONNX en `policies-v1/` | Paso 5 |
