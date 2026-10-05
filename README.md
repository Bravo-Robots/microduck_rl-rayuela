# Instalación para el taller: Microduck RL + rayuela (ROS 2)

> **Información adicional del proyecto** (tareas, modelo de actuadores, modelos
> del robot, la rayuela en detalle y cómo publicar una política):
> [README_PROYECTO.md](README_PROYECTO.md), en inglés. Las reglas de diseño de
> recompensas y el flujo para crear tareas nuevas están en [AGENTS.md](AGENTS.md).

Guía paso a paso para dejar listo el proyecto antes del taller. Hay **dos
entornos Python separados a propósito** y conviene no mezclarlos:

| Entorno | Python | Qué contiene | Para qué |
|---|---|---|---|
| venv del proyecto (`.venv/`, gestionado por `uv`) | 3.12 | mjlab, MuJoCo Warp, torch, rsl_rl, onnxruntime, BAM | Entrenar, exportar a ONNX, simular con los actuadores BAM |
| ROS 2 Humble (Python del sistema) | 3.10 | rclpy, mensajes, colcon, OpenCV | Los nodos de la rayuela (visión, control, teleop, puente) |

`rclpy` solo existe para el Python 3.10 de Humble y `bam` exige Python ≥ 3.12,
por eso la simulación corre en el venv (`sim_worker.py`) y habla con ROS por un
socket Unix (`bridge_node.py`, `rayuela_ipc.py`).

**Regla de oro:** usa una terminal para `uv` y otra para ROS 2. Lo ideal es no
poner `source /opt/ros/humble/setup.bash` en tu `~/.bashrc`: el `PYTHONPATH` de
ROS (paquetes de Python 3.10) se cuela en el venv de 3.12. Si tu máquina ya lo
tiene ahí, empieza la terminal del venv con `unset PYTHONPATH`.

## 0. Requisitos

- Ubuntu 22.04 (es la plataforma oficial de ROS 2 Humble).
- GPU NVIDIA con driver reciente **solo para entrenar**. La demo de la rayuela y
  `scripts/infer_policy.py` corren en CPU.
- Unos 10 GB libres (torch + CUDA + ROS 2).

## 1. Clonar el repositorio

```bash
git clone https://github.com/dasanplaen-cmyk/microduck_rl
cd ~/microduck_rl
git checkout rayuela
```

Clona donde quieras: la rayuela encuentra el repo sola
(`ros2_microduck/src/rayuela/rayuela/paths.py`) siempre que compiles
`ros2_microduck/` dentro del clon, como en el paso 4. Si compilas el workspace
en otro sitio, exporta `MICRODUCK_RL_ROOT=/ruta/a/microduck_rl`.

## 2. Entorno Python del proyecto (uv + venv 3.12)

Terminal A, **sin** ROS cargado:

```bash
unset PYTHONPATH                                  # solo si tu ~/.bashrc carga ROS
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
uv run --with pytest pytest tests/test_ball_kick_speed_cfg.py tests/test_rayuela_paths.py # tests en CPU
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
cd ~/microduck_rl/ros2_microduck && colcon build --symlink-install && source install/setup.bash
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

Descomprimir el paquete `policies-v1.zip`, donde estaran las polizas principales
para correr el juego de la rayuela.

Si no están las dos de BallKickSpeed, se usan `ball_kick_right.onnx` y
`ball_kick_left.onnx` (patada de fuerza fija). Para generar cualquiera desde un
run de wandb, en la terminal A:

```bash
uv run scripts/export.py <TASK_ID> --wandb-run-path <entidad/proyecto/run_id>
```

Exporta siempre con `scripts/export.py`: mete el normalizador de observaciones
dentro del ONNX. 

`scripts/mirror_policy.py` genera la pierna izquierda a partir de la derecha.

## 6. Lanzar la rayuela

Terminal B (con `install/setup.bash` cargado):

```bash
ros2 launch rayuela rayuela.launch.py use_bam_bridge:=true use_viewer:=true enable_teleop:=true
```

- `use_bam_bridge:=true` lanza `sim_worker.py` con el Python del venv
  (`<repo>/.venv/bin/python3`) y `bridge_node` en ROS. Si tu venv está en otro
  sitio, añade `venv_python:=/ruta/al/python3`.
- En la ventana de teleop: dígitos 1–9 patean a esa casilla, 0 al cielo.

En otra terminal con ROS cargado:

```bash
cd ~/microduck_rl/ros2_microduck && colcon build --symlink-install && source install/setup.bash
ros2 run rayuela teleop_keyboard
```

## 7. Ver las redes neuronales (Netron)

Para el bloque de PPO se abre la política exportada en
[Netron](https://netron.app), un visor de redes. No instala nada en el venv:

```bash
uvx netron policies-v1/ball_kick_speed_right.onnx   # abre el navegador en localhost
```

O arrastra el `.onnx` a https://netron.app (el archivo se procesa en tu navegador).

## 8. Sin ROS 2 (plan B)

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
| Errores raros de import en `uv run` | ROS cargado en esa terminal | `unset PYTHONPATH` o abre una terminal sin `source /opt/ros/...` |
| `No module named 'lark'` al arrancar pytest | Con ROS cargado, pytest intenta cargar los plugins de pytest de ROS (Python 3.10) | `pyproject.toml` ya los bloquea; si aparece con otro plugin, `unset PYTHONPATH` |
| `IndexError` en `select_gpus()` al entrenar en ARM | Torch sin CUDA | Ver la sección aarch64 de `AGENTS.md` |
| `sim_worker.py` no arranca | No existe `<repo>/.venv` | Ejecuta `uv sync` (paso 2) o pasa `venv_python:=...` |
| `Could not find the microduck_rl checkout` | El workspace de ROS 2 está fuera del clon | `export MICRODUCK_RL_ROOT=/ruta/a/microduck_rl` |
| El pato no se mueve en ROS | Faltan ONNX en `policies-v1/` | Paso 5 |
