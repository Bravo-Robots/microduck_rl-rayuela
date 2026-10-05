# Instalación: Microduck RL + rayuela (ROS 2)

Pasos para dejar el proyecto listo antes del taller. Qué es cada tarea, cada
script y cómo funciona la rayuela por dentro está en
[README_PROYECTO.md](README_PROYECTO.md) (en inglés); las reglas para diseñar
recompensas y crear tareas nuevas, en [AGENTS.md](AGENTS.md).

Se usan **dos entornos Python separados**, cada uno en su propia terminal:

- **Terminal A, venv del proyecto** (`uv`, Python 3.12): entrenar, exportar y
  simular con los actuadores BAM.
- **Terminal B, ROS 2 Humble** (Python 3.10 del sistema): los nodos de la rayuela.

No cargues ROS en la terminal A: su `PYTHONPATH` se cuela en el venv. Si tu
`~/.bashrc` hace `source /opt/ros/humble/setup.bash`, empieza la terminal A con
`unset PYTHONPATH`.

## 0. Requisitos

- Ubuntu 22.04 (la plataforma de ROS 2 Humble).
- GPU NVIDIA **solo para entrenar**; la rayuela y `infer_policy.py` corren en CPU.
- Unos 10 GB libres.

## 1. Clonar

```bash
git clone https://github.com/dasanplaen-cmyk/microduck_rl
cd microduck_rl
git checkout rayuela
```

La rayuela encuentra el repo sola siempre que compiles `ros2_microduck/` dentro
del clon (paso 4). Si no, exporta `MICRODUCK_RL_ROOT=/ruta/a/microduck_rl`.

## 2. Venv del proyecto (terminal A)

```bash
unset PYTHONPATH                                  # solo si tu ~/.bashrc carga ROS
curl -LsSf https://astral.sh/uv/install.sh | sh   # instala uv
exec $SHELL
uv sync                                           # crea .venv/ con Python 3.12
```

En máquinas ARM (DGX Spark, Jetson) exporta antes `UV_HTTP_TIMEOUT=600`.

Comprobación:

```bash
uv run list-envs        # debe listar Mjlab-BallKickSpeed-*
uv run --with pytest pytest tests/test_ball_kick_speed_cfg.py tests/test_rayuela_paths.py
uv run train Mjlab-BallKickSpeed-Right-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5     # smoke test, necesita GPU
```

## 3. ROS 2 Humble (terminal B)

Instala Humble siguiendo la guía oficial:
https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

Después, lo que usa la rayuela además de ROS:

```bash
sudo apt install -y python3-colcon-common-extensions python3-numpy python3-opencv
```

## 4. Compilar la rayuela (terminal B)

```bash
source /opt/ros/humble/setup.bash
cd microduck_rl/ros2_microduck
colcon build --symlink-install && source install/setup.bash
ros2 interface show rayuela_msgs/msg/TargetCasilla   # comprueba el mensaje propio
```

## 5. Políticas ONNX

Los `.onnx` no están en git. Descomprime las del juego en `policies-v1/`, que es
donde las busca la simulación:

```bash
unzip policies-v1.zip -d policies-v1
```

Qué tarea entrenó cada una, y cómo exportar las tuyas, en
[README_PROYECTO.md](README_PROYECTO.md#policies-the-game-loads).

## 6. Lanzar la rayuela (terminal B)

```bash
ros2 launch rayuela rayuela.launch.py use_bam_bridge:=true use_viewer:=true
```

`use_bam_bridge:=true` corre la simulación con el Python del venv
(`.venv/bin/python3`); si tu venv está en otro sitio, añade
`venv_python:=/ruta/al/python3`. Para ver lo que detecta la visión añade
`vis_rect:=true` (tablero rectificado con la pelota), `vis_raw:=true` (cámara
inclinada) y `vertical_view:=true` (tablero en vertical, con el cielo arriba).

El teclado va en otra terminal con ROS cargado:

```bash
cd microduck_rl/ros2_microduck && source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run rayuela teleop_keyboard      # dígitos 1–9 patean a esa casilla, 0 al cielo
```

## 7. Ver las redes (Netron)

```bash
uvx netron policies-v1/ball_kick_speed_right.onnx   # abre el navegador en localhost
```

O arrastra el `.onnx` a https://netron.app (se procesa en tu navegador).

## 8. Sin ROS 2 (plan B, terminal A)

```bash
uv run scripts/infer_policy.py --walking policies-v1/alpha_walking.onnx \
    --standing policies-v1/alpha_standing.onnx \
    --kick-right policies-v1/ball_kick_speed_right.onnx \
    --kick-speed-commanded --kick-speed 1.0 --new-cmd-obs
```

## Problemas frecuentes

| Síntoma | Arreglo |
|---|---|
| `ModuleNotFoundError: rclpy` en `uv run` | Normal, rclpy no está en el venv: usa la terminal B |
| Errores raros de import o `No module named 'lark'` en `uv run` | ROS está cargado en esa terminal: `unset PYTHONPATH` o abre otra |
| `IndexError` en `select_gpus()` al entrenar en ARM | Torch sin CUDA: ver la sección aarch64 de [AGENTS.md](AGENTS.md) |
| `sim_worker.py` no arranca | Falta `.venv`: `uv sync` (paso 2) o `venv_python:=...` |
| `Cannot find the microduck_rl checkout` | Workspace fuera del clon: `export MICRODUCK_RL_ROOT=/ruta/a/microduck_rl` |
| El pato no se mueve | Faltan los ONNX: paso 5 |
