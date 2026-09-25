"""Run a one-footed policy on the OTHER foot by mirroring it left<->right.

The BallKick tasks are exact mirror images: the robot is bilaterally symmetric
(verified — left/right masses identical, inertia origins mirrored to 0.000),
the ball spawns at the kicking foot's side, and the speed command rides in the
twist vx slot, which does not change sign under a left-right reflection. So a
right-foot policy fed a mirrored observation, with its output mirrored back, IS
a left-foot policy — with the original's aim quality, and no training.

Tables come from tasks/symmetry.py. The check that catches an indexing error
immediately: mirroring the absolute HOME pose must return HOME (it does, to
0.0). MirroredSession quacks like an InferenceSession; --export bakes the same
mirror into a standalone .onnx.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from mjlab_microduck.tasks.symmetry import (
    _JOINT_PERM,
    _JOINT_SIGN,
    _OBS_PERM,
    _OBS_SIGN,
)

OBS_PERM = np.array(_OBS_PERM, dtype=np.int64)
OBS_SIGN = np.array(_OBS_SIGN, dtype=np.float32)
ACT_PERM = np.array(_JOINT_PERM, dtype=np.int64)
ACT_SIGN = np.array(_JOINT_SIGN, dtype=np.float32)


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Reflect a 61-D actor observation about the sagittal plane."""
    return obs[..., OBS_PERM] * OBS_SIGN


def mirror_action(action: np.ndarray) -> np.ndarray:
    """Reflect a 14-D joint action back into the real robot's frame."""
    return action[..., ACT_PERM] * ACT_SIGN


class MirroredSession:
    """An ONNX policy seen through a left-right mirror: the caller works
    entirely in the real robot's frame and never sees the flip."""

    def __init__(self, onnx_path: str):
        self._session = ort.InferenceSession(onnx_path)
        self._in = self._session.get_inputs()[0].name
        self._out = self._session.get_outputs()[0].name
        n = self._session.get_inputs()[0].shape[-1]
        if n != len(OBS_PERM):
            raise ValueError(
                f"{onnx_path} takes {n}-D obs, the mirror table is "
                f"{len(OBS_PERM)}-D — layouts must match"
            )

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def run(self, output_names, input_feed):
        obs = next(iter(input_feed.values()))
        action = self._session.run([self._out], {self._in: mirror_obs(obs)})[0]
        return [mirror_action(action)]


def export_mirrored_onnx(src_path: str, dst_path: str) -> str:
    """Bake the mirror INTO an ONNX file, so the result is an ordinary policy
    the runtime loads with no special case.

    Graph surgery, not a re-export: the original weights and the baked
    observation normalizer are untouched, only wrapped:

        obs -> Gather(OBS_PERM) -> Mul(OBS_SIGN) -> [original graph]
            -> Gather(ACT_PERM) -> Mul(ACT_SIGN) -> actions
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(src_path)
    graph = model.graph
    # Keep the graph's input/output NAMES exactly as they were: the runtime
    # reads them once from the first policy it loads and reuses them for every
    # other session, so a file with different names is not a drop-in.
    outer_in, outer_out = graph.input[0].name, graph.output[0].name
    inner_in, inner_out = "mirror_inner_in", "mirror_inner_out"

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == outer_in:
                node.input[i] = inner_in
        for i, name in enumerate(node.output):
            if name == outer_out:
                node.output[i] = inner_out

    graph.initializer.extend([
        numpy_helper.from_array(OBS_PERM, "mirror_obs_perm"),
        numpy_helper.from_array(OBS_SIGN, "mirror_obs_sign"),
        numpy_helper.from_array(ACT_PERM, "mirror_act_perm"),
        numpy_helper.from_array(ACT_SIGN, "mirror_act_sign"),
    ])

    pre = [
        helper.make_node("Gather", [outer_in, "mirror_obs_perm"],
                         ["mirror_obs_g"], axis=1, name="mirror_obs_gather"),
        helper.make_node("Mul", ["mirror_obs_g", "mirror_obs_sign"],
                         [inner_in], name="mirror_obs_mul"),
    ]
    post = [
        helper.make_node("Gather", [inner_out, "mirror_act_perm"],
                         ["mirror_act_g"], axis=1, name="mirror_act_gather"),
        helper.make_node("Mul", ["mirror_act_g", "mirror_act_sign"],
                         [outer_out], name="mirror_act_mul"),
    ]
    # Nodes must stay topologically ordered: the pre-nodes produce the tensor
    # the original graph consumes, so they go first.
    nodes = pre + list(graph.node) + post
    del graph.node[:]
    graph.node.extend(nodes)
    # Stale shape hints for the renamed internal tensors would fail the checker.
    for vi in list(graph.value_info):
        if vi.name in (outer_in, outer_out):
            graph.value_info.remove(vi)

    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    return dst_path


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, "scripts")
    from infer_policy import DEFAULT_POSE

    assert np.allclose(mirror_action(DEFAULT_POSE), DEFAULT_POSE, atol=1e-6)
    assert np.allclose(mirror_obs(mirror_obs(np.arange(61, dtype=np.float32))),
                       np.arange(61, dtype=np.float32), atol=1e-6)
    print("mirror tables OK: HOME is invariant and the mirror is its own inverse")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", nargs=2, metavar=("SRC", "DST"),
                    help="bake the mirror into SRC and write DST")
    args = ap.parse_args()
    if args.export:
        src, dst = args.export
        export_mirrored_onnx(src, dst)
        # The baked file must agree with the runtime wrapper, exactly.
        rng = np.random.default_rng(0)
        baked = ort.InferenceSession(dst)
        wrapped = MirroredSession(src)
        name = baked.get_inputs()[0].name
        err = 0.0
        for _ in range(16):   # the graph is fixed at batch 1
            obs = rng.standard_normal((1, 61)).astype(np.float32)
            a = baked.run(None, {name: obs})[0]
            b = wrapped.run(None, {"obs": obs})[0]
            err = max(err, float(np.abs(a - b).max()))
        assert err < 1e-5, f"baked graph disagrees with the wrapper by {err}"
        print(f"wrote {dst} (max disagreement vs the wrapper: {err:.2e})")
