#!/usr/bin/env python3
"""
ONNX conversion script for tienkung_stu_future_cc_stage1.

Usage:
    python save_onnx_tienkung_stu_future_cc_stage1.py \
        --ckpt_path <absolute_path_to_checkpoint>
"""

import os
import sys
import argparse

sys.path.append("../../../rsl_rl")

import torch
import torch.nn as nn

try:
    from termcolor import cprint
except ImportError:
    def cprint(msg, _color=None):
        print(msg)

from rsl_rl.modules.actor_critic_future import ActorFuture, get_activation


class HardwareTienkungStudentFutureNN(nn.Module):
    """Deployment wrapper for TianKung future-student policy."""

    def __init__(
        self,
        num_observations,
        num_motion_observations,
        num_priop_observations,
        num_motion_steps,
        num_future_observations,
        num_future_steps,
        motion_latent_dim,
        future_latent_dim,
        num_actions,
        actor_hidden_dims,
        activation,
        history_latent_dim,
        num_history_steps,
        layer_norm=False,
        tanh_encoder_output=False,
        **kwargs,
    ):
        super().__init__()

        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_motion_observations = num_motion_observations
        self.num_priop_observations = num_priop_observations

        activation = get_activation(activation)
        self.normalizer = None

        self.actor = ActorFuture(
            num_observations=num_observations,
            num_motion_observations=num_motion_observations,
            num_priop_observations=num_priop_observations,
            num_motion_steps=num_motion_steps,
            num_future_observations=num_future_observations,
            num_future_steps=num_future_steps,
            motion_latent_dim=motion_latent_dim,
            future_latent_dim=future_latent_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            history_latent_dim=history_latent_dim,
            num_history_steps=num_history_steps,
            layer_norm=layer_norm,
            tanh_encoder_output=tanh_encoder_output,
            **kwargs,
        )

    def load_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, obs):
        assert obs.shape[1] == self.num_observations, (
            f"Expected {self.num_observations} but got {obs.shape[1]}"
        )
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        return self.actor(obs)


def convert_to_onnx(args):
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        cprint(f"Error: checkpoint file not found: {ckpt_path}", "red")
        return 1

    # TianKung student-future cc-stage1 configuration.
    robot_name = "tienkung"
    num_actions = 30
    history_len = 10

    num_motion_steps = 1
    num_motion_observations = 36
    num_priop_observations = 95

    num_future_steps = 1
    n_future_obs_single = 36
    num_future_observations = num_future_steps * n_future_obs_single

    n_obs_single = num_motion_observations + num_priop_observations
    num_observations = n_obs_single * (history_len + 1) + num_future_observations

    motion_latent_dim = 128
    future_latent_dim = 128
    history_latent_dim = 128
    actor_hidden_dims = [512, 512, 256, 128]
    activation = "silu"

    print("TienKung Student Future CC Stage1 Policy Configuration:")
    print(f"  Robot: {robot_name}")
    print(f"  Actions: {num_actions}")
    print(f"  History length: {history_len}")
    print(f"  Motion observations: {num_motion_observations}")
    print(f"  Proprioceptive observations: {num_priop_observations}")
    print(f"  Future observations: {num_future_observations}")
    print(f"  Single obs size: {n_obs_single}")
    print(f"  Total observations: {num_observations}")
    print(f"  Motion latent dim: {motion_latent_dim}")
    print(f"  Future latent dim: {future_latent_dim}")
    print(f"  History latent dim: {history_latent_dim}")
    print("")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = HardwareTienkungStudentFutureNN(
        num_observations=num_observations,
        num_motion_observations=num_motion_observations,
        num_priop_observations=num_priop_observations,
        num_motion_steps=num_motion_steps,
        num_future_observations=num_future_observations,
        num_future_steps=num_future_steps,
        motion_latent_dim=motion_latent_dim,
        future_latent_dim=future_latent_dim,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        activation=activation,
        history_latent_dim=history_latent_dim,
        num_history_steps=history_len,
        layer_norm=True,
        tanh_encoder_output=False,
        use_history_encoder=True,
        use_motion_encoder=True,
    ).to(device)

    cprint(f"Loading model from: {ckpt_path}", "green")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    missing_keys, unexpected_keys = policy.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    if missing_keys:
        cprint(f"Missing keys when loading actor wrapper: {missing_keys}", "yellow")
    if unexpected_keys:
        cprint(
            f"Ignoring non-actor checkpoint keys: {unexpected_keys[:10]}"
            + (" ..." if len(unexpected_keys) > 10 else ""),
            "yellow",
        )

    policy.load_normalizer(ckpt.get("normalizer"))
    policy.eval()

    obs_input = torch.ones(1, num_observations, device=device)
    cprint(f"Input observation shape: {tuple(obs_input.shape)}", "cyan")

    onnx_path = ckpt_path.replace(".pt", ".onnx")
    torch.onnx.export(
        policy,
        obs_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    cprint(f"ONNX model saved to: {onnx_path}", "green")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tienkung_stu_future_cc_stage1 checkpoint to ONNX"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Absolute path to checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for export, e.g. cpu or cuda:0. Defaults to cuda if available.",
    )
    raise SystemExit(convert_to_onnx(parser.parse_args()))
