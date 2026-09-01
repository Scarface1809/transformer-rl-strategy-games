from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from replay_buffer import ReplayBuffer, TrajectoryExample
from config import TrainingConfig
from envs.core.enums import Nation
from envs.core.entities import Action
from envs.env import SimpleHispaniaEnv

if TYPE_CHECKING:
    from metrics import MetricsCollector


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    config: TrainingConfig,
    env: SimpleHispaniaEnv,
    device: str,
    num_epochs: int,
    batch_size: int,
    optimizer_step_start: int = 0,
) -> tuple[list[Dict[str, float]], int]:
    model.train()
    playing_nations: list[Nation] = list(env.state.playing_nations)
    nation_to_idx: Dict[Nation, int] = {n: i for i, n in enumerate(playing_nations)}
    
    epoch_logs: list[Dict[str, float]] = []

    optimizer_step: int = optimizer_step_start

    for epoch in range(num_epochs):
        batches: list[list[TrajectoryExample]] = buffer.epoch_batches(batch_size)

        policy_sum: float = 0.0
        value_sum: float = 0.0
        grad_norm_sum: float = 0.0
        n_batches: int = 0

        for batch in batches:
            if not batch:
                print("Warning: Empty batch encountered during training. Skipping this batch.")
                continue

            policy_losses: list[torch.Tensor] = []
            values_list: list[torch.Tensor] = []
            value_targets_rows: list[list[float]] = []

            for i, example in enumerate(batch):
                # Forward pass
                out: Dict = model(
                    example.global_feats.to(device),
                    example.tile_feats.to(device),
                    example.unit_feats.to(device),
                    masks=example.masks,
                )

                has_nan = False
                for k, v in out.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        print(f"NaN in {k}")
                        has_nan = True

                if has_nan:
                    print(f"Batch example: {i}")
                    print(f"Acting nation: {example.acting_nation}")
                    print(f"Policy targets: {len(example.pi)} actions")
                    print(f"Value: {example.value}")
                
                # Value head output
                v: torch.Tensor = out["value"].squeeze(0)
                values_list.append(v)

                # Policy loss:
                loss_policy: torch.Tensor = torch.tensor(0.0, device=device)
                for action, p in example.pi.items():
                    logp = _log_prob_action(model=model, out=out, action=action, index_to_unit_id=example.index_to_unit_id)
                    loss_policy += -p * logp
                policy_losses.append(loss_policy)

                # VP gain target for this state across all playing nations
                value_targets_rows.append(
                    [
                        float(example.value.get(nation, 0.0))
                        for nation in playing_nations
                    ]
                )

            # Stack for batch processing
            policy_losses_batch: torch.Tensor = torch.stack(policy_losses)
            values_batch: torch.Tensor = torch.stack(values_list)

            targets_tensor: torch.Tensor = torch.tensor(
                value_targets_rows, device=device, dtype=torch.float32
            )

            # Loss components: keep it simple and standard
            loss_policy: torch.Tensor = policy_losses_batch.mean()
            loss_value: torch.Tensor = F.mse_loss(
                values_batch,
                targets_tensor,
            )

            # Total: policy + weighted value
            loss_total: torch.Tensor = loss_policy + config.value_coef * loss_value

            # Backward pass
            # Compute gradient norms per-loss (policy and value) using autograd.grad
            params = [p for p in model.parameters() if p.requires_grad]
            try:
                grads_policy = torch.autograd.grad(
                    loss_policy, params, retain_graph=True, allow_unused=True
                )
            except RuntimeError:
                grads_policy = [None] * len(params)

            try:
                grads_value = torch.autograd.grad(
                    loss_value, params, retain_graph=True, allow_unused=True
                )
            except RuntimeError:
                grads_value = [None] * len(params)

            def _norm_from_grads(grads):
                total = 0.0
                for g in grads:
                    if g is None:
                        continue
                    try:
                        total += float(torch.norm(g).item()) ** 2
                    except Exception:
                        continue
                return total ** 0.5

            grad_norm_policy = _norm_from_grads(grads_policy)
            grad_norm_value = _norm_from_grads(grads_value)

            # Now perform the actual parameter update from the combined loss
            optimizer.zero_grad()
            loss_total.backward()

            # Compute raw grad norm (after backward, before clipping)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm_raw = total_norm ** 0.5

            grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.max_grad_norm,
            )
            optimizer.step()

            optimizer_step += 1

            # Accumulate stats
            policy_sum += loss_policy.item()
            value_sum += loss_value.item()
            grad_norm_sum += float(grad_norm_clipped.item() if isinstance(grad_norm_clipped, torch.Tensor) else grad_norm_clipped)
            n_batches += 1

        # Log epoch stats
        epoch_log: Dict[str, float] = {
            "epoch": epoch,
            "policy": policy_sum / max(n_batches, 1),
            "value": value_sum / max(n_batches, 1),
            "grad_norm": grad_norm_sum / max(n_batches, 1),
        }
        epoch_logs.append(epoch_log)

    return epoch_logs, optimizer_step

def _log_prob_action(
    model: torch.nn.Module,
    out: Dict,
    action: Action,
    index_to_unit_id: list[int],
) -> torch.Tensor:
    # Reconstruct mapping: unit_id -> row index in unit_feats/unit_logits
    unit_id_to_index = {
        uid: idx
        for idx, uid in enumerate(index_to_unit_id)
    }

    action_type_logp = model.checked_log_softmax(out["action_type_logits"], name="action_type",).squeeze(0)

    logp = action_type_logp[action.type.value]

    # Lookup the unit row once (if this action uses a unit)
    mapped_index: int | None = None
    if action.unit_id is not None:
        mapped_index = unit_id_to_index.get(action.unit_id)
        if mapped_index is None:
            return torch.tensor(float("-inf"), device=logp.device)

        unit_logp = model.checked_log_softmax(out["unit_logits"], name="unit",).squeeze(0)

        logp = logp + unit_logp[mapped_index]

    # Unit type head
    if getattr(action, "unit_type", None) is not None:
        unit_type_logp = model.checked_log_softmax(out["unit_type_logits"], name="unit_type",).squeeze(0)

        logp = logp + unit_type_logp[action.unit_type.value]

    # Tile head
    if action.target_tile is not None:
        tile_logp = model.checked_log_softmax(out["tile_logits"], name="tile",).squeeze(0)

        unit_index = mapped_index if mapped_index is not None else 0

        logp = logp + tile_logp[
            action.type.value,
            unit_index,
            action.target_tile,
        ]

    return logp


#TODO: Implement proper batching ight now i do forawrd model everytime. i should use the batching on my transformer correctly need to do a proper padding for my units.