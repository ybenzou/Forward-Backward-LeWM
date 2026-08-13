"""Train FBLeWM: official LeWM + Forward/Backward imaginers in one run."""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module import SIGReg
from utils import SaveCkptCallback, count_params, get_column_normalizer, get_img_preprocessor


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite value in {name}: {tensor}")


def fblewm_forward(self, batch, stage, cfg):
    """Official JEPA loss + detached Forward/Backward imaginer losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    lam_f = cfg.loss.forward.weight
    alpha_f = cfg.loss.forward.roll_weight
    lam_b = cfg.loss.backward.weight
    alpha_b = cfg.loss.backward.roll_weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # Official LeWM loss (no stop-grad on targets).
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["official_loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Detached latents for imaginers.
    z = emb.detach()
    p = pred_emb.detach()
    # Alignment: p[:,i] ~ z[:,i+1]; with T=4, p has length 3 -> p1,p2,p3
    # Forward one-step: F(p1)->z2, F(p2)->z3  i.e. F(p[:,0:2]) -> z[:,2:4]
    f_in = p[:, 0:2]
    f_tgt = z[:, 2:4]
    f_pred = self.model.forward_imaginer(f_in)
    output["forward_step_loss"] = (f_pred - f_tgt).pow(2).mean()

    # Forward recursive: F(F(p1)) -> z3
    f_roll = self.model.forward_imaginer(self.model.forward_imaginer(p[:, 0:1]))
    output["forward_roll_loss"] = (f_roll - z[:, 3:4]).pow(2).mean()
    output["forward_loss"] = (
        output["forward_step_loss"] + alpha_f * output["forward_roll_loss"]
    )

    # Backward is selected by loss.backward.target:
    #   pred    — unary B(z_{t+1}) -> p_t
    #   encoder — unary B(z_{t+1}) -> z_t
    #   now     — conditional g <- B(z0, g) in z-space
    b_target_kind = str(cfg.loss.backward.get("target", "pred"))
    is_cond = bool(getattr(self.model.backward_imaginer, "is_conditional", False))
    if b_target_kind == "now":
        if not is_cond:
            raise ValueError(
                "loss.backward.target=now requires ConditionalLatentImaginer "
                "(set model.backward_imaginer._target_=module.ConditionalLatentImaginer)"
            )
        z0 = z[:, 0:1]
        z1 = z[:, 1:2]
        z2 = z[:, 2:3]
        z3 = z[:, 3:4]
        b_from_z3 = self.model.backward_imaginer(z0, z3)
        b_from_z2 = self.model.backward_imaginer(z0, z2)
        output["backward_step_loss"] = (
            (b_from_z3 - z2).pow(2).mean() + (b_from_z2 - z1).pow(2).mean()
        ) * 0.5
        b_roll = self.model.backward_imaginer(z0, b_from_z3)
        output["backward_roll_loss"] = (b_roll - z1).pow(2).mean()
        b_pred = b_from_z3
        b_tgt = z2
    elif b_target_kind in ("pred", "encoder"):
        if is_cond:
            raise ValueError(
                f"loss.backward.target={b_target_kind!r} is the unary B objective; "
                "use CausalLatentImaginer, or set target=now for B(z_now, z_goal)"
            )
        b_in = torch.stack([z[:, 3], z[:, 2]], dim=1)
        if b_target_kind == "pred":
            b_tgt = torch.stack([p[:, 1], p[:, 0]], dim=1)  # p2, p1
            b_roll_tgt = p[:, 0:1]  # p1
        else:
            b_tgt = torch.stack([z[:, 2], z[:, 1]], dim=1)
            b_roll_tgt = z[:, 1:2]
        b_pred = self.model.backward_imaginer(b_in)
        output["backward_step_loss"] = (b_pred - b_tgt).pow(2).mean()
        b_roll = self.model.backward_imaginer(
            self.model.backward_imaginer(z[:, 3:4])
        )
        output["backward_roll_loss"] = (b_roll - b_roll_tgt).pow(2).mean()
    else:
        raise ValueError(
            "loss.backward.target must be 'pred', 'encoder', or 'now', "
            f"got {b_target_kind!r}"
        )

    output["backward_loss"] = (
        output["backward_step_loss"] + alpha_b * output["backward_roll_loss"]
    )

    output["loss"] = (
        output["official_loss"]
        + lam_f * output["forward_loss"]
        + lam_b * output["backward_loss"]
    )

    # Diagnostics (detached).
    output["forward_out_norm"] = f_pred.detach().float().norm(dim=-1).mean()
    output["forward_tgt_norm"] = f_tgt.detach().float().norm(dim=-1).mean()
    output["backward_out_norm"] = b_pred.detach().float().norm(dim=-1).mean()
    output["backward_tgt_norm"] = b_tgt.detach().float().norm(dim=-1).mean()

    for key in (
        "pred_loss",
        "sigreg_loss",
        "official_loss",
        "forward_step_loss",
        "forward_roll_loss",
        "forward_loss",
        "backward_step_loss",
        "backward_roll_loss",
        "backward_loss",
        "loss",
    ):
        _require_finite(key, output[key])

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k.endswith("_norm")
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


# Legacy joint-train artifacts; never write new runs into these names.
_PROTECTED_OUTPUT_NAMES = frozenset(
    {"fblewm", "fblewm_bp", "fblewm_tworoom", "fblewm_cube"}
)


@hydra.main(version_base=None, config_path="./config/train", config_name="fblewm")
def run(cfg):
    out_name = str(cfg.output_model_name)
    if out_name in _PROTECTED_OUTPUT_NAMES:
        raise ValueError(
            f"Refusing to train into protected output_model_name={out_name!r} "
            "(would overwrite an existing checkpoint tree). "
            "Use a new name, e.g. output_model_name=fblewm_tworoom_v2."
        )
    b_target = str(cfg.loss.backward.get("target", "pred"))
    print(
        f"[FBLeWM] train run output_model_name={out_name} "
        f"backward.target={b_target} "
        f"(checkpoints -> $STABLEWM_HOME/checkpoints/{out_name}/)",
        flush=True,
    )

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    resolve_name = dataset_name
    if cache_dir:
        direct = Path(cache_dir) / dataset_name
        under_datasets = Path(cache_dir) / "datasets" / dataset_name
        if direct.exists():
            resolve_name = str(direct.resolve())
        elif under_datasets.exists():
            resolve_name = str(under_datasets.resolve())

    dataset = swm.data.load_dataset(
        resolve_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    world_model = hydra.utils.instantiate(cfg.model)

    # Parameter counts for metadata.
    n_official = (
        count_params(world_model.encoder)
        + count_params(world_model.predictor)
        + count_params(world_model.action_encoder)
        + count_params(world_model.projector)
        + count_params(world_model.pred_proj)
    )
    n_f = count_params(world_model.forward_imaginer)
    n_b = count_params(world_model.backward_imaginer)
    print(
        f"[FBLeWM] params official={n_official} forward={n_f} backward={n_b} "
        f"total={n_official + n_f + n_b}",
        flush=True,
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(fblewm_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    with open(run_dir / "param_counts.json", "w") as f:
        import json

        json.dump(
            {
                "official": n_official,
                "forward_imaginer": n_f,
                "backward_imaginer": n_b,
                "total": n_official + n_f + n_b,
            },
            f,
            indent=2,
        )

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
