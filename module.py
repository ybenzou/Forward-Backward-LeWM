"""Official LeWM modules plus CausalLatentImaginer for FBLeWM."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class ResidualMLPBlock(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear with residual (dropout=0)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h


class CausalLatentImaginer(nn.Module):
    """Deterministic single-latent imaginer: z -> z_next (or previous).

    Accepts any leading shape ending in ``dim`` and preserves those dims.
    Recursion is performed by repeatedly calling this module.
    Used by Forward and by unary Backward (``target=encoder`` / ``pred``).
    """

    is_conditional = False
    is_action_aligned = False
    is_sequential_action = False

    def __init__(
        self,
        dim: int = 192,
        hidden_dim: int = 768,
        depth: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(dim, hidden_dim) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def _forward_features(self, z: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor]:
        """Shared trunk. Returns (leading_shape, features) with features (N, dim)."""
        if z.size(-1) != self.dim:
            raise ValueError(
                f"{type(self).__name__} expects last dim={self.dim}, got {tuple(z.shape)}"
            )
        leading = z.shape[:-1]
        x = z.reshape(-1, self.dim)
        for block in self.blocks:
            x = block(x)
        return leading, x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        leading, x = self._forward_features(z)
        x = self.out_proj(self.out_norm(x))
        return x.reshape(*leading, self.dim)


class ActionAlignedCausalLatentImaginer(CausalLatentImaginer):
    """Unary Forward that jointly predicts next action block and next latent.

    ``forward(z)`` still returns only the next latent so
    ``FBLeWM.imagine_forward`` can recurse unchanged. Training uses
    ``forward_with_action(z) -> (action_hat, latent_hat)``. Action is a
    supervision target, never an input.
    """

    is_conditional = False
    is_action_aligned = True

    def __init__(
        self,
        dim: int = 192,
        hidden_dim: int = 768,
        depth: int = 2,
        action_dim: int = 10,
    ):
        super().__init__(dim=dim, hidden_dim=hidden_dim, depth=depth)
        action_dim = int(action_dim)
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self.action_dim = action_dim
        self.action_head = nn.Linear(dim, action_dim)

    def forward_with_action(self, z: torch.Tensor):
        leading, features = self._forward_features(z)
        latent = self.out_proj(self.out_norm(features)).reshape(*leading, self.dim)
        action = self.action_head(features).reshape(*leading, self.action_dim)
        return action, latent

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward_with_action(z)[1]


class SequentialActionCausalLatentImaginer(CausalLatentImaginer):
    """Sequential action-aligned Forward: A=G(z), z_next=H(z, A).

    ``forward(z)`` still returns only the next latent so
    ``FBLeWM.imagine_forward`` can recurse unchanged. Training uses
    ``predict_action`` / ``transition`` / ``forward_teacher_forced``.
    Predicted action is an imagined-latent conditioner, never executed.
    """

    is_conditional = False
    is_action_aligned = True
    is_sequential_action = True

    def __init__(
        self,
        dim: int = 192,
        hidden_dim: int = 768,
        depth: int = 2,
        action_dim: int = 10,
    ):
        super().__init__(dim=dim, hidden_dim=hidden_dim, depth=depth)
        action_dim = int(action_dim)
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self.action_dim = action_dim
        self.hidden_dim = int(hidden_dim)
        self.action_head = nn.Linear(dim, action_dim)
        self.action_embed = nn.Linear(action_dim, dim)
        self.fuse = nn.Linear(dim * 2, dim)
        self.transition_blocks = nn.ModuleList(
            [ResidualMLPBlock(dim, hidden_dim) for _ in range(depth)]
        )

    def predict_action(self, z: torch.Tensor) -> torch.Tensor:
        leading, features = self._forward_features(z)
        action = self.action_head(features).reshape(*leading, self.action_dim)
        return action.to(dtype=z.dtype)

    def transition(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        leading, state_feat = self._forward_features(z)
        if action.shape[:-1] != leading:
            raise ValueError(
                f"action leading shape {tuple(action.shape[:-1])} "
                f"incompatible with latent {tuple(leading)}"
            )
        if action.size(-1) != self.action_dim:
            raise ValueError(
                f"action last dim must equal imaginer.action_dim={self.action_dim}, "
                f"got {int(action.size(-1))}"
            )
        action = action.reshape(-1, self.action_dim).to(
            device=state_feat.device, dtype=state_feat.dtype
        )
        action_feat = self.action_embed(action)
        fused = self.fuse(torch.cat([state_feat, action_feat], dim=-1))
        for block in self.transition_blocks:
            fused = block(fused)
        latent = self.out_proj(self.out_norm(fused))
        return latent.reshape(*leading, self.dim).to(dtype=z.dtype)

    def forward_teacher_forced(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition(z, action)

    def forward_with_action(self, z: torch.Tensor):
        action_hat = self.predict_action(z)
        latent_hat = self.transition(z, action_hat)
        return action_hat, latent_hat

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward_with_action(z)[1]


class BranchPreservingCausalLatentImaginer(nn.Module):
    """History-conditioned multi-branch Forward: F_m([z_{t-1}, z_t]) -> z_{t+1}.

    Shared trunk plus M linear heads. Recursion keeps branch identity fixed.
    ``forward`` is an alias of ``forward_branches`` and does not accept unary
    ``(..., D)`` latents.
    """

    is_conditional = False
    is_action_aligned = False
    is_sequential_action = False
    is_branch_preserving = True
    history_size = 2

    def __init__(
        self,
        dim: int = 192,
        hidden_dim: int = 768,
        depth: int = 2,
        num_branches: int = 4,
    ):
        super().__init__()
        num_branches = int(num_branches)
        if num_branches < 1:
            raise ValueError(f"num_branches must be >= 1, got {num_branches}")
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_branches = num_branches
        self.history_fuse = nn.Linear(self.dim * self.history_size, self.dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(self.dim, hidden_dim) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(self.dim)
        self.branch_heads = nn.ModuleList(
            [nn.Linear(self.dim, self.dim) for _ in range(num_branches)]
        )

    def _history_features(
        self, history: torch.Tensor
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        """Encode ``(..., 2, D)`` history. Returns ``(leading_shape, features(N, D))``."""
        if history.size(-1) != self.dim:
            raise ValueError(
                f"{type(self).__name__} expects last dim={self.dim}, "
                f"got {tuple(history.shape)}"
            )
        if history.ndim < 2 or history.size(-2) != self.history_size:
            raise ValueError(
                f"{type(self).__name__} expects (..., {self.history_size}, "
                f"{self.dim}), got {tuple(history.shape)}"
            )
        leading = history.shape[:-2]
        x = history.reshape(-1, self.history_size * self.dim)
        x = self.history_fuse(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return leading, x

    def forward_branches(self, history: torch.Tensor) -> torch.Tensor:
        """Predict all branch latents. ``history``: ``(..., 2, D)`` -> ``(..., M, D)``."""
        leading, features = self._history_features(history)
        heads = [head(features) for head in self.branch_heads]
        latents = torch.stack(heads, dim=1)
        return latents.reshape(*leading, self.num_branches, self.dim).to(
            dtype=history.dtype
        )

    def forward_assigned(self, history_by_branch: torch.Tensor) -> torch.Tensor:
        """Apply head ``m`` only to history ``[..., m, :, :]``.

        Input ``(..., M, 2, D)`` -> ``(..., M, D)``.
        """
        if history_by_branch.size(-1) != self.dim:
            raise ValueError(
                f"{type(self).__name__} expects last dim={self.dim}, "
                f"got {tuple(history_by_branch.shape)}"
            )
        if history_by_branch.ndim < 3 or history_by_branch.size(-2) != self.history_size:
            raise ValueError(
                f"{type(self).__name__}.forward_assigned expects "
                f"(..., M, {self.history_size}, {self.dim}), "
                f"got {tuple(history_by_branch.shape)}"
            )
        if history_by_branch.size(-3) != self.num_branches:
            raise ValueError(
                f"{type(self).__name__}.forward_assigned expects M="
                f"{self.num_branches} at dim -3, got {tuple(history_by_branch.shape)}"
            )
        leading = history_by_branch.shape[:-3]
        flat = history_by_branch.reshape(-1, self.num_branches, self.history_size, self.dim)
        n = flat.size(0)
        _, features = self._history_features(
            flat.reshape(n * self.num_branches, self.history_size, self.dim)
        )
        features = features.reshape(n, self.num_branches, self.dim)
        heads = [
            self.branch_heads[m](features[:, m]) for m in range(self.num_branches)
        ]
        latents = torch.stack(heads, dim=1)
        return latents.reshape(*leading, self.num_branches, self.dim).to(
            dtype=history_by_branch.dtype
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_branches(history)


class ConditionalLatentImaginer(nn.Module):
    """Conditional imaginer: out <- B(anchor, g).

    Used by now-B (anchor=z_now), pred_goal, and fixed_bridge
    (anchor=predicted latent P).
    Concatenates ``(anchor, g)`` then maps back to ``dim``.
    """

    is_conditional = True

    def __init__(
        self,
        dim: int = 192,
        hidden_dim: int = 768,
        depth: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.fuse = nn.Linear(dim * 2, dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(dim, hidden_dim) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, z_now: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        if z_now.size(-1) != self.dim or z_goal.size(-1) != self.dim:
            raise ValueError(
                "ConditionalLatentImaginer expects last dim="
                f"{self.dim}, got z_now={tuple(z_now.shape)} z_goal={tuple(z_goal.shape)}"
            )
        z_now = _broadcast_now(z_now, z_goal)
        leading = z_goal.shape[:-1]
        fused = torch.cat([z_now, z_goal], dim=-1).reshape(-1, self.dim * 2)
        x = self.fuse(fused)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(self.out_norm(x))
        return x.reshape(*leading, self.dim)


def _broadcast_now(z_now: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Expand z_now so it matches z_goal's leading shape."""
    if z_now.shape == z_goal.shape:
        return z_now
    while z_now.ndim < z_goal.ndim:
        z_now = z_now.unsqueeze(-2)
    try:
        return z_now.expand_as(z_goal)
    except RuntimeError as exc:
        raise ValueError(
            f"cannot broadcast z_now {tuple(z_now.shape)} to z_goal {tuple(z_goal.shape)}"
        ) from exc
