"""
VDK_Shield — Variational Dynamics & Koopman Shield
====================================================
Upgrades over the original dense-A version:

1. SPECTRAL DIAGONAL KOOPMAN DYNAMICS
   - Replaces the dense A (d×d) matrix with a diagonal complex operator Λ.
   - Λ_i = μ_i + i·ω_i  (learnable, one per latent dimension)
   - Propagation is elementwise: z_{t+1} = Λ ⊙ z_t + B·u   (O(d), not O(d²))
   - The real part μ is clamped to ≤ 0, enforcing stability of every eigenmode.
     (Ref: RoboKoop CoRL 2024, Mondal et al. ICLR 2024)
   - Imaginary part ω captures oscillatory modes; initialised at increasing
     frequencies ω_j = α·j·π to span the spectrum from the start.

2. MODALITY-AGNOSTIC ENCODER
   - An abstract ObservationEncoder base class defines the interface.
   - TabularEncoder: MLP stack, drop-in for MuJoCo state vectors.
   - VisualEncoder: DrQ-style 3-layer CNN → flatten → MLP.
   - VDK_Shield accepts *any* encoder at construction; the rest of the
     pipeline (dynamics, CBF, uncertainty) is completely unchanged.

3. COMPLEX ↔ REAL BOOKKEEPING
   - Internally dynamics live in ℂ^d (stored as torch.complex64).
   - The encoder outputs a real-valued (μ, logvar) pair of size d each.
     We interpret μ as the real part of z and initialise the imaginary part
     to zero at every encode call (the linear dynamics will populate it).
   - Before the CBF head we project back to ℝ^{2d} by concatenating
     [Re(z), Im(z)].  The w vector and safety margin live in this 2d space.

4. UNCERTAINTY PROPAGATION (updated for diagonal Λ)
   - diag(Σ_{t+1}) = |Λ|² ⊙ diag(Σ_t) + ε_proc
   - |Λ_i|² = μ_i² + ω_i²   →  fully vectorised, O(d).
   - The safety-margin projection σ²_safety = (w_re² + w_im²) ⊙ |Λ|² ⊙ σ²_t
     is likewise O(d).

References
----------
- Mondal et al., "Efficient Dynamics Modeling in Interactive Environments
  with Koopman Theory", ICLR 2024.  (diagonal spectral formulation)
- Kumawat et al., "RoboKoop", CoRL 2024.  (stability constraint on Re(Λ),
  contrastive spectral encoder)
- Lyu et al., "Task-Oriented Koopman-Based Control with Contrastive Encoder",
  CoRL 2023.  (end-to-end Koopman + LQR)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  ENCODER INTERFACE & CONCRETE IMPLEMENTATIONS
# ---------------------------------------------------------------------------

class ObservationEncoder(ABC, nn.Module):
    """
    Every encoder must map an observation batch to (mu, logvar) of the
    *real-valued* latent distribution.  Shape contract:

        input  : (B, *obs_shape)
        output : mu      (B, latent_dim)
                 logvar  (B, latent_dim)

    The VDK_Shield interprets mu as Re(z) and sets Im(z) = 0 before feeding
    into the complex dynamics.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, obs: torch.Tensor):  # -> (mu, logvar)
        ...


class TabularEncoder(ObservationEncoder):
    """
    MLP encoder for low-dimensional numerical observations (e.g. MuJoCo
    state vectors).  Architecture: [state_dim → 256 → 256] → (μ, log σ²).
    """

    def __init__(self, state_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.fc_mu     = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, obs: torch.Tensor):
        h       = self.net(obs)
        return self.fc_mu(h), self.fc_logvar(h)


class VisualEncoder(ObservationEncoder):
    """
    CNN encoder for pixel observations.

    Expected input shape: (B, C, H, W) with H=W=84 (standard DMControl).
    Architecture (DrQ-style):
        Conv2d 32×8×8 s4 → ReLU
        Conv2d 64×4×4 s2 → ReLU
        Conv2d 64×3×3 s1 → ReLU  (pad=1 keeps spatial size)
        AdaptiveAvgPool2d(1)      → flatten to 64-dim
        Linear 64 → hidden       → SiLU
        Linear hidden → hidden   → SiLU
        → (μ, log σ²)

    For non-84×84 inputs the AdaptiveAvgPool handles it gracefully.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 3,
        hidden: int = 256,
    ):
        super().__init__(latent_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   # → (B, 64, 1, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(64, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.fc_mu     = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, obs: torch.Tensor):
        # obs : (B, C, H, W)
        h = self.cnn(obs)                        # (B, 64, 1, 1)
        h = h.flatten(start_dim=1)               # (B, 64)
        h = self.mlp(h)                          # (B, hidden)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------------------------------------------------------------------------
# 2.  OPTIONAL DECODER  (for Stage-1 reconstruction / prediction loss)
# ---------------------------------------------------------------------------

class TabularDecoder(nn.Module):
    """Decodes a *real* latent vector back to state space."""

    def __init__(self, latent_dim: int, state_dim: int, hidden: int = 256):
        super().__init__()
        # Input is the 2d real projection [Re(z), Im(z)]
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, z_real: torch.Tensor) -> torch.Tensor:
        return self.net(z_real)


class VisualDecoder(nn.Module):
    """
    Transpose-conv decoder targeting 84×84 output.

    Strategy (standard CURL/DrQ decoder pattern):
        FC projects to (64 channels, 7×7) spatial base.
        Three transposed convolutions upsample:  7→14→28→56.
        A final bilinear interpolate brings it to exactly 84×84.
        This avoids painful kernel/stride arithmetic and is the standard
        approach in modern pixel-based RL decoders.
    """

    def __init__(self, latent_dim: int, out_channels: int = 3, hidden: int = 256):
        super().__init__()
        self._spatial_base  = 7
        self._base_channels = 64

        # FC: [Re(z); Im(z)] -> spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self._base_channels * self._spatial_base ** 2),
            nn.SiLU(),
        )

        # Upsample stack: 7 -> 14 -> 28 -> 56  (then interpolate to 84)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),            # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),            # 14->28
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # 28->56
            nn.Tanh(),   # pixel range [-1, 1] after normalisation
        )

    def forward(self, z_real: torch.Tensor) -> torch.Tensor:
        h = self.fc(z_real)                                            # (B, 64*49)
        h = h.view(h.size(0), self._base_channels,
                   self._spatial_base, self._spatial_base)             # (B,64,7,7)
        h = self.deconv(h)                                             # (B, C, 56, 56)
        # Final resize to target 84×84  (bilinear, cheap)
        h = F.interpolate(h, size=(84, 84), mode='bilinear', align_corners=False)
        return h                                                       # (B, C, 84, 84)


# ---------------------------------------------------------------------------
# 3.  SPECTRAL DIAGONAL KOOPMAN DYNAMICS  (the core upgrade)
# ---------------------------------------------------------------------------

class SpectralKoopmanDynamics(nn.Module):
    """
    Diagonal complex Koopman operator with a control-affine input term.

    State equation (in ℂ^d):
        z_{t+1} = Λ ⊙ z_t  +  B_c · u_t

    where
        Λ_i      = μ_i + i·ω_i          (learnable eigenvalues)
        B_c      = B_re + i·B_im         (learnable, shape d × m)
        μ_i ≤ 0                          (stability constraint, enforced via softplus)

    Parameters are stored as *real* tensors; complex arithmetic is done
    explicitly so that autograd works cleanly with mixed-precision.
    """

    def __init__(self, latent_dim: int, control_dim: int, alpha: float = 0.1):
        """
        Args:
            latent_dim:  d  – number of complex Koopman modes.
            control_dim: m  – action dimension.
            alpha:       frequency initialisation scale  (ω_j = alpha·j·π).
        """
        super().__init__()
        self.latent_dim  = latent_dim
        self.control_dim = control_dim

        # --- Eigenvalue real part  (μ ≤ 0 enforced at forward time) ---
        # Initialise near zero so |Λ| ≈ 1 at start.
        self.mu_raw = nn.Parameter(torch.zeros(latent_dim))   # will be mapped through -softplus

        # --- Eigenvalue imaginary part  (ω, unconstrained) ---
        # Increasing-frequency initialisation: ω_j = alpha · j · π
        omega_init = alpha * torch.arange(1, latent_dim + 1, dtype=torch.float32) * math.pi
        self.omega = nn.Parameter(omega_init)

        # --- Control input matrix  B_c = B_re + i·B_im ---
        self.B_re = nn.Parameter(torch.empty(latent_dim, control_dim))
        self.B_im = nn.Parameter(torch.empty(latent_dim, control_dim))
        nn.init.xavier_normal_(self.B_re)
        nn.init.xavier_normal_(self.B_im)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mu(self) -> torch.Tensor:
        """Real part of eigenvalues, guaranteed ≤ 0."""
        return -F.softplus(self.mu_raw)          # shape (d,)

    @property
    def lambda_modulus_sq(self) -> torch.Tensor:
        """
        |Λ_i|² = μ_i² + ω_i².  Used for variance propagation.
        """
        return self.mu ** 2 + self.omega ** 2    # shape (d,)

    # ------------------------------------------------------------------
    # Forward dynamics
    # ------------------------------------------------------------------

    def propagate(
        self,
        z_re: torch.Tensor,   # (B, d)  – real part of latent
        z_im: torch.Tensor,   # (B, d)  – imaginary part of latent
        u:    torch.Tensor,   # (B, m)  – action
    ):
        """
        One-step linear propagation in the spectral Koopman space.

        Returns
        -------
        z_next_re, z_next_im : (B, d) each
        """
        mu    = self.mu                          # (d,)
        omega = self.omega                       # (d,)

        # Λ ⊙ z   (complex elementwise multiply, broadcast over batch)
        #   (μ + iω)(z_re + i·z_im) = (μ·z_re − ω·z_im) + i·(ω·z_re + μ·z_im)
        Lz_re = mu * z_re - omega * z_im         # (B, d)
        Lz_im = omega * z_re + mu * z_im         # (B, d)

        # B_c · u   (real matrix-vector per component)
        #   (B_re + i·B_im) @ u  →  (B_re @ u) + i·(B_im @ u)
        Bu_re = F.linear(u, self.B_re)           # (B, d)
        Bu_im = F.linear(u, self.B_im)           # (B, d)

        return Lz_re + Bu_re, Lz_im + Bu_im

    def propagate_variance(self, sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        Propagate diagonal covariance through the spectral operator.

            σ²_{next,i} = |Λ_i|² · σ²_{t,i} + ε_proc

        Args:
            sigma_sq: (B, d)  – encoder variance (real part only; we treat
                      the imaginary-part variance as zero at the encode step).

        Returns:
            sigma_sq_next: (B, d)
        """
        eps_proc = 1e-4
        return self.lambda_modulus_sq.unsqueeze(0) * sigma_sq + eps_proc  # (B, d)


# ---------------------------------------------------------------------------
# 4.  VDK_SHIELD  –  the top-level module
# ---------------------------------------------------------------------------

class VDK_Shield(nn.Module):
    """
    Variational Dynamics & Koopman Shield.

    Composes:
        encoder   – ObservationEncoder subclass (tabular OR visual)
        dynamics  – SpectralKoopmanDynamics
        decoder   – optional (TabularDecoder / VisualDecoder) for Stage 1
        cbf_head  – linear safety value  V(z) = w^T [Re(z); Im(z)] + β

    Typical usage
    -------------
    # MuJoCo (tabular)
    enc = TabularEncoder(state_dim=111, latent_dim=32)
    dec = TabularDecoder(latent_dim=32, state_dim=111)
    shield = VDK_Shield(encoder=enc, control_dim=8, decoder=dec)

    # Pixel-based (visual)
    enc = VisualEncoder(latent_dim=64, in_channels=3)
    dec = VisualDecoder(latent_dim=64, out_channels=3)
    shield = VDK_Shield(encoder=enc, control_dim=6, decoder=dec)
    """

    def __init__(
        self,
        encoder:     ObservationEncoder,
        control_dim: int,
        decoder:     Optional[nn.Module] = None,   # optional
        alpha:       float = 0.1,               # freq init scale
    ):
        super().__init__()

        self.latent_dim  = encoder.latent_dim
        self.control_dim = control_dim

        # --- Sub-modules ---
        self.encoder  = encoder
        self.dynamics = SpectralKoopmanDynamics(self.latent_dim, control_dim, alpha)
        self.decoder  = decoder   # may be None

        # --- CBF head lives in ℝ^{2d}  (Re ‖ Im concatenation) ---
        self.w    = nn.Parameter(torch.randn(2 * self.latent_dim, 1) * 0.01)
        self.beta = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Helpers: real ↔ complex bookkeeping
    # ------------------------------------------------------------------

    @staticmethod
    def _to_complex_pair(mu: torch.Tensor):
        """
        Encoder outputs real mu (B, d).  We interpret this as Re(z) and
        set Im(z) = 0.  Returns (z_re, z_im).
        """
        return mu, torch.zeros_like(mu)

    @staticmethod
    def _to_real_flat(z_re: torch.Tensor, z_im: torch.Tensor) -> torch.Tensor:
        """Concatenate [Re(z), Im(z)] → (B, 2d) for the CBF head / decoder."""
        return torch.cat([z_re, z_im], dim=-1)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, obs: torch.Tensor):
        """
        Returns
        -------
        mu     : (B, d)  – mean (= Re(z) at t=0)
        logvar : (B, d)  – log variance of the encoder posterior
        """
        return self.encoder(obs)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Sample z_re ~ N(mu, exp(logvar)).  Im part stays 0.
        Returns (z_re, z_im).
        """
        std   = torch.exp(0.5 * logvar)
        eps   = torch.randn_like(std)
        z_re  = mu + std * eps
        z_im  = torch.zeros_like(z_re)
        return z_re, z_im

    # ------------------------------------------------------------------
    # Decode  (only if decoder is provided)
    # ------------------------------------------------------------------

    def decode(self, z_re: torch.Tensor, z_im: torch.Tensor) -> Optional[torch.Tensor]:
        if self.decoder is None:
            return None
        return self.decoder(self._to_real_flat(z_re, z_im))

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def predict_next(
        self,
        z_re: torch.Tensor,
        z_im: torch.Tensor,
        u:    torch.Tensor,
    ):
        """
        z_{t+1} = Λ ⊙ z_t + B·u

        Returns (z_next_re, z_next_im).
        """
        return self.dynamics.propagate(z_re, z_im, u)

    def predict_variance(self, sigma_sq: torch.Tensor) -> torch.Tensor:
        """Propagate encoder variance through |Λ|²."""
        return self.dynamics.propagate_variance(sigma_sq)

    # ------------------------------------------------------------------
    # Safety / CBF
    # ------------------------------------------------------------------

    def cbf_score(self, z_re: torch.Tensor, z_im: torch.Tensor) -> torch.Tensor:
        """
        V(z) = w^T [Re(z); Im(z)] + β

        Returns (B, 1).
        """
        z_flat = self._to_real_flat(z_re, z_im)   # (B, 2d)
        return z_flat @ self.w + self.beta         # (B, 1)

    def safety_margin_variance(self, sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        Variance of the projected safety value after one dynamics step.

            σ²_safety = Σ_i  (w_re_i² + w_im_i²) · |Λ_i|² · σ²_{t,i}

        This is the scalar uncertainty that M_t is built from in the QP.

        Args:
            sigma_sq : (B, d) – encoder variance.

        Returns:
            (B,) – per-sample safety variance.
        """
        d = self.latent_dim

        # Split w into real and imaginary halves
        w_re = self.w[:d, 0]     # (d,)
        w_im = self.w[d:, 0]     # (d,)

        # |w_i|² (in the real-imag sense)
        w_sq = w_re ** 2 + w_im ** 2                        # (d,)

        # |Λ_i|²
        lam_sq = self.dynamics.lambda_modulus_sq            # (d,)

        # Broadcast over batch
        # σ²_safety = sum_i  w_sq_i * lam_sq_i * sigma_sq_i
        return (w_sq.unsqueeze(0) * lam_sq.unsqueeze(0) * sigma_sq).sum(dim=-1)  # (B,)

    # ------------------------------------------------------------------
    # Convenience: full forward (Stage-1 training)
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, u: torch.Tensor, obs_next: torch.Tensor):
        """
        Returns a dict of everything Stage 1 needs:

            mu, logvar          – encoder stats for obs
            mu_next, logvar_next– encoder stats for obs_next  (ground-truth posterior)
            z_re, z_im          – sampled latent for obs
            z_next_re, z_next_im– predicted next latent (via dynamics)
            recon               – decoded obs (None if no decoder)
            recon_next          – decoded predicted next (None if no decoder)
            sigma_sq_next_pred  – propagated variance
        """
        # Current
        mu, logvar           = self.encode(obs)
        z_re, z_im           = self.reparameterize(mu, logvar)

        # Ground-truth next (for KL target)
        mu_next, logvar_next = self.encode(obs_next)

        # Predicted next via dynamics
        z_next_re, z_next_im = self.predict_next(z_re, z_im, u)

        # Variance propagation
        sigma_sq             = torch.exp(logvar)                       # (B, d)
        sigma_sq_next_pred   = self.predict_variance(sigma_sq)         # (B, d)

        # Reconstruct (if decoder exists)
        recon      = self.decode(z_re,      z_im)
        recon_next = self.decode(z_next_re, z_next_im)

        return dict(
            mu=mu,
            logvar=logvar,
            mu_next=mu_next,
            logvar_next=logvar_next,
            z_re=z_re,
            z_im=z_im,
            z_next_re=z_next_re,
            z_next_im=z_next_im,
            recon=recon,
            recon_next=recon_next,
            sigma_sq_next_pred=sigma_sq_next_pred,
        )


# ---------------------------------------------------------------------------
# 5.  LOSS HELPERS  (Stage 1 & Stage 2)
# ---------------------------------------------------------------------------

def stage1_loss(
    out: dict,
    obs: torch.Tensor,
    obs_next: torch.Tensor,
    lam_kl:  float = 0.1,
    lam_lin: float = 1.0,
    lam_frob: float = 1e-5,
    dynamics: Optional[SpectralKoopmanDynamics] = None,
) -> torch.Tensor:
    """
    Stage-1: Variational Koopman dynamics learning.

    Losses:
        L_rec   – reconstruction of obs  (if decoder present)
        L_pred  – one-step prediction of obs_next  (if decoder present)
        L_KL    – KL between encoder posterior for obs_next and the
                  linear-dynamics prior  N(z_next_pred, σ²_next_pred)
        L_lin   – explicit latent-space linearity: ‖μ_next − z_next_re‖²
        L_frob  – spectral regulariser: penalise |Λ| drifting far from 1
                  (replaces the old ‖A‖_F² term; uses Σ(|Λ_i|−1)²)
    """
    losses = {}

    # --- Reconstruction ---
    if out["recon"] is not None:
        losses["rec"]  = F.mse_loss(out["recon"], obs)
        losses["pred"] = F.mse_loss(out["recon_next"], obs_next)
    else:
        losses["rec"]  = torch.tensor(0.0, device=obs.device)
        losses["pred"] = torch.tensor(0.0, device=obs.device)

    # --- Linearity (latent-space) ---
    # ‖ μ(obs_next) − predicted_Re(z_next) ‖²
    losses["lin"] = F.mse_loss(out["mu_next"], out["z_next_re"])

    # --- KL:  N(μ_next, σ²_next)  ‖  N(z_next_re, σ²_next_pred) ---
    # Both are diagonal Gaussians → closed-form KL.
    logvar_post  = out["logvar_next"]                          # (B, d)  – ground truth
    mu_post      = out["mu_next"]                              # (B, d)

    mu_prior     = out["z_next_re"]                            # (B, d)  – from dynamics
    logvar_prior = torch.log(out["sigma_sq_next_pred"] + 1e-8) # (B, d)

    # KL(post ‖ prior)
    kl = 0.5 * (
        (logvar_prior - logvar_post)
        + (torch.exp(logvar_post) + (mu_post - mu_prior) ** 2)
          / (torch.exp(logvar_prior) + 1e-8)
        - 1.0
    )
    losses["kl"] = kl.mean()

    # --- Spectral regulariser  (penalise deviation of |Λ| from 1) ---
    if dynamics is not None:
        lam_mod = torch.sqrt(dynamics.lambda_modulus_sq + 1e-8)   # (d,)
        losses["frob"] = ((lam_mod - 1.0) ** 2).mean()
    else:
        losses["frob"] = torch.tensor(0.0, device=obs.device)

    total = (
        losses["rec"]
        + losses["pred"]
        + lam_kl   * losses["kl"]
        + lam_lin  * losses["lin"]
        + lam_frob * losses["frob"]
    )
    return total, losses


def stage2_loss(
    V_pred:  torch.Tensor,   # (B, 1)  – CBF score from shield
    y_H:     torch.Tensor,   # (B, 1)  – horizon oracle target
    lam_robust: float = 10.0,
    lam_reg:    float = 0.01,
    w:          Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Stage-2: Robust safety value regression.

    L_mse    – regression onto horizon oracle
    L_robust – asymmetric: punish over-confidence (V > y_H) harder
    L_reg    – ‖w‖² to keep Lipschitz constant low
    """
    losses = {}

    losses["mse"]    = F.mse_loss(V_pred, y_H)
    losses["robust"] = F.relu(V_pred - y_H).mean()   # optimism penalty

    if w is not None:
        losses["reg"] = (w ** 2).mean()
    else:
        losses["reg"] = torch.tensor(0.0, device=V_pred.device)

    total = losses["mse"] + lam_robust * losses["robust"] + lam_reg * losses["reg"]
    return total, losses
