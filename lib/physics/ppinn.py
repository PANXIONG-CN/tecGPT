from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_laplacian(A: torch.Tensor) -> torch.Tensor:
    """Build symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.

    Args:
        A: [N, N] adjacency matrix (non-negative, typically 0/1).

    Returns:
        L: [N, N] symmetric normalized Laplacian.
    """
    deg = A.sum(dim=1).clamp(min=1e-6)
    Dm12 = torch.diag(torch.pow(deg, -0.5))
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    return I - Dm12 @ A @ Dm12


class UncertaintyAggregator(nn.Module):
    """Homoscedastic uncertainty weighting with learned log-variances per term.

    Adds a term: 0.5 * exp(-s) * mean(resid^2) + 0.5 * s, where s=log(sigma^2).
    """

    def __init__(self, names):
        super().__init__()
        self.log_vars = nn.ParameterDict({n: nn.Parameter(torch.zeros(1)) for n in names})

    def add(self, name: str, resid: torch.Tensor) -> torch.Tensor:
        s = self.log_vars[name]
        return 0.5 * torch.exp(-s) * torch.mean(resid ** 2) + 0.5 * s


class PPinnLoss(nn.Module):
    """Pure-physics-driven PINN loss (minimal-intrusion version).

    Pipeline per batch:
      1) y_pred01 (in tec01 domain) -> inverse_transform to TECU -> nondimensional y' = y/TEC_ref
      2) dt' = dt_seconds / T_ref_sec (nondimensional time step)
      3) Compute physics residuals in nondimensional domain and aggregate via uncertainty.

    Supports zero-driver mode (nonneg + ROT + ROTI + diffusion) and optional external drivers.
    """

    def __init__(self,
                 scaler_data,                # MinMax01 scaler used for TEC base channel
                 A: torch.Tensor | None,     # adjacency; if None, diffusion term disabled
                 dt_minutes: int = 120,
                 tec_ref: float = 50.0,
                 t_ref_sec: float = 7200.0,
                 rot_cap_tecu_per_min: float = 0.5,
                 roti_cap_scale: float = 0.7,
                 kappa_nd: float = 0.05,
                 roti_w: int = 3,
                 use_diffusion: bool = True,
                 use_drivers: bool = False,
                 use_dynamic_rot_cap: bool = True,
                 rot_cap_k: float = 0.01):
        super().__init__()
        self.scaler = scaler_data
        self.use_diffusion = bool(use_diffusion)
        self.use_drivers = bool(use_drivers)
        # time scaling
        self.dt_sec = float(dt_minutes) * 60.0
        self.t_ref_sec = float(t_ref_sec)
        self.dt_nd = self.dt_sec / self.t_ref_sec
        # value scaling
        self.tec_ref = float(tec_ref)
        # ROT/ROTI caps in nondimensional units (static base)
        rot_cap_per_sec = float(rot_cap_tecu_per_min) / 60.0
        self.rot_cap_nd = (self.t_ref_sec / self.tec_ref) * rot_cap_per_sec
        self.roti_cap_nd = float(roti_cap_scale) * self.rot_cap_nd
        # dynamic options
        self.use_dynamic_rot_cap = bool(use_dynamic_rot_cap)
        self.rot_cap_base_tecu_per_min = float(rot_cap_tecu_per_min)
        self.roti_cap_scale = float(roti_cap_scale)
        self.rot_cap_k = float(rot_cap_k)
        self.roti_w = max(2, int(roti_w))
        # diffusion coefficient (nondimensional)
        self.kappa_nd = float(kappa_nd)
        # Laplacian (nondimensional, PSD)
        self.L = None
        if A is not None and self.use_diffusion:
            self.L = build_norm_laplacian(A)
        # uncertainty aggregator with base terms present in zero-driver mode
        names = ['nonneg', 'rot', 'roti']
        if self.use_diffusion and self.L is not None:
            names.append('diff')
        if self.use_drivers:
            names.extend(['chem', 'night', 'adv'])
        self.uncert = UncertaintyAggregator(names)

    # ------------ helpers ------------
    @staticmethod
    def _as_BTN1(y):
        # Accept [B,T,N] or [B,T,N,1] and return [B,T,N,1]
        if y.dim() == 4:
            return y
        if y.dim() == 3:
            return y.unsqueeze(-1)
        raise ValueError(f'Unexpected pred/true dims: {y.shape}')

    def _to_nd(self, y01_BTN1: torch.Tensor) -> torch.Tensor:
        # inverse_transform from tec01 -> TECU, then nondimensionalize by TEC_ref
        # inverse_transform supports broadcasting; keep device/dtype
        y_tecu = self.scaler.inverse_transform(y01_BTN1)
        y_nd = y_tecu / self.tec_ref
        return y_nd

    @staticmethod
    def _delta_t(y_BTN1: torch.Tensor) -> torch.Tensor:
        # y: [B,T,N,1] -> dy: [B,T-1,N,1]
        return y_BTN1[:, 1:] - y_BTN1[:, :-1]

    def _rot_roti_nd(self, y_nd_BTN1: torch.Tensor):
        # ROT' = dy'/dt'; ROTI' = std(ROT' over small window)
        dy = self._delta_t(y_nd_BTN1)  # [B,T-1,N,1]
        rot_nd = dy / self.dt_nd
        # sliding std with window = self.roti_w, stride 1
        if rot_nd.shape[1] >= self.roti_w:
            roti_nd = rot_nd.unfold(dimension=1, size=self.roti_w, step=1).contiguous().std(dim=3, unbiased=False)
            # pad both ends to match length of rot_nd along time
            if roti_nd.shape[1] != rot_nd.shape[1]:
                pad_l = (rot_nd.shape[1] - roti_nd.shape[1]) // 2
                pad_r = rot_nd.shape[1] - roti_nd.shape[1] - pad_l
                roti_nd = F.pad(roti_nd, (0, 0, 0, 0, pad_l, pad_r), mode='replicate')
        else:
            roti_nd = torch.zeros_like(rot_nd)
        return rot_nd, roti_nd

    def forward(self,
                y_pred01: torch.Tensor,          # [B,T,N] or [B,T,N,1]
                y_true01: torch.Tensor | None = None,
                drivers: dict | None = None) -> tuple[torch.Tensor, dict]:
        y_pred01 = self._as_BTN1(y_pred01)
        y_nd = self._to_nd(y_pred01)  # [B,T,N,1]
        B, T, N, _ = y_nd.shape

        # (0) nonneg: penalize negative nondimensional TEC (soft hinge)
        loss_nonneg = F.relu(-y_nd).mean()

        # (1) ROT / ROTI caps (soft penalty)
        rot_nd, roti_nd = self._rot_roti_nd(y_nd)
        # dynamic ROT/ROTI caps if drivers available
        dyn_rot_cap_nd = None
        dyn_roti_cap_nd = None
        if self.use_drivers and self.use_dynamic_rot_cap and drivers is not None and ('Dst' in drivers):
            try:
                dst = drivers['Dst']  # [B?,T,N] or [1,T,N]
                if dst.dim() == 3:
                    # TECU/min base cap adjusted by Dst (nT): rot_cap = base * (1 + k * max(0,-Dst)/100)
                    dstn = torch.clamp(-dst, min=0.0)  # positive when Dst<0
                    scale = 1.0 + self.rot_cap_k * (dstn / 100.0)
                    cap_tecu_per_min = self.rot_cap_base_tecu_per_min * scale  # [B?,T,N]
                    cap_per_sec = cap_tecu_per_min / 60.0
                    dyn_rot_cap_nd = (self.t_ref_sec / self.tec_ref) * cap_per_sec  # [B?,T,N]
                    # align to [B,T-1,N,1]
                    if dyn_rot_cap_nd.dim() == 3:
                        dyn_rot_cap_nd = dyn_rot_cap_nd[:, 1:, :].unsqueeze(-1)
                    dyn_roti_cap_nd = self.roti_cap_scale * dyn_rot_cap_nd
            except Exception:
                dyn_rot_cap_nd = None
        # penalties
        if dyn_rot_cap_nd is not None:
            rot_pen = F.relu(rot_nd.abs() - dyn_rot_cap_nd)
            roti_pen = F.relu(roti_nd - dyn_roti_cap_nd)
        else:
            rot_pen = F.relu(rot_nd.abs() - self.rot_cap_nd)
            roti_pen = F.relu(roti_nd - self.roti_cap_nd)

        # (2) diffusion: (y_{t+1}' - y_t') - dt' * kappa' * (-L y_t') ~ 0
        loss_diff = torch.tensor(0.0, device=y_nd.device)
        if self.use_diffusion and self.L is not None and T >= 2:
            y_t = y_nd[:, :-1, :, 0]      # [B,T-1,N]
            y_tp1 = y_nd[:, 1:, :, 0]     # [B,T-1,N]
            Ly = torch.einsum('ij,btj->bti', self.L, y_t)
            diff_res = (y_tp1 - y_t) - self.dt_nd * self.kappa_nd * (-Ly)
            loss_diff = (diff_res ** 2).mean()

        # (3) optional drivers (photochemistry, night, advection proxy)
        loss_chem = torch.tensor(0.0, device=y_nd.device)
        loss_night = torch.tensor(0.0, device=y_nd.device)
        loss_adv = torch.tensor(0.0, device=y_nd.device)
        if self.use_drivers and drivers is not None and T >= 2:
            def _bx(x):
                if x is None:
                    return None
                if x.dim() == 2:  # [B,T]
                    return x.unsqueeze(-1).expand(-1, -1, N)
                if x.dim() == 3:  # [B,T,N]
                    return x
                if x.dim() == 4:  # [B,T,Ny,Nx]
                    return x.flatten(2)
                raise ValueError('driver dim not supported')

            cosz = _bx(drivers.get('cosSZA'))
            f107 = _bx(drivers.get('F107'))
            ae = drivers.get('AE')
            kp = drivers.get('Kp')
            # y_t
            y_t = y_nd[:, :-1, :, 0]
            # chemistry: dy' - dt' * (alpha' * F10.7 * max(cosZ,0) - beta' * y' * (1 + r*AE))
            alpha_p = float(drivers.get('alpha_nd', 1e-2))
            beta_p = float(drivers.get('beta_nd', 1e-2))
            ae_fac = 1.0
            if ae is not None:
                ae_b = _bx(ae)
                ae_fac = (1.0 + 1e-3 * ae_b[:, :-1, :])
            if f107 is not None and cosz is not None:
                P = alpha_p * f107[:, :-1, :].clamp_min(0.0) * cosz[:, :-1, :].clamp_min(0.0)
                Lc = beta_p * y_t * ae_fac
                chem_res = (y_nd[:, 1:, :, 0] - y_t) - self.dt_nd * (P - Lc)
                loss_chem = (chem_res ** 2).mean()
            # night non-increase
            if cosz is not None:
                night_mask = (cosz[:, :-1, :] <= 1e-6).float()
                loss_night = (night_mask * F.relu(y_nd[:, 1:, :, 0] - y_t)).mean()
            # advection proxy (optional)
            if kp is not None and bool(drivers.get('use_adv', False)):
                Ny, Nx = drivers.get('grid_shape', (None, None))
                if Ny and Nx and Ny * Nx == N:
                    y2d = y_t.view(B, -1, Ny, Nx)
                    ypad = F.pad(y2d, (0, 0, 1, 1), mode='replicate')
                    gy = (ypad[:, :, 2:, :] - ypad[:, :, :-2, :])
                    gy = gy.reshape(B, -1, N)
                    kp_b = _bx(kp)[:, :-1, :]
                    vy = 1e-3 * kp_b
                    adv_res = (y_nd[:, 1:, :, 0] - y_t) + self.dt_nd * (vy * gy)
                    loss_adv = (adv_res ** 2).mean()

        # (4) aggregate with uncertainties (nan-safe)
        y_nd = torch.nan_to_num(y_nd)
        rot_pen = torch.nan_to_num(rot_pen)
        roti_pen = torch.nan_to_num(roti_pen)
        total = self.uncert.add('nonneg', F.relu(-y_nd))
        total += self.uncert.add('rot', rot_pen)
        total += self.uncert.add('roti', roti_pen)
        terms = {
            'nonneg': loss_nonneg,
            'rot': rot_pen.mean(),
            'roti': roti_pen.mean(),
        }
        if self.use_diffusion and self.L is not None:
            if 'diff_res' in locals():
                total += self.uncert.add('diff', torch.nan_to_num(diff_res))
            else:
                total += self.uncert.add('diff', torch.zeros_like(rot_pen))
            terms['diff'] = loss_diff
        if self.use_drivers:
            if 'chem_res' in locals():
                total += self.uncert.add('chem', torch.nan_to_num(chem_res))
            if 'night_mask' in locals():
                total += self.uncert.add('night', torch.nan_to_num(night_mask * F.relu(y_nd[:, 1:, :, 0] - y_nd[:, :-1, :, 0])))
            if 'adv_res' in locals():
                total += self.uncert.add('adv', torch.nan_to_num(adv_res))
            terms.update({'chem': loss_chem, 'night': loss_night, 'adv': loss_adv})

        # (optional) include data supervision term via uncertainty
        if y_true01 is not None:
            y_true01 = self._as_BTN1(y_true01)
            resid_data = torch.nan_to_num(y_pred01 - y_true01)  # still in tec01 domain
            if 'data' not in self.uncert.log_vars:
                # register lazily on the right device
                self.uncert.log_vars['data'] = nn.Parameter(torch.zeros(1, device=y_nd.device))
            s = self.uncert.log_vars['data']
            total += 0.5 * torch.exp(-s) * torch.mean(resid_data ** 2) + 0.5 * s
            terms['data'] = torch.mean(resid_data ** 2)

        return total, terms
