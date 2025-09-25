from __future__ import annotations

import numpy as np
import torch


class DriversStore:
    """Load drivers prepared by tools/prepare_drivers_omni.py and slice by global time index.

    This is optional; for zero-driver PPINN, you do not need this.
    """

    def __init__(self, npz_path: str, device='cpu'):
        self.device = device
        self.grid_shape = (71, 73)
        self.synthetic = False
        try:
            z = np.load(npz_path)
            self.time = z['time']                  # [T]
            self.AE = z['AE'].astype('float32')
            self.Dst = z['Dst'].astype('float32')
            self.Kp = z['Kp'].astype('float32')
            self.F107 = z['F107'].astype('float32')
            self.cosZ = z['cosSZA'].astype('float32') if 'cosSZA' in z.files else None
        except Exception:
            # Fallback: synthetic zeros drivers (safe for smoke tests)
            self.synthetic = True
            self.time = np.arange(0, 10_000, dtype=np.int64)
            self.AE = np.zeros((self.time.shape[0],), dtype=np.float32)
            self.Dst = np.zeros_like(self.AE)
            self.Kp = np.zeros_like(self.AE)
            self.F107 = np.zeros_like(self.AE)
            self.cosZ = np.zeros((self.time.shape[0], self.grid_shape[0]*self.grid_shape[1]), dtype=np.float32)

    def slice_bt(self, start_idx: int, T: int, N: int) -> dict:
        s, e = int(start_idx), int(start_idx + T)
        # Safe slicing with padding if out-of-range
        def _slice_pad(arr1d):
            L = arr1d.shape[0]
            if s < 0:
                s0 = 0
            else:
                s0 = s
            e0 = min(e, L)
            core = arr1d[s0:e0]
            if core.shape[0] < T:
                pad_len = T - core.shape[0]
                pad_val = core[-1] if core.shape[0] > 0 else (arr1d[0] if L > 0 else 0.0)
                core = np.concatenate([core, np.full((pad_len,), pad_val, dtype=np.float32)], axis=0)
            return torch.from_numpy(core.astype('float32')).to(self.device)

        def _slice_pad_2d(arr2d):
            L = arr2d.shape[0]
            s0 = max(0, s)
            e0 = min(e, L)
            core = arr2d[s0:e0, :N]
            if core.shape[0] < T:
                pad_len = T - core.shape[0]
                pad_row = core[-1:] if core.shape[0] > 0 else np.zeros((1, min(N, arr2d.shape[1])), dtype=np.float32)
                pad_block = np.repeat(pad_row, pad_len, axis=0)
                core = np.concatenate([core, pad_block], axis=0)
            return torch.from_numpy(core.astype('float32')).to(self.device)

        AE = _slice_pad(self.AE).view(1, T, 1).expand(1, T, N)
        Dst = _slice_pad(self.Dst).view(1, T, 1).expand(1, T, N)
        Kp = _slice_pad(self.Kp).view(1, T, 1).expand(1, T, N)
        F107 = _slice_pad(self.F107).view(1, T, 1).expand(1, T, N)
        out = {
            'AE': AE,
            'Dst': Dst,
            'Kp': Kp,
            'F107': F107,
            'grid_shape': self.grid_shape,
            'alpha_nd': 1e-2, 'beta_nd': 1e-2, 'use_adv': False,
        }
        if self.cosZ is not None:
            out['cosSZA'] = _slice_pad_2d(self.cosZ).view(1, T, N)
        return out
