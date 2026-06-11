import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BBDM(nn.Module):

    def __init__(self, model, T=1000, s=1.0, alpha=0.0, eta=0.01):
        super().__init__()
        self.model = model
        self.T = T
        self.s = s
        self.alpha = alpha 
        self.eta = eta
        self._precompute_schedules()

    def _precompute_schedules(self):
        """Precompute m_t and delta_t schedules (used in q_sample / loss)."""
        T, s = self.T, self.s
        t_vals = torch.arange(1, T + 1, dtype=torch.float64)
        m_t = t_vals / T
        delta_t = 2 * s * (m_t - m_t ** 2)
        self.register_buffer("m_t", m_t.float())
        self.register_buffer("delta_t", delta_t.float())

    def _posterior_coeffs(self, t, t_prev):
        s = self.s
        m_t = (t.float() / self.T).clamp(max=1.0 - 1e-4)
        m_t1 = t_prev.float() / self.T

        delta_t = 2 * s * (m_t - m_t ** 2)
        delta_t1 = 2 * s * (m_t1 - m_t1 ** 2)

        ratio = ((1 - m_t) / (1 - m_t1 + 1e-8)) ** 2
        delta_t_given_t1 = (delta_t - delta_t1 * ratio).clamp(min=0)
        delta_tilde = (delta_t_given_t1 * delta_t1) / (delta_t + 1e-8)

        c_xt = (delta_t1 / (delta_t + 1e-8)) * (1 - m_t) / (1 - m_t1 + 1e-8) + \
               (delta_t_given_t1 / (delta_t + 1e-8)) * (1 - m_t1)
        c_yt = m_t1 - m_t * (1 - m_t) / (1 - m_t1 + 1e-8) * \
               (delta_t1 / (delta_t + 1e-8))
        c_et = (1 - m_t1) * (delta_t_given_t1 / (delta_t + 1e-8))

        return c_xt, c_yt, c_et, delta_tilde

    def q_sample(self, x0, y, t):
        idx = t - 1
        m = self.m_t[idx][:, None, None, None]
        d = self.delta_t[idx][:, None, None, None]
        eps = torch.randn_like(x0)
        x_t = (1 - m) * y + m * x0 + torch.sqrt(d) * eps
        return x_t, eps

    def loss(self, x0, y):
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=x0.device)
        x_t, _ = self.q_sample(x0, y, t)
        pred = self.model(x_t, t)
        return F.mse_loss(pred, y)

    @torch.no_grad()
    def sample(self, planck, S=200):
        y_cond = planck
        x_t = planck + math.sqrt(self.eta) * torch.randn_like(planck)

        steps = torch.linspace(self.T, 1, S).round().long().to(planck.device)

        for i, t_val in enumerate(steps):
            t_prev_val = steps[i + 1] if i + 1 < len(steps) else steps.new_zeros(())

            t = t_val.expand(planck.shape[0])
            t_prev = t_prev_val.expand(planck.shape[0])

            pred = self.model(x_t, t)

            c_x, c_y, c_e, d_tilde = self._posterior_coeffs(t, t_prev)
            c_x = c_x[:, None, None, None]
            c_y = c_y[:, None, None, None]
            c_e = c_e[:, None, None, None]
            d_tilde = d_tilde[:, None, None, None]

            z = torch.randn_like(x_t) if t_val > 1 else torch.zeros_like(x_t)
            x_t = (c_x - c_e) * x_t + c_e * pred + c_y * y_cond \
                + torch.sqrt(d_tilde.clamp(min=0)) * z

        return x_t