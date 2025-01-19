import torch

torch.set_default_dtype(torch.float64)

class NormalDistribution:
    def __init__(self, mean, logsig, v=None, r=None, A=None):
        """
        :param mean: mu in the paper
        :param logvar: log of the variance (log(Sigma) in the paper)
        :param v: vector used for constructing A
        :param r: vector used for constructing A
        :param A: transformation matrix (A = I + v^T r if not None)
        If A is not provided, covariance is diag(exp(logvar)).
        If A is provided, covariance = A * diag(exp(logvar)) * A^T.
        """
        self.mean = mean
        self.logsig = logsig
        self.v = v
        self.r = r

        # Compute diagonal covariance matrix exp(logvar)
        self.logvar = torch.exp(2*logsig)


    @staticmethod
    def KL_divergence(q_z_next_pred, q_z_next):
        """
        :param q_z_next_pred: q(z_{t+1} | z_bar_t, q_z_t, u_t) using the transition
        :param q_z_next: q(z_t+1 | x_t+1) using the encoder
        :return: KL divergence between two distributions
        """
        mu_0 = q_z_next_pred.mean
        mu_1 = q_z_next.mean
        sigma_0 = torch.exp(q_z_next_pred.logsig)
        sigma_1 = torch.exp(q_z_next.logsig)
        v = q_z_next_pred.v
        r = q_z_next_pred.r
        k = float(q_z_next_pred.mean.size(1))

        sum = lambda x: torch.sum(x, dim=1)

        KL = 0.5 * torch.mean(sum((sigma_0**2 + 2*sigma_0**2*v*r) / sigma_1**2)
                              + sum(r.pow(2) * sigma_0**2) * sum(v.pow(2) / sigma_1**2)
                              + sum(torch.pow(mu_1-mu_0, 2) / sigma_1**2) - k
                              + 2 * (sum(q_z_next.logvar - q_z_next_pred.logvar) - torch.log(1 + sum(v*r)))
                              )
        return KL

