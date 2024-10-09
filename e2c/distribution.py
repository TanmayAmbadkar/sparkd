import torch

torch.set_default_dtype(torch.float64)

class NormalDistribution:
    def __init__(self, mean, logvar, v=None, r=None, A=None):
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
        self.logvar = logvar
        self.v = v
        self.r = r

        # Compute diagonal covariance matrix exp(logvar)
        sigma = torch.diag_embed(torch.exp(logvar))

        if A is None:
            self.cov = sigma
        else:
            # A * diag(exp(logvar)) * A^T
            self.cov = A.bmm(sigma.bmm(A.transpose(1, 2)))

    @staticmethod
    def KL_divergence(q_z_next_pred, q_z_next):
        """
        Computes the KL divergence between two normal distributions.
        :param q_z_next_pred: predicted distribution (q(z_{t+1} | z_bar_t, q_z_t, u_t)) from the transition model
        :param q_z_next: distribution from the encoder (q(z_{t+1} | x_{t+1}))
        :return: KL divergence between the two normal distributions
        """
        mu_0 = q_z_next_pred.mean  # Mean of the predicted distribution
        mu_1 = q_z_next.mean        # Mean of the encoder distribution
        sigma_0 = torch.exp(q_z_next_pred.logvar)  # Variance of the predicted distribution
        sigma_1 = torch.exp(q_z_next.logvar)       # Variance of the encoder distribution
        v = q_z_next_pred.v  # v vector from transition
        r = q_z_next_pred.r  # r vector from transition
        k = float(q_z_next_pred.mean.size(1))  # Dimensionality of the latent space

        # Helper function to sum over the appropriate dimensions
        sum_over_dim = lambda x: torch.sum(x, dim=1)

        # KL divergence formula
        KL = 0.5 * torch.mean(
            sum_over_dim((sigma_0 + 2 * sigma_0 * v * r) / sigma_1)  # Trace term
            + sum_over_dim(r.pow(2) * sigma_0) * sum_over_dim(v.pow(2) / sigma_1)  # Adjustment term
            + sum_over_dim(torch.pow(mu_1 - mu_0, 2) / sigma_1)  # Difference in means
            - k  # Subtract dimensionality
            + 2 * (sum_over_dim(q_z_next.logvar - q_z_next_pred.logvar)  # Log variance terms
            - torch.log(1 + sum_over_dim(v * r)))  # Adjustment for matrix A
        )

        return KL
