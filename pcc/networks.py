import torch
from torch import nn
from abstract_interpretation.neural_network import LinearLayer, ReLULayer, SigmoidLayer, NeuralNetwork
from torch import nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal


torch.set_default_dtype(torch.float64)


def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


class Encoder(nn.Module):
    # P(z_t | x_t) and Q(z^_t+1 | x_t+1)
    def __init__(self, n_features: int, reduced_dim: int):
        super(Encoder, self).__init__()
        self.shared_net = NeuralNetwork([LinearLayer(n_features, 32), ReLULayer(), LinearLayer(32, 16), ReLULayer()])  # Shared layers
        self.fc_mu = NeuralNetwork([LinearLayer(16, reduced_dim)])  # Output mean
        self.fc_logsig = NeuralNetwork([LinearLayer(16, reduced_dim)])  # Output log variance
        self.obs_dim = n_features
        self.z_dim = reduced_dim

    def forward(self, x: torch.Tensor):
        # mean and variance of p(z|x)
        hidden_neurons = self.shared_net(x)
        mean = self.fc_mu(hidden_neurons)
        logstd = self.fc_logsig(hidden_neurons)
        return MultivariateNormalDiag(mean, torch.exp(logstd))


class Decoder(nn.Module):
    # P(x_t+1 | z^_t+1)
    def __init__(self, reduced_dim: int, n_features: int):
        super(Decoder, self).__init__()
        self.net = NeuralNetwork([LinearLayer(reduced_dim, 16), ReLULayer(), LinearLayer(16, 32), ReLULayer()])
        self.fc_mu = NeuralNetwork([LinearLayer(32, n_features)])  # Output mean
        self.fc_logsig = NeuralNetwork([LinearLayer(32, n_features)])  # Output log variance
        # self.net.apply(weights_init),
        self.z_dim = reduced_dim
        self.obs_dim = n_features

    def forward(self, z: torch.Tensor):
        """
        :param z: latent representation
        :return: reconstructed x
        """
        hidden_neurons = self.net(z)
        mean = self.fc_mu(hidden_neurons)
        logstd = self.fc_logsig(hidden_neurons)
        return MultivariateNormalDiag(mean, torch.exp(logstd))



class Dynamics(nn.Module):
    # P(z^_t+1 | z_t, u_t)
    def __init__(self, z_dim: int, u_dim: int, amortized: bool):
        super(Dynamics, self).__init__()
        self.net_hidden = nn.Sequential(nn.Linear(z_dim + u_dim, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU())
        self.net_mean = nn.Linear(20, z_dim)
        self.net_logstd = nn.Linear(20, z_dim)
        if amortized:
            self.net_A = nn.Linear(20, z_dim ** 2)
            self.net_B = nn.Linear(20, u_dim * z_dim)
        else:
            self.net_A, self.net_B = None, None
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.amortized = amortized

    def forward(self, z_t: torch.Tensor, u_t: torch.Tensor):
        z_u_t = torch.cat((z_t, u_t), dim=-1)
        hidden_neurons = self.net_hidden(z_u_t)
        mean = self.net_mean(hidden_neurons) + z_t  # skip connection
        logstd = self.net_logstd(hidden_neurons)
        if self.amortized:
            A = self.net_A(hidden_neurons)
            B = self.net_B(hidden_neurons)
        else:
            A, B = None, None
        return MultivariateNormalDiag(mean, torch.exp(logstd)), A, B


class BackwardDynamics(nn.Module):
    # Q(z_t | z^_t+1, x_t, u_t)
    def __init__(self, z_dim:int, u_dim:int, x_dim:int):
        super(BackwardDynamics, self).__init__()
        self.net_z = nn.Linear(z_dim, 5)
        self.net_u = nn.Linear(u_dim, 5)
        self.net_x = nn.Linear(x_dim, 16)
        self.net_joint_hidden = nn.Sequential(
            nn.Linear(5 + 5 + 16, 16),
            nn.ReLU(),
        )
        self.net_joint_mean = nn.Linear(16, z_dim)
        self.net_joint_logstd = nn.Linear(16, z_dim)
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim

    def forward(self, z_t: torch.Tensor, u_t: torch.Tensor, x_t: torch.Tensor):
        z_t_out = self.net_z(z_t)
        u_t_out = self.net_u(u_t)
        x_t_out = self.net_x(x_t)

        hidden_neurons = self.net_joint_hidden(torch.cat((z_t_out, u_t_out, x_t_out), dim=-1))
        mean = self.net_joint_mean(hidden_neurons)
        logstd = self.net_joint_logstd(hidden_neurons)
        return MultivariateNormalDiag(mean, torch.exp(logstd))

