import torch
from torch import nn
from abstract_interpretation.neural_network import LinearLayer, ReLULayer, SigmoidLayer, NeuralNetwork
from torch import nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

torch.set_default_dtype(torch.float64)

def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)

def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)

class Encoder(nn.Module):
    # P(z_t | x_t) and Q(z^_t+1 | x_t+1)
    def __init__(self, net_hidden, net_mean, net_logstd, x_dim, z_dim):
        super(Encoder, self).__init__()
        self.net_hidden = net_hidden
        self.net_mean = net_mean
        self.net_logstd = net_logstd
        self.x_dim = x_dim
        self.z_dim = z_dim

    def forward(self, x):
        # mean and variance of p(z|x)
        hidden_neurons = self.net_hidden(x)
        mean = self.net_mean(hidden_neurons)
        logstd = self.net_logstd(hidden_neurons)
        return MultivariateNormalDiag(mean, torch.exp(logstd))


class Decoder(nn.Module):
    # P(x_t+1 | z^_t+1)
    def __init__(self, net_hidden, net_logits, z_dim, x_dim):
        super(Decoder, self).__init__()
        self.net_hidden = net_hidden
        self.net_logits = net_logits
        self.z_dim = z_dim
        self.x_dim = x_dim

    def forward(self, z):
        hidden_neurons = self.net_hidden(z)
        logits = self.net_logits(hidden_neurons)
        return Bernoulli(logits=logits)


class Dynamics(nn.Module):
    # P(z^_t+1 | z_t, u_t)
    def __init__(self, net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized):
        super(Dynamics, self).__init__()
        self.net_hidden = net_hidden
        self.net_mean = net_mean
        self.net_logstd = net_logstd
        self.net_A = net_A
        self.net_B = net_B
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.armotized = armotized

    def forward(self, z_t, u_t):
        z_u_t = torch.cat((z_t, u_t), dim=-1)
        hidden_neurons = self.net_hidden(z_u_t)
        mean = self.net_mean(hidden_neurons) + z_t  # skip connection
        logstd = self.net_logstd(hidden_neurons)
        if self.armotized:
            A = self.net_A(hidden_neurons)
            B = self.net_B(hidden_neurons)
        else:
            A, B = None, None
        return MultivariateNormalDiag(mean, torch.exp(logstd)), A, B


class BackwardDynamics(nn.Module):
    # Q(z_t | z^_t+1, x_t, u_t)
    def __init__(self, net_z, net_u, net_x, net_joint_hidden, net_joint_mean, net_joint_logstd, z_dim, u_dim, x_dim):
        super(BackwardDynamics, self).__init__()
        self.net_z = net_z
        self.net_u = net_u
        self.net_x = net_x
        self.net_joint_hidden = net_joint_hidden
        self.net_joint_mean = net_joint_mean
        self.net_joint_logstd = net_joint_logstd
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim

    def forward(self, z_t, u_t, x_t):
        z_t_out = self.net_z(z_t)
        u_t_out = self.net_u(u_t)
        x_t_out = self.net_x(x_t)

        hidden_neurons = self.net_joint_hidden(torch.cat((z_t_out, u_t_out, x_t_out), dim=-1))
        mean = self.net_joint_mean(hidden_neurons)
        logstd = self.net_joint_logstd(hidden_neurons)
        return MultivariateNormalDiag(mean, torch.exp(logstd))


class PlanarEncoder(Encoder):
    def __init__(self, x_dim=1600, z_dim=2):
        net_hidden = NeuralNetwork([LinearLayer(x_dim, 32), ReLULayer(), LinearLayer(32, 16), ReLULayer()])  # Shared layers
        
        net_mean = NeuralNetwork([LinearLayer(16, z_dim)])
        net_logstd = NeuralNetwork([LinearLayer(16, z_dim)])
        super(PlanarEncoder, self).__init__(net_hidden, net_mean, net_logstd, x_dim, z_dim)


class PlanarDecoder(Decoder):
    def __init__(self, z_dim=2, x_dim=1600):
        net_hidden = NeuralNetwork([LinearLayer(z_dim, 16), ReLULayer(), LinearLayer(16, 32), ReLULayer(), LinearLayer(32, x_dim)])
        
        super(PlanarDecoder, self).__init__(net_hidden, z_dim, x_dim)


class PlanarDynamics(Dynamics):
    def __init__(self, armotized, z_dim=2, u_dim=2):
        net_hidden = nn.Sequential(nn.Linear(z_dim + u_dim, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU())
        net_mean = nn.Linear(20, z_dim)
        net_logstd = nn.Linear(20, z_dim)
        if armotized:
            net_A = nn.Linear(20, z_dim ** 2)
            net_B = nn.Linear(20, u_dim * z_dim)
        else:
            net_A, net_B = None, None
        super(PlanarDynamics, self).__init__(net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized)


class PlanarBackwardDynamics(BackwardDynamics):
    def __init__(self, z_dim=2, u_dim=2, x_dim=1600):
        net_z = nn.Linear(z_dim, 5)
        net_u = nn.Linear(u_dim, 5)
        net_x = nn.Linear(x_dim, 100)
        net_joint_hidden = nn.Sequential(
            nn.Linear(5 + 5 + 100, 100),
            nn.ReLU(),
        )
        net_joint_mean = nn.Linear(100, z_dim)
        net_joint_logstd = nn.Linear(100, z_dim)
        super(PlanarBackwardDynamics, self).__init__(
            net_z, net_u, net_x, net_joint_hidden, net_joint_mean, net_joint_logstd, z_dim, u_dim, x_dim
        )
