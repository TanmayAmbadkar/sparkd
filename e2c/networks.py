import torch
from torch import nn
from abstract_interpretation.neural_network import LinearLayer, ReLULayer, TanhLayer, NeuralNetwork

torch.set_default_dtype(torch.float64)

def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)

class Encoder(nn.Module):
    def __init__(self, n_features, reduced_dim):
        super(Encoder, self).__init__()
        self.net = NeuralNetwork([LinearLayer(n_features, 12), TanhLayer(), LinearLayer(12, reduced_dim)])
        # self.net.apply(weights_init)
        self.obs_dim = n_features
        self.z_dim = reduced_dim

    def forward(self, x):
        """
        :param x: observation
        :return: the latent representation z
        """
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, reduced_dim, n_features):
        super(Decoder, self).__init__()
        self.net = NeuralNetwork([LinearLayer(reduced_dim, 12), TanhLayer(), LinearLayer(12, n_features)])
        # self.net.apply(weights_init)
        self.z_dim = reduced_dim
        self.obs_dim = n_features

    def forward(self, z):
        """
        :param z: latent representation
        :return: reconstructed x
        """
        return self.net(z)

class Transition(nn.Module):
    def __init__(self, net, z_dim, u_dim):
        super(Transition, self).__init__()
        self.net = net  # network to output the last layer before predicting A_t, B_t and o_t
        self.net.apply(weights_init)
        self.h_dim = 12
        self.z_dim = z_dim
        self.u_dim = u_dim

        self.fc_A = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim * 2),  # v_t and r_t
            nn.Sigmoid()
        )
        self.fc_A.apply(weights_init)

        self.fc_B = nn.Linear(self.h_dim, self.z_dim * self.u_dim)
        torch.nn.init.orthogonal_(self.fc_B.weight)

        self.fc_o = nn.Linear(self.h_dim, self.z_dim)
        torch.nn.init.orthogonal_(self.fc_o.weight)

    def forward(self, z_bar_t, u_t):
        """
        :param z_bar_t: the reference point (latent state)
        :param u_t: the action taken
        :return: the predicted z^_t+1, A_t, B_t, o_t, and z_t
        """
        h_t = self.net(z_bar_t)
        B_t = self.fc_B(h_t)  # Compute control matrix B_t
        o_t = self.fc_o(h_t)  # Compute offset term o_t

        # Compute matrices v_t and r_t for constructing A_t
        v_t, r_t = self.fc_A(h_t).chunk(2, dim=1)
        v_t = torch.unsqueeze(v_t, dim=-1)
        r_t = torch.unsqueeze(r_t, dim=-2)

        # A_t is identity matrix plus outer product of v_t and r_t
        A_t = torch.eye(self.z_dim).repeat(z_bar_t.size(0), 1, 1).to(z_bar_t.device) + torch.bmm(v_t, r_t)

        # Reshape B_t to have dimensions (batch_size, z_dim, u_dim)
        B_t = B_t.view(-1, self.z_dim, self.u_dim)

        # Predict the next latent state
        
        z_t_next = A_t.bmm(z_bar_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t

        # Return the predicted next state, A_t, B_t, o_t, and z_t
        return z_t_next, A_t, B_t, o_t