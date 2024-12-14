from pyearth import Earth
from pyearth._basis import ConstantBasisFunction, LinearBasisFunction, \
    HingeBasisFunction
from typing import Optional, List, Callable
import numpy as np
import torch
import scipy.stats
from src.env_model import MARSModel, MARSComponent, ResidualEnvModel, get_environment_model
from e2c.e2c_model import E2CPredictor, fit_e2c
from abstract_interpretation.verification import get_constraints 


class MarsE2cModel:
    """
    A model that uses the E2CPredictor to obtain A, B, and c matrices
    and provides a similar interface to MARSModel.
    """
    def __init__(self, e2c_predictor, s_dim=None):
        self.e2c_predictor = e2c_predictor
        self.s_dim = s_dim

    def __call__(self, point,  normalized: bool = False) -> np.ndarray:
        """
        Predict the next state given the current state x and action u.
        """
        
        x_norm = point[:self.s_dim]
        u_norm = point[self.s_dim:]
        # Convert to tensors
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        u_tensor = torch.tensor(u_norm, dtype=torch.float32).unsqueeze(0)

        # Use E2CPredictor to predict next state
        z_t_next = self.e2c_predictor.get_next_state(x_tensor, u_tensor)

        # Predict next latent state
        

        return z_t_next

    def get_matrix_at_point(self, point: np.ndarray, s_dim: int, steps: int = 1, normalized: bool = False):
        """
        Get the linear model at a particular point.
        Returns M and eps similar to the original MARSModel.
        M is such that the model output can be approximated as M @ [x; 1],
        where x is the input state-action vector.

        Parameters:
        - x: The current state.
        - u: The action.
        - s_dim: The dimension of the state.
        - steps: Number of steps to unroll for error estimation.
        - normalized: Whether the input is already normalized.

        Returns:
        - M: The linear approximation matrix.
        - eps: The error bound (uncertainty) associated with this approximation.
        """
        # if not normalized:
            # point = (point - self.inp_means) / self.inp_stds
        x_norm = point[:s_dim]
        u_norm = point[s_dim:]

        # Convert to tensors
        x_tensor = torch.tensor(x_norm).unsqueeze(0).double()
        u_tensor = torch.tensor(u_norm).unsqueeze(0).double()

        # Use E2CPredictor to obtain z_t, A_t, B_t, c_t, z_var
        _, A_t, B_t, c_t = self.e2c_predictor.transition(x_tensor, u_tensor)

        # Convert tensors to numpy arrays
        A_t = A_t.detach().numpy().squeeze(0)
        B_t = B_t.detach().numpy().squeeze(0)
        c_t = c_t.detach().numpy().squeeze(0)

        # Combine into a single matrix M
        M = np.hstack((A_t, B_t, c_t[:, None]))
        eps = np.zeros(s_dim)
        return M, eps

    def __str__(self):
        return "MarsE2cModel using E2CPredictor"

class EnvModel:
    """
    A full environment model including a symbolic model and a neural model.

    This model includes a symbolic (MARS) model of the dynamics, a neural
    model which accounts for dynamics not captured by the symbolic model, and a
    second neural model for the reward function.
    """

    def __init__(
            self,
            mars: MarsE2cModel,
            symb_reward: MARSModel,
            net: ResidualEnvModel,
            reward: ResidualEnvModel,
            use_neural_model: bool,
            observation_space_low,
            observation_space_high):
        """
        Initialize an environment model.

        Parameters:
        mars - A symbolic model.
        net - A neural model for the residuals.
        reward - A neural model for the reward.
        """
        self.mars = mars
        self.symb_reward = symb_reward
        self.net = net
        self.reward = reward
        self.use_neural_model = use_neural_model
        self
        self.observation_space_low = np.array(observation_space_low)
        self.observation_space_high = np.array(observation_space_high)
        

    def __call__(self,
                 state: np.ndarray,
                 action: np.ndarray,
                 use_neural_model: bool = True) -> np.ndarray:
        """
        Predict a new state and reward value for a given state-action pair.

        Parameters:
        state (1D array) - The current state of the system.
        action (1D array) - The action to take

        Returns:
        A tuple consisting of the new state and the reward.
        """
        state = state.reshape(-1, )
        action = action.reshape(-1, )
        inp = np.concatenate((state, action), axis=0)
        symb = self.mars(inp)
        if self.use_neural_model:
            neur = self.net(torch.tensor(inp, dtype=torch.float32)). \
                detach().numpy()
            rew = self.reward(torch.tensor(inp, dtype=torch.float32)).item()
        else:
            neur = np.zeros_like(symb)
            rew = self.symb_reward(inp)[0]
            
        return np.clip(symb + neur, self.observation_space_low, self.observation_space_high), rew

    def get_symbolic_model(self) -> MARSModel:
        """
        Get the symbolic component of this model.
        """
        return self.mars

    def get_residual_model(self) -> ResidualEnvModel:
        """
        Get the residual neural component of this model.
        """
        return self.net

    def get_confidence(self) -> float:
        return self.confidence

    @property
    def error(self) -> float:
        return self.mars.error






def get_environment_model(     # noqa: C901
        input_states: np.ndarray,
        actions: np.ndarray,
        output_states: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        domain,
        seed: int = 0,
        use_neural_model: bool = True,
        arch: Optional[List[int]] = None,
        cost_model: torch.nn.Module = None,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        data_stddev: float = 0.01,
        model_pieces: int = 10,
        latent_dim: int = 4,
        horizon: int = 5,
        e2c_predictor = None) -> EnvModel:
    """
    Get a neurosymbolic model of the environment.

    This function takes a dataset consisting of M sample input states and
    actions together with the observed output states and rewards. It then
    trains a neurosymbolic model to imitate that data.

    An architecture may also be supplied for the neural parts of the model.
    The architecture format is a list of hidden layer sizes. The networks are
    always fully-connected. The default architecture is [280, 240, 200].

    Parameters:
    input_states (M x S array) - An array of input states.
    actions (M x A array) - An array of actions taken from the input states.
    output_states (M x S array) - The measured output states.
    rewards (M array) - The measured rewards.
    arch: A neural architecture for the residual and reward models.
    """

    if e2c_predictor is None:
        e2c_predictor = E2CPredictor(input_states.shape[1], latent_dim, actions.shape[1], horizon = horizon)
    fit_e2c(input_states, actions, output_states, e2c_predictor, e2c_predictor.horizon)

    domain = get_constraints(e2c_predictor.encoder.net, domain)
    lows, highs = domain.calculate_bounds()
    
    input_states = e2c_predictor.transform(input_states)
    output_states = e2c_predictor.transform(output_states)

    
    
    states_mean = np.concatenate((input_states, output_states),
                                 axis=0).mean(axis=0)
    states_std = np.maximum(np.concatenate((input_states, output_states),
                                           axis=0).std(axis=0), 1e-5)
    actions_mean = actions.mean(axis=0)
    actions_std = np.maximum(actions.std(axis=0), 1e-5)
    rewards_mean = rewards.mean()
    rewards_std = np.maximum(rewards.std(), 1e-5)

    print("State stats:", states_mean, states_std)
    print("Action stats:", actions_mean, actions_std)
    print("Reward stats:", rewards_mean, rewards_std)
    
    
    parsed_mars = MarsE2cModel(e2c_predictor, latent_dim)
    
    X = np.concatenate((input_states, actions), axis=1)
    Yh = np.array([parsed_mars(state, normalized=True) for state in X]).reshape(input_states.shape[0], -1)
    
    print("Model estimation error:", np.mean((Yh - output_states)**2))

    
    input_states = (input_states - states_mean) / states_std
    output_states = (output_states - states_mean) / states_std
    actions = (actions - actions_mean) / actions_std
    rewards = (rewards - rewards_mean) / rewards_std

    if policy is not None:
        policy_actions = (actions - actions_mean) / actions_std
        next_policy_actions = (actions - actions_mean) / actions_std

    terms = 20
    # Lower penalties allow more model complexity
    X = np.concatenate((input_states, actions), axis=1)


    
    # Get the maximum distance between a predction and a datapoint
    diff = np.amax(np.abs(Yh - output_states))

    # Get a confidence interval based on the quantile of the chi-squared
    # distribution
    conf = data_stddev * np.sqrt(scipy.stats.chi2.ppf(
        0.9, output_states.shape[1]))
    err = diff + conf
    print("Computed error:", err, "(", diff, conf, ")")
    parsed_mars.error = err

    if use_neural_model:
        # Set up a neural network for the residuals.
        state_action = np.concatenate((input_states, actions), axis=1)
        if arch is None:
            arch = [280, 240, 200]
        arch.insert(0, state_action.shape[1])
        arch.append(latent_dim)
        model = ResidualEnvModel(
            arch,
            np.concatenate((states_mean, actions_mean)),
            np.concatenate((states_std, actions_std)),
            states_mean, states_std)
        model.train()

        # Set up a training environment
        optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        loss = torch.nn.MSELoss()

        data = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(state_action, dtype=torch.float32),
                    torch.tensor(output_states - Yh, dtype=torch.float32)),
                batch_size=128,
                shuffle=True)

        # Train the neural network.
        for epoch in range(100):
            losses = []
            for batch_data, batch_outp in data:
                pred = model(batch_data, normalized=True)
                # Normalize predictions and labels to the range [-1, 1]
                loss_val = loss(pred, batch_outp)
                losses.append(loss_val.item())
                optim.zero_grad()
                loss_val.backward()
                optim.step()
            print("Epoch:", epoch,
                torch.tensor(losses, dtype=torch.float32).mean())

        model.eval()

        # Get a symbolic reward model
    reward_symb = Earth(max_degree=1, max_terms=model_pieces, penalty=1.0,
                        endspan=terms, minspan=terms)
    reward_symb.fit(X, rewards)

    rew_coeffs = reward_symb.coef_
    rew_basis = []
    for fn in reward_symb.basis_:
        if fn.is_pruned():
            continue
        if isinstance(fn, ConstantBasisFunction):
            rew_basis.append(MARSComponent())
        elif isinstance(fn, LinearBasisFunction):
            rew_basis.append(MARSComponent(fn.get_variable()))
        elif isinstance(fn, HingeBasisFunction):
            rew_basis.append(MARSComponent(term=fn.get_variable(),
                                           knot=fn.get_knot(),
                                           negate=fn.get_reverse()))
        else:
            raise Exception("Unrecognized basis function: " + type(fn))
    parsed_rew = MARSModel(
        rew_basis, rew_coeffs, 0.01,
        np.concatenate((states_mean, actions_mean)),
        np.concatenate((states_std, actions_std)),
        rewards_mean[None], rewards_std[None])

    if use_neural_model:
        # Set up a neural network for the rewards
        arch[-1] = 1
        rew_model = ResidualEnvModel(
            arch,
            np.concatenate((states_mean, actions_mean)),
            np.concatenate((states_std, actions_std)),
            rewards_mean[None], rewards_std[None])

        optim = torch.optim.Adam(rew_model.parameters(), lr=1e-5)
        loss = torch.nn.SmoothL1Loss()

        # Set up training data for the rewards
        reward_data = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(state_action, dtype=torch.float32),
                    torch.tensor(rewards[:, None], dtype=torch.float32)),
                batch_size=128,
                shuffle=True)

        rew_model.train()

        # Train the network.
        for epoch in range(100):
            losses = []
            for batch_data, batch_outp in reward_data:
                pred = rew_model(batch_data, normalized=True)
                loss_val = loss(pred, batch_outp)
                losses.append(loss_val.item())
                optim.zero_grad()
                loss_val.backward()
                optim.step()
            print("Epoch:", epoch,
                torch.tensor(losses, dtype=torch.float32).mean())

        rew_model.eval()
    else:
        rew_model, model = None, None

    if policy is not None:
        if cost_model is None:
            cost_model = ResidualEnvModel(
                arch,
                np.concatenate((states_mean, actions_mean)),
                np.concatenate((states_std, actions_std)),
                0.0, 1.0)

        optim = torch.optim.Adam(cost_model.parameters(), lr=1e-4)
        loss = torch.nn.SmoothL1Loss()

        # Set up training data for the cost_model
        cost_data = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(input_states, dtype=torch.float32),
                    torch.tensor(actions, dtype=torch.float32),
                    torch.tensor(policy_actions, dtype=torch.float32),
                    torch.tensor(next_policy_actions, dtype=torch.float32),
                    torch.tensor(costs[:, None], dtype=torch.float32)),
                batch_size=128,
                shuffle=True)

        cost_model.train()

        # Negative weight overestimates the safety critic rather than
        # underestimating
        q_weight = -1.0
        for epoch in range(1):
            losses = []
            for batch_states, batch_acts, batch_pacts, \
                    batch_npacts, batch_costs in cost_data:
                pred = cost_model(torch.cat((batch_states, batch_acts), dim=1))
                main_loss = loss(pred, batch_costs)
                q_cur = cost_model(torch.cat((batch_states, batch_pacts),
                                             dim=1))
                q_next = cost_model(torch.cat((batch_states, batch_npacts),
                                              dim=1))
                q_cat = torch.cat([q_cur, q_next], dim=1)
                q_loss = torch.logsumexp(q_cat, dim=1).mean() * q_weight
                q_loss = q_loss - pred.mean() * q_weight
                loss_val = main_loss + q_loss
                losses.append(loss_val.item())
                optim.zero_grad()
                loss_val.backward()
                optim.step()
            print("Epoch:", epoch,
                  torch.tensor(losses, dtype=torch.float32).mean())

        cost_model.eval()

    # print(symb.summary())
    print(parsed_mars)
    print("Model MSE:", np.mean(np.sum((Yh - output_states)**2, axis=1)))
    print(reward_symb.summary())

    return EnvModel(parsed_mars, parsed_rew, model, rew_model,
                    use_neural_model, lows[:input_states.shape[1]], highs[:input_states.shape[1]]), cost_model