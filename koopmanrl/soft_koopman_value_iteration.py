import os
import random
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from tap import Tap
from torch.utils.tensorboard import SummaryWriter

from koopmanrl.environments import DoubleWell, FluidFlow, LinearSystem, Lorenz
from koopmanrl.koopman_observables import monomials
from koopmanrl.koopman_tensor.torch_tensor import (
    KoopmanTensor,
    generate_koopman_tensor,
)
from koopmanrl.utils import (
    create_folder,
    load_and_apply_config,
    make_env,
    resolve_device,
    resolve_dtype,
)

torch.set_default_dtype(torch.float64)
delta = torch.finfo(torch.float64).eps  # 2.220446049250313e-16

# ---------------------------------------------------------------------------
# JSON config loading
# ---------------------------------------------------------------------------

# Maps JSON hyphenated keys to ArgumentParser attribute names.
_CONFIG_KEY_MAP: dict[str, str] = {
    "env-id": "env_id",
    "seed": "seed",
    "learning-rate": "lr",
    "number-of-train-epochs": "num_training_epochs",
    "num-paths": "num_paths",
    "num-steps-per-path": "num_steps_per_path",
    "state-order": "state_order",
    "action-order": "action_order",
    "total-timesteps": "total_timesteps",
}

# Fallback values that reproduce the original hard-coded defaults so that
# omitting --config_file leaves behaviour completely unchanged.
_FALLBACKS: dict[str, object] = {
    "env_id": "LinearSystem-v0",
    "seed": 1,
    "lr": 1e-3,
    "num_training_epochs": 150,
    "num_paths": 100,
    "num_steps_per_path": 300,
    "state_order": 2,
    "action_order": 2,
    "total_timesteps": 50000,
}


class ArgumentParser(Tap):
    exp_name: str = os.path.basename(__file__).rstrip(".py")  # the name of this experiment
    seed: Optional[int] = None  # seed of the experiment; loaded from config if not set (default: 1)
    torch_deterministic: bool = True  # if toggled, `torch.backends.cudnn.deterministic=False` (default: True)
    cuda: bool = False  # if toggled, cuda will be enabled by default (default: True)
    fp32: bool = False  # use float32 instead of float64 (default: False)
    env_id: Optional[str] = None  # id of the environment; loaded from config if not set (default: LinearSystem-v0)
    total_timesteps: Optional[int] = None  # total timesteps; loaded from config if not set (default: 50000)
    gamma: float = 0.99  # the discount factor gamma (default: 0.99)
    batch_size: int = 2**14  # the batch size of sample from the reply memory (default: 2^14 = 16_384)
    lr: Optional[float] = None  # learning rate; loaded from config if not set (default: 0.001)
    alpha: float = 1.0  # entropy regularization coefficient (default: 1.0)
    num_actions: int = 101  # number of actions that the policy can pick from (default: 101)
    num_training_epochs: Optional[int] = None  # training epochs; loaded from config if not set (default: 150)
    batch_scale: int = 1  # increase batch size by this multiple for computing bellman error (default: 1)
    num_paths: Optional[int] = None  # number of paths for dataset; loaded from config if not set (default: 100)
    num_steps_per_path: Optional[int] = None  # steps per path; loaded from config if not set (default: 300)
    state_order: Optional[int] = None  # state monomial order; loaded from config if not set (default: 2)
    action_order: Optional[int] = None  # action monomial order; loaded from config if not set (default: 2)
    regressor: str = "ols"  # Which regressor to use to build the Koopman tensor (default: 'ols')
    config_file: Optional[str] = None  # path to a JSON config file; CLI flags override file values


class DiscreteKoopmanValueIterationPolicy:
    def __init__(
        self,
        env_id,
        gamma,
        alpha,
        dynamics_model: KoopmanTensor,
        all_actions,
        cost,
        seed,
        use_ols=True,
        learning_rate=0.003,
        dt=None,
        device=None,
    ):
        """
        Initialize DiscreteKoopmanValueIterationPolicy.

        Parameters
        ----------
        args
            The command line arguments parsed using argparse.
        gamma : float
            The discount factor of the system.
        alpha : float
            The regularization parameter of the policy (temperature).
        dynamics_model : KoopmanTensor
            The trained Koopman tensor for the system.
        all_actions : array-like
            The actions that the policy can take.
        cost : function
            The cost function of the system. Function must take in states and actions and return scalars.
        use_ols : bool, optional
            Boolean to indicate whether or not to use OLS in computing new value function weights,
            by default True.
        learning_rate : float, optional
            The learning rate of the policy, by default 0.003.
        dt : float, optional
            The time step of the system, by default 1.0.
        load_model : bool, optional
            Boolean indicating whether or not to load a saved model, by default False.

        Returns
        -------
        DiscreteKoopmanValueIterationPolicy
            Instance of the DiscreteKoopmanValueIterationPolicy class.
        """

        # Store env id
        self.env_id = env_id

        # Set settings for determinism
        # self.seed = args.seed
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # torch.backends.cudnn.deterministic = args.torch_deterministic

        # Set up algorithm variables
        self.gamma = gamma
        self.alpha = alpha
        self.dynamics_model = dynamics_model
        self.all_actions = all_actions
        self.cost = cost
        self.start_timestamp = int(time.time())
        self.save_data_path = f"./saved_models/SKVI/{self.env_id}/skvi_chkpts_{seed}_{self.start_timestamp}"
        self.use_ols = use_ols
        self.learning_rate = learning_rate
        self.dt = dt
        if self.dt is None:
            self.dt = 1.0
        self.device = device if device is not None else torch.device("cpu")

        self.discount_factor = self.gamma**self.dt

        # Handle model initialization
        if self.use_ols:
            self.value_function_weights = torch.zeros((self.dynamics_model.phi_dim, 1), device=self.device)
        else:
            self.value_function_weights = torch.zeros(
                (self.dynamics_model.phi_dim, 1), requires_grad=True, device=self.device
            )
            self.value_function_optimizer = torch.optim.Adam([self.value_function_weights], lr=self.learning_rate)

    def load_model(
        self,
        value_function_weights=None,
        trained_model_start_timestamp=None,
        chkpt_epoch_number=None,
    ):
        # If provided, use value function weights
        # Make sure to enable gradient computations if not using OLS
        # Otherwise, load a previously trained model with given start timestamp and epoch number
        if value_function_weights is not None:
            if self.use_ols:
                self.value_function_weights = torch.tensor(value_function_weights)
            else:
                self.value_function_weights = torch.tensor(value_function_weights, requires_grad=True)
        else:
            self.value_function_weights = torch.load(
                f"./saved_models/SKVI/{self.env_id}/skvi_chkpts_{trained_model_start_timestamp}/epoch_{chkpt_epoch_number}.pt"
            )

    def pis(self, xs):
        """
        Compute the probability distribution of actions for a given set of states.

        Parameters
        ----------
        xs : array-like
            2D array of state column vectors.

        Returns
        -------
        array-like
            2D array of action probability column vectors.
        """

        # Compute phi(x) for each x
        phi_xs = self.dynamics_model.phi(xs.T)  # (phi_dim, batch)

        # Compute phi(x') for all ( phi(x), action ) pairs and compute V(x')s (vectorized)
        K_us = self.dynamics_model.K_(self.all_actions)  # (num_actions, phi_dim, phi_dim)
        phi_x_prime_batch = torch.einsum("aij,jb->aib", K_us, phi_xs)  # (num_actions, phi_dim, batch)
        w = self.value_function_weights.squeeze(-1)  # (phi_dim,)
        V_x_prime_batch = torch.einsum("p,apb->ab", w, phi_x_prime_batch)  # (num_actions, batch)

        # Get costs indexed by the action and the state
        costs = self.cost(xs, self.all_actions.T)  # (num_actions, batch)

        # Compute policy distribution
        inner_pi_us_values = -(costs + self.discount_factor * V_x_prime_batch)  # (all_actions.shape[1], xs.shape[1])
        inner_pi_us = inner_pi_us_values / self.alpha  # (all_actions.shape[1], xs.shape[1])
        real_inner_pi_us = torch.real(inner_pi_us)  # (all_actions.shape[1], xs.shape[1])

        # Max trick
        max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0)  # xs.shape[1]
        diff = real_inner_pi_us - max_inner_pi_u

        pi_us = torch.exp(diff) + delta  # (all_actions.shape[1], xs.shape[1])
        Z_x = torch.sum(pi_us, axis=0)  # xs.shape[1]

        return pi_us / Z_x  # (all_actions.shape[1], xs.shape[1])

    def V_phi_x(self, phi_x):
        """
        Compute the value function V(phi_x) for a given observable of the state.

        Parameters
        ----------
        phi_x : array-like
            Column vector of the observable of the state.

        Returns
        -------
        float
            Value function output.
        """

        return self.value_function_weights.T @ phi_x

    def V_x(self, x):
        """
        Compute the value function V(x) for a given state.

        Parameters
        ----------
        x : array-like
            Column vector of the state.

        Returns
        -------
        float
            Value function output.
        """

        return self.V_phi_x(self.dynamics_model.phi(x))

    def discrete_bellman_error(self, batch_size):
        """
        Compute the Bellman error for a batch of samples.

        Parameters
        ----------
        batch_size : int
            Number of samples of the state space used to compute the Bellman error.

        Returns
        -------
        float
            Mean squared Bellman error.
        """

        # Get random sample of xs and phi(x)s from dataset
        x_batch_indices = torch.from_numpy(
            np.random.choice(self.dynamics_model.X.shape[1], batch_size, replace=False)
        ).to(self.device)
        x_batch = self.dynamics_model.X[:, x_batch_indices.long()]  # (X.shape[0], batch_size)
        phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices.long()]  # (dim_phi, batch_size)

        # Compute V(x) for all phi(x)s
        V_xs = self.V_phi_x(phi_x_batch)  # (1, batch_size)

        # Get costs indexed by the action and the state
        costs = self.cost(x_batch.T, self.all_actions.T)  # (num_actions, batch)

        # Compute phi(x') for all ( phi(x), action ) pairs and compute V(x')s (vectorized)
        K_us = self.dynamics_model.K_(self.all_actions)  # (num_actions, phi_dim, phi_dim)
        phi_x_prime_batch = torch.einsum("aij,jb->aib", K_us, phi_x_batch)  # (num_actions, phi_dim, batch)
        w = self.value_function_weights.squeeze(-1)  # (phi_dim,)
        V_x_prime_batch = torch.einsum("p,apb->ab", w, phi_x_prime_batch)  # (num_actions, batch)

        # Compute policy distribution
        inner_pi_us_values = -(costs + self.discount_factor * V_x_prime_batch)  # (all_actions.shape[1], batch_size)
        inner_pi_us = inner_pi_us_values / self.alpha  # (all_actions.shape[1], batch_size)
        real_inner_pi_us = torch.real(inner_pi_us)  # (all_actions.shape[1], batch_size)

        # Max trick
        max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0)  # (batch_size,)
        diff = real_inner_pi_us - max_inner_pi_u  # (all_actions.shape[1], batch_size)

        # Softmax distribution
        pi_us = torch.exp(diff) + delta  # (all_actions.shape[1], batch_size)
        Z_x = torch.sum(pi_us, axis=0)  # (batch_size,)
        pis_response = pi_us / Z_x  # (all_actions.shape[1], batch_size)

        # Compute log probabilities
        log_pis = torch.log(pis_response)  # (all_actions.shape[1], batch_size)

        # Compute expectation
        expectation_u = torch.sum(
            (costs + self.alpha * log_pis + self.discount_factor * V_x_prime_batch) * pis_response,
            axis=0,
        ).reshape(1, -1)  # (1, batch_size)

        # Compute mean squared error
        squared_error = torch.pow(V_xs - expectation_u, 2)  # (1, batch_size)
        mean_squared_error = torch.mean(squared_error)  # scalar

        return mean_squared_error

    def get_action_and_log_prob(self, x, sample_size=None, is_greedy=False):
        """
        Compute the action given the current state.

        Parameters
        ----------
        x : array_like
            State of the system as a column vector.
        sample_size : int or None, optional
            How many actions to sample. None gives 1 sample.
        is_greedy : bool, optional
            If True, select the action with maximum probability greedily.
            If False, sample actions based on the probability distribution.

        Returns
        -------
        actions : array
            Selected actions from the value iteration policy.
        log_probabilities : array
            Logarithm of the probabilities corresponding to the selected actions.

        Notes
        -----
        This function computes the action to be taken given the current state `x`.
        If `sample_size` is provided, it selects multiple actions based on the
        policy distribution. If `is_greedy` is True, it selects the action with
        the maximum probability greedily; otherwise, it samples actions according
        to the probability distribution defined by the policy.
        """

        if sample_size is None:
            sample_size = self.dynamics_model.u_column_dim

        pis_response = self.pis(x)[:, 0]

        if is_greedy:
            selected_indices = torch.ones(sample_size, dtype=torch.int8, device=self.device) * torch.argmax(
                pis_response
            )
        else:
            selected_indices = torch.from_numpy(
                np.random.choice(np.arange(len(pis_response)), size=sample_size, p=pis_response.detach().cpu().numpy())
            ).to(self.device)

        return (
            self.all_actions[0][selected_indices.long()],
            torch.log(pis_response[selected_indices.long()]),
        )

    def get_action(self, x, sample_size=None, is_greedy=False):
        """
        Compute the action given the current state.

        Parameters
        ----------
        x : array_like
            State of the system as a column vector.
        sample_size : int or None, optional
            How many actions to sample. None gives 1 sample.
        is_greedy : bool, optional
            If True, select the action with maximum probability greedily.
            If False, sample actions based on the probability distribution.

        Returns
        -------
        action : array
            Selected action(s) from the value iteration policy.

        Notes
        -----
        This function computes the action to be taken given the current state `x`.
        If `sample_size` is provided, it selects multiple actions based on the
        policy distribution. If `is_greedy` is True, it selects the action with
        the maximum probability greedily; otherwise, it samples actions according
        to the probability distribution defined by the policy.
        """

        return self.get_action_and_log_prob(x, sample_size, is_greedy)[0]

    def train(
        self,
        training_epochs,
        batch_size=2**14,
        batch_scale=1,
        epsilon=1e-2,
        gammas=[],
        gamma_increment_amount=0.0,
        how_often_to_chkpt=250,
    ):
        """
        Train the value iteration model.

        Parameters
        ----------
        training_epochs : int
            Number of epochs for which to train the model.
        batch_size : int, optional
            Sample of states for computing the value function weights.
        batch_scale : int, optional
            Scale factor that is multiplied by batch_size for computing the Bellman error.
        epsilon : float, optional
            End the training process if the Bellman error < epsilon.
        gammas : list of float, optional
            Array of gammas to try in case of iterating on the discounting factors.
        gamma_increment_amount : float, optional
            Amount by which to increment gamma until it reaches 0.99. If 0.0, no incrementing.
        how_often_to_chkpt : int, optional
            Number of training iterations to do before saving model weights and training data.

        Notes
        -----
        This function updates the class parameters without returning anything.
        After running this function, you can call `policy.get_action(x)` to get an action using the trained policy.
        """

        # Create all directories needed to save data
        create_folder(f"{self.save_data_path}/training_data")

        # Save original gamma and set gamma to first in array
        original_gamma = self.gamma
        if len(gammas) > 0:
            self.gamma = gammas[0]
        self.discount_factor = self.gamma**self.dt

        # Compute initial Bellman error
        BE = self.discrete_bellman_error(batch_size=batch_size * batch_scale).detach().cpu().numpy()
        bellman_errors = [BE]
        print(f"Initial Bellman error: {BE}")

        step = 0
        gamma_iteration_condition = self.gamma <= 0.99 or self.gamma == 1
        while gamma_iteration_condition:
            print(f"gamma for iteration #{step + 1}: {self.gamma}")
            self.discount_factor = self.gamma**self.dt

            for epoch in range(training_epochs):
                # Get random batch of X and Phi_X from tensor training data
                x_batch_indices = torch.from_numpy(
                    np.random.choice(self.dynamics_model.X.shape[1], batch_size, replace=False)
                ).to(self.device)
                x_batch = self.dynamics_model.X[:, x_batch_indices.long()]  # (X.shape[0], batch_size)
                phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices.long()]  # (dim_phi, batch_size)

                # Compute costs indexed by the action and the state
                costs = self.cost(x_batch.T, self.all_actions.T)  # (num_actions, batch)

                # Compute V(x')s (vectorized)
                K_us = self.dynamics_model.K_(self.all_actions)  # (num_actions, phi_dim, phi_dim)
                phi_x_prime_batch = torch.einsum("aij,jb->aib", K_us, phi_x_batch)  # (num_actions, phi_dim, batch)
                w = self.value_function_weights.squeeze(-1)  # (phi_dim,)
                V_x_prime_batch = torch.einsum("p,apb->ab", w, phi_x_prime_batch)  # (num_actions, batch)

                # Compute policy distribution
                inner_pi_us_values = -(
                    costs + self.discount_factor * V_x_prime_batch
                )  # (all_actions.shape[1], batch_size)
                inner_pi_us = inner_pi_us_values / self.alpha  # (all_actions.shape[1], batch_size)
                real_inner_pi_us = torch.real(inner_pi_us)  # (all_actions.shape[1], batch_size)

                # Max trick
                max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0)  # (batch_size,)
                diff = real_inner_pi_us - max_inner_pi_u  # (all_actions.shape[1], batch_size)

                # Softmax distribution
                pi_us = torch.exp(diff) + delta  # (all_actions.shape[1], batch_size)
                Z_x = torch.sum(pi_us, axis=0)  # (batch_size,)
                pis_response = pi_us / Z_x  # (all_actions.shape[1], batch_size)

                # Compute log pi
                log_pis = torch.log(pis_response)  # (all_actions.shape[1], batch_size)

                # Compute expectations
                expectation_term_1 = torch.sum(
                    (costs + self.alpha * log_pis + self.discount_factor * V_x_prime_batch) * pis_response,
                    dim=0,
                ).reshape(1, -1)  # (1, batch_size)

                # Optimize value function weights
                if self.use_ols:
                    # OLS as in Lewis
                    self.value_function_weights = torch.linalg.lstsq(phi_x_batch.T, expectation_term_1.T).solution
                else:
                    # Compute loss
                    loss = torch.pow(V_x_prime_batch - expectation_term_1, 2).mean()

                    # Backpropogation for value function weights
                    self.value_function_optimizer.zero_grad()
                    loss.backward()
                    self.value_function_optimizer.step()

                # Recompute Bellman error
                BE = self.discrete_bellman_error(batch_size=batch_size * batch_scale).detach().cpu().numpy()
                bellman_errors.append(BE)

                # Print epoch number
                print(f"Epoch number: {epoch + 1}")

                # Every so often, print out and save the model weights and bellman errors
                if epoch == 0 or (epoch + 1) % how_often_to_chkpt == 0:
                    torch.save(
                        self.value_function_weights,
                        f"{self.save_data_path}/epoch_{epoch + 1}.pt",
                    )
                    torch.save(
                        bellman_errors,
                        f"{self.save_data_path}/training_data/bellman_errors.pt",
                    )
                    print(f"Bellman error at epoch {epoch + 1}: {BE}")

                    if BE <= epsilon:
                        break

            step += 1

            if len(gammas) == 0 and gamma_increment_amount == 0:
                gamma_iteration_condition = False
                break

            if self.gamma == 0.99:
                break

            if len(gammas) > 0:
                self.gamma = gammas[step]
            else:
                self.gamma += gamma_increment_amount

            if self.gamma > 0.99:
                self.gamma = 0.99

            gamma_iteration_condition = self.gamma <= 0.99

        self.gamma = original_gamma
        self.discount_factor = self.gamma**self.dt


def main():
    args = load_and_apply_config(ArgumentParser().parse_args(), _CONFIG_KEY_MAP, _FALLBACKS)
    run_name = f"{args.env_id}__{args.exp_name}__{args.num_actions}__{args.num_training_epochs}__{args.seed}__{int(time.time())}"  # noqa: E501

    writer = SummaryWriter(f"runs/SKVI/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Resolve compute device and dtype from flags
    device = resolve_device(args.cuda)
    torch.set_default_dtype(resolve_dtype(args.fp32))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])

    # Koopman tensor generation logic
    koopman_tensor = generate_koopman_tensor(
        env_id=args.env_id,
        seed=args.seed,
        num_paths=args.num_paths,
        num_steps_per_path=args.num_steps_per_path,
        state_order=args.state_order,
        action_order=args.action_order,
        regressor=args.regressor,
        device=device,
        dtype=resolve_dtype(args.fp32),
    )

    try:
        dt = envs.envs[0].unwrapped.dt
    except Exception:
        dt = None

    # Construct set of all possible actions
    all_actions = torch.from_numpy(
        np.linspace(
            start=envs.single_action_space.low,
            stop=envs.single_action_space.high,
            num=args.num_actions,
        )
    ).T.to(device=device, dtype=resolve_dtype(args.fp32))

    # Construct value iteration policy
    value_iteration_policy = DiscreteKoopmanValueIterationPolicy(
        env_id=args.env_id,
        gamma=args.gamma,
        alpha=args.alpha,
        dynamics_model=koopman_tensor,
        all_actions=all_actions,
        cost=envs.envs[0].unwrapped.vectorized_cost_fn,
        use_ols=True,
        learning_rate=args.lr,
        seed=args.seed,
        dt=dt,
        device=device,
    )

    # Use Koopman tensor training data to train policy
    value_iteration_policy.train(
        args.num_training_epochs,
        args.batch_size,
        args.batch_scale,
        how_often_to_chkpt=10,
    )

    envs.single_observation_space.dtype = np.float64
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actions = value_iteration_policy.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            ep_return = infos["episode"]["r"]
            ep_length = infos["episode"]["l"]
            if hasattr(ep_return, "item"):
                ep_return = ep_return.item()
            if hasattr(ep_length, "item"):
                ep_length = ep_length.item()
            print(f"global_step={global_step}, episodic_return={ep_return}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # Write data
        if global_step % 100 == 0:
            sps = int(global_step / (time.time() - start_time))
            print("Steps per second (SPS):", sps)
            writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
