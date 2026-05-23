from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.envs.registration import register
from scipy.integrate import solve_ivp

dt = 0.01
max_episode_steps = int(20 / dt)
# max_episode_steps = int(2 / dt)

register(
    id="FluidFlow-v0", entry_point="koopmanrl.environments.fluid_flow:FluidFlow", max_episode_steps=max_episode_steps
)


class FluidFlow(gym.Env):
    def __init__(self):
        # Configuration with hardcoded values
        self.state_dim = 3
        self.action_dim = 1

        self.state_range = [-1.0, 1.0]

        self.action_range = [-10.0, 10.0]

        # Dynamics
        self.omega = 1.0
        self.mu = 0.1
        self.A = -0.1
        self.lamb = 1

        self.dt = dt
        self.n_substeps = 4
        self.max_episode_steps = max_episode_steps

        # For LQR
        x_bar = 0
        y_bar = 0
        z_bar = 0
        self.continuous_A = np.array(
            [
                [self.mu + self.A * z_bar, -self.omega, self.A * x_bar],
                [self.omega, self.mu + self.A * z_bar, self.A * y_bar],
                [2 * self.lamb * x_bar, 2 * self.lamb * y_bar, -self.lamb],
            ]
        )
        self.continuous_B = np.array([[0], [1], [0]])

        # Define cost/reward values
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim)

        self.reference_point = np.zeros(self.state_dim)

        # Observations are 3-dimensional vectors indicating spatial location.
        self.state_minimums = np.array([-1.0, -1.0, 0.0])
        self.state_maximums = np.array([1.0, 1.0, 1.0])
        self.observation_space = spaces.Box(
            low=self.state_minimums, high=self.state_maximums, shape=(self.state_dim,), dtype=np.float64
        )

        # We have a continuous action space. In this case, there is only 1 dimension per action
        self.action_minimums = np.ones(self.action_dim) * self.action_range[0]
        self.action_maximums = np.ones(self.action_dim) * self.action_range[1]
        self.action_space = spaces.Box(
            low=self.action_minimums, high=self.action_maximums, shape=(self.action_dim,), dtype=np.float64
        )

        # History of states traversed during the current episode
        self.states: list[np.ndarray] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)

        options = options or {}
        state = options.get("state")
        if state is None:
            self.state = self.np_random.uniform(
                low=self.state_minimums,
                high=self.state_maximums,
                size=(self.state_dim,),
            )
        else:
            self.state = state
        self.states = [self.state]

        # Track number of steps taken
        self.step_count = 0

        return self.state, {}

    def cost_fn(self, state, action):
        _state = state - self.reference_point

        cost = _state @ self.Q @ _state.T + action @ self.R @ action.T

        return cost

    def reward_fn(self, state, action):
        return -self.cost_fn(state, action)

    def vectorized_cost_fn(self, states, actions):
        Q = torch.as_tensor(self.Q, dtype=states.dtype, device=states.device)
        R = torch.as_tensor(self.R, dtype=states.dtype, device=states.device)
        ref = torch.as_tensor(self.reference_point, dtype=states.dtype, device=states.device)
        _states = (states - ref).T
        state_cost = torch.einsum("bi,ij,bj->b", _states.T, Q, _states.T).unsqueeze(-1)
        mat = state_cost + torch.pow(actions.T, 2) * R
        return mat.T

    def vectorized_reward_fn(self, states, actions):
        return -self.vectorized_cost_fn(states, actions)

    def _deriv_batch(self, states, actions):
        x, y, z = states[:, 0], states[:, 1], states[:, 2]
        u = actions[:, 0]
        x_dot = self.mu * x - self.omega * y + self.A * x * z
        y_dot = self.omega * x + self.mu * y + self.A * y * z + u
        z_dot = -self.lamb * (z - x.pow(2) - y.pow(2))
        return torch.stack([x_dot, y_dot, z_dot], dim=1)

    def f_batch(self, states, actions, generator=None):
        """Batched dynamics via substepped RK4 over a single dt. generator unused (deterministic)."""
        h = self.dt / self.n_substeps
        x = states
        for _ in range(self.n_substeps):
            k1 = self._deriv_batch(x, actions)
            k2 = self._deriv_batch(x + 0.5 * h * k1, actions)
            k3 = self._deriv_batch(x + 0.5 * h * k2, actions)
            k4 = self._deriv_batch(x + h * k3, actions)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def reset_batch(self, n, device, dtype=torch.float64, generator=None):
        low = torch.as_tensor(self.state_minimums, device=device, dtype=dtype)
        high = torch.as_tensor(self.state_maximums, device=device, dtype=dtype)
        return low + (high - low) * torch.rand(n, self.state_dim, device=device, dtype=dtype, generator=generator)

    def continuous_f(self, action=None):
        """
        Ground-truth, continuous dynamics of the system.

        Parameters
        ----------
        action : np.ndarray
            Action vector. If left as None, then random policy is used.
        """

        def f_u(t, input):
            """
            Parameters
            ----------
            t : float
                Timestep.
            input : np.ndarray
                State vector.
            """

            x, y, z = input

            x_dot = self.mu * x - self.omega * y + self.A * x * z
            y_dot = self.omega * x + self.mu * y + self.A * y * z
            z_dot = -self.lamb * (z - np.power(x, 2) - np.power(y, 2))

            u = action
            if u is None:
                u = np.zeros(self.action_dim)

            return [x_dot, y_dot + u[0], z_dot]

        return f_u

    def f(self, state, action):
        """
        Ground-truth, discretized dynamics of the system. Pushes forward from (t) to (t + dt) using a constant action.

        Parameters
        ----------
        state : any
            State array.
        action : any
            Action array.

        Returns
        -------
            State array pushed forward in time.
        """

        soln = solve_ivp(fun=self.continuous_f(action), t_span=[0, dt], y0=state, method="RK45")

        return soln.y[:, -1]

    def step(self, action):
        # Compute reward of system
        reward = self.reward_fn(self.state, action)

        # Update state
        self.state = self.f(self.state, action)
        self.states.append(self.state)

        # Update global step count
        self.step_count += 1

        # An episode is done if the system has run for max_episode_steps
        terminated = self.step_count >= max_episode_steps

        return self.state, reward, terminated, False, {}
