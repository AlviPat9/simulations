"""

Reinforcement learning Pilot
=============================

Pilot for aircraft simulators based on AI (Reinforcement learning). In this file it is defined the agent for the
simulation. Here it is defined the environment of the simulator. After, the reinforcement model should be defined
(based on Neural Networks).

Author: Alvaro Marcos Canedo
"""
from simulations.utilities.keys import AircraftKeys as Ak
from simulations.utilities.tools import haversine, convert_dict_to_ndarray
from simulations.io.base_agent_rl import BaseAgentRL

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v1 import Adam

from typing import Any

import gymnasium as gym
import numpy as np
import logging as log


class PilotRL(gym.Env):
    """

    Pilot environment based on Reinforcement learning (AI) for Aircraft simulators.

    """

    def __init__(self, simulator, destination: np.ndarray, initial_state: np.ndarray, step_size=0.3, final_time=1000):
        """

        Constructor method.

        @param simulator: Aircraft simulator.
        @type simulator:
        @param destination: Coordinates (latitude and longitude) of the place to get to.
        @type destination: np.ndarray
        @param initial_state: Initial state of the model.
        @type initial_state: np.ndarray
        @param step_size: Step size for the calculation.
        @type step_size: float, optional
        @param final_time: Final time for the simulation.
        @type final_time: float, optional

        """
        # Observation space definition
        # At first, only latitude and longitude
        # position(latitude & longitude), speed (u, v, w), euler_angles (phi, theta, psi), fuel
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi/2, -np.pi, -120.0, -120.0, -20.0, -np.pi, -np.pi/2, 0.0, 0.0]),
            high=np.array([np.pi/2, np.pi, 120.0, 120.0, 20.0, np.pi, np.pi/2, 2*np.pi, 550.0]),
            dtype=np.float32
        )

        # Action space definition
        # Elevator, aileron, rudder and throttle lever
        self.action_space = gym.spaces.Box(
            low=np.array([-np.pi/6, -np.pi/6, -np.pi/6, 0.0]),
            high=np.array([np.pi/6, np.pi/6, np.pi/6, 2400.0]),
            dtype=np.float32
        )

        # Create instance of the simulator
        self.simulator = simulator(initial_state, step_size, final_time)

        # Set initial state
        self.state = initial_state

        # Set final destination
        self.destination = destination

        # Initialize step of the model
        self.current_step = 0

        # Set max step for the environment
        self.max_step = 1000

        # Set a tolerance for checking if it arrived at the destination port
        self.tol = 1000.0  # Defined as 1km

        # Initialize last distance
        self.last_distance = None

        # Initialize the logger
        self.logger = log.getLogger(__name__)
        self.logger.setLevel(log.INFO)
        self.logger.addHandler(log.StreamHandler())

    def step(self, action: gym.spaces.Dict) -> tuple:
        """

        Step method of Pilot environment

        @param action: Action taken by the pilot to fly the aircraft.
        @type action: gym.spaces.Dict

        @return: Current state, current reward of the function and status of the goal.
        @rtype: tuple
        """

        # Calculate current step
        self.state = self._get_state(self.simulator.step(action))

        # Update current step
        self.current_step += 1

        # Calculate distance to destination based on Haversine formula
        distance = haversine(self.destination[0], self.destination[1], self.state[Ak.position][0],
                             self.state[Ak.position][1])

        # TODO -> Define reward function. The reward function can take multiple inputs. The first one must be the
        #  distance, then maybe it is a good approach to take fuel and steps as a penalty.
        distance_reward = max(0, self.last_distance - distance)
        # fuel_penalty = self.state[Ak.fuel]
        # time_penalty = self.simulator.integration.time / self.max_step
        # reward = distance_reward - fuel_penalty - time_penalty
        reward = distance_reward

        done = self._check_done(distance)

        # TODO -> Define relevant information in the log
        # Log relevant information
        info = f"Step: {self.current_step}, Reward: {reward}, Done: {done}"
        self.logger.info(info)

        # Set last distance
        self.last_distance = distance

        return self.state, reward, done

    def reset(self) -> np.ndarray:
        """

        Reset method of the Pilot environment.

        @return: Initial state of the model.
        @rtype: np.ndarray
        """

        # Reset simulation
        self.state = self.simulator.reset()

        # Reset step
        self.current_step = 0

        return self.state

    def render(self):
        """

        Render method of the Pilot environment. Used to generate and return a visual representation of the
        current state.

        @return:
        """
        pass

    def _check_done(self, distance):
        """

        Method to check if the objective goal has been reached.

        :param distance: Distance to the destination. Based on haversine formula.
        :type distance: float
        :return: Boolean operator to check if done.
        :rtype: bool
        """

        if distance < self.tol:
            return True
        elif self.current_step >= self.max_step:
            return True
        else:
            return False

    @staticmethod
    def _get_state(state: np.ndarray) -> np.ndarray:
        """

        Method to get the current important information of the state of the aircraft. At the moment,

        :param state: Current information of the state of the aircraft.
        :type state: np.ndarray
        :return: Relevant information for the observation space of the model.
        :rtype: np.ndarray
        """
        # TODO
        # The position of each variable may change. So, this method should be reworked.
        return state[[15, 16, 6, 7, 8, 0, 1, 2, 14]]


class PilotAgentTF(BaseAgentRL):
    """

    Pilot agent definition. This definition is based on Neural networks model. It has predefined values for the neural
    networks. It is also done in tensorflow. Maybe an implementation with pytorch could be great.

    It is based on a policy gradient approach. It is a basic policy-based reinforcement learning algorithm.

    """

    def __init__(self, layers: int, input_shape: int, neurons: np.ndarray, activation_function: list, action_size: int,
                 env: Any):
        """

        Constructor method

        @param layers: Number of layers that the neural network must have. At least, the input and output layer.
        @type layers: int
        @param input_shape: Input shape of the model. Should be the same as the state.
        @type input_shape: int
        @param  neurons: Number of neurons of each layer.
        @type neurons: np.ndarray
        @param activation_function: List with the activation function for the layers of the neural network.
        @type activation_function: list
        @param action_size: Action length for the output layer.
        @type action_size: int
        @param env: Environment defined for training the agent.
        @type env: Any

        """

        super().__init__(action_size, env)

        # Build Neural Network model
        self.model = self._build_model(layers, input_shape, neurons, activation_function)

        # Set optimizer
        self.optimizer = Adam(lr=0.01)

    def _build_model(self, layers: int, input_shape: int, neurons: np.ndarray, activation_function: list) -> Sequential:
        """

        @param layers: Number of layers that the neural network must have. At least, the input and output layer.
        @type layers: int
        @param input_shape: Input shape of the model. Should be the same as the state.
        @type input_shape: int
        @param  neurons: Number of neurons of each layer.
        @type neurons: np.ndarray
        @param activation_function: List with the activation function for the layers of the neural network.
        @type activation_function: list

        @return: Neural networks model for the Reinforcement Learning model.
        @rtype: Sequential
        """
        model = Sequential(Dense(neurons[0], activation=activation_function[0], input_shape=(input_shape,)))

        for i in range(1, layers):
            model.add(Dense(neurons[i], activation=activation_function[i]))

        model.add(Dense(self.action_size, activation=activation_function[-1]))

        return model

    def get_action(self, state: dict) -> dict:
        """

        Method to retrieve actions based on the current state of the aircraft.

        @param state: Actual state of the aircraft.
        @type state: dict

        @return: Actions predicted by the neural network model for the simulator.
        @rtype: dict
        """
        # Convert state (dictionary) to np.ndarray type
        state_array = convert_dict_to_ndarray(state)

        # Predict action values with neural network model
        action_values = self.model.predict(np.expand_dims(state_array, axix=0))

        # Return of the policy (In this case is greedy policy)
        return {key: action_values[i] for i, key in enumerate(state.keys())}

    def train(self, episodes: int, max_step_per_episode: int) -> list:
        """

        Train the agent using the provided environment.

        @param episodes: Total number of episodes to train the agent.
        @type episodes: int
        @param max_step_per_episode: Maximum number of steps or actions the agent can take within a single episode.
        @type max_step_per_episode: int

        @return: List containing the reward assigned to each episode.
        @rtype: list
        """

        # Initialize reward storage
        rewards_storage = []

        for i in range(episodes):
            # Initialize for each episode
            state = self.env.reset()
            state = self.env.simulator.state_to_dict(state)
            episode_reward = 0

            for step in range(max_step_per_episode):

                # Get action from the agent.
                action = self.get_action(state)

                # Get calculation step
                state, reward, done = self.env.step(action)

                # Update reward
                episode_reward += reward

                # Check if calculation is done
                if done:
                    break

            # Append reward to list
            rewards_storage.append(episode_reward)

        return rewards_storage


__all__ = ["PilotRL", "PilotAgentTF"]
