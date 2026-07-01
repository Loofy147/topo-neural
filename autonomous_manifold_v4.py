import math
import random
import json
import os
import sys

# Import CGCE components from the local module
from cgce import CGCE

class SmallNeuralNetwork:
    """A compact, Multi-Layer Perceptron for an agent's internal logic."""
    def __init__(self, input_size, output_size, hidden_size=8, learning_rate=0.05):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases (deterministic for stability in this context)
        self.weights1 = [[(i * 0.1 + j * 0.05) / (input_size * hidden_size) for j in range(hidden_size)] for i in range(input_size)]
        self.bias1 = [0.01 * i for i in range(hidden_size)]

        self.weights2 = [[(i * 0.1 + j * 0.05) / (hidden_size * output_size) for j in range(output_size)] for i in range(hidden_size)]
        self.bias2 = [0.01 * i for i in range(output_size)]

    def _relu(self, x):
        return max(0.0, x)

    def _sigmoid(self, x):
        # Cap x to avoid overflow in exp
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))

    def forward(self, input_vector):
        # Input to Hidden
        self.hidden_layer_input = [0.0] * self.hidden_size
        self.hidden_layer_output = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            sum_val = self.bias1[j]
            for i in range(self.input_size):
                sum_val += input_vector[i] * self.weights1[i][j]
            self.hidden_layer_input[j] = sum_val
            self.hidden_layer_output[j] = self._relu(sum_val)

        # Hidden to Output
        self.output_layer_input = [0.0] * self.output_size
        self.output_layer_output = [0.0] * self.output_size
        for j in range(self.output_size):
            sum_val = self.bias2[j]
            for i in range(self.hidden_size):
                sum_val += self.hidden_layer_output[i] * self.weights2[i][j]
            self.output_layer_input[j] = sum_val
            self.output_layer_output[j] = self._sigmoid(sum_val)

        return self.output_layer_output

    def train(self, input_vector, target_output):
        # 1. Forward pass
        outputs = self.forward(input_vector)

        # 2. Backward pass
        output_deltas = [0.0] * self.output_size
        for j in range(self.output_size):
            error = target_output[j] - outputs[j]
            output_deltas[j] = error * outputs[j] * (1 - outputs[j])

        hidden_deltas = [0.0] * self.hidden_size
        for i in range(self.hidden_size):
            error = 0.0
            for j in range(self.output_size):
                error += output_deltas[j] * self.weights2[i][j]
            hidden_deltas[i] = error * (1.0 if self.hidden_layer_input[i] > 0 else 0.0)

        # 3. Update weights and biases
        for j in range(self.output_size):
            self.bias2[j] += self.learning_rate * output_deltas[j]
            for i in range(self.hidden_size):
                self.weights2[i][j] += self.learning_rate * output_deltas[j] * self.hidden_layer_output[i]

        for j in range(self.hidden_size):
            self.bias1[j] += self.learning_rate * hidden_deltas[j]
            for i in range(self.input_size):
                self.weights1[i][j] += self.learning_rate * hidden_deltas[j] * input_vector[i]

class Agent:
    def __init__(self, name, dimension_focus, initial_mass=0.1, nn_input_size=7, nn_output_size=1, config=None):
        self.name = name
        self.dimension_focus = dimension_focus
        self.mass = initial_mass
        self.config = config or {}
        self.will_strength_multiplier = self.config.get("will_strength_multiplier", 1.5)
        self.will_strength = initial_mass * self.will_strength_multiplier

        nn_config = self.config.get("nn_hyperparameters", {})
        self.insider_network = SmallNeuralNetwork(
            nn_input_size,
            nn_output_size,
            hidden_size=nn_config.get("hidden_layer_size", 8),
            learning_rate=nn_config.get("learning_rate", 0.05)
        )

    def _get_nn_input(self, state_vector, world_model):
        base_input = [state_vector.get(dim, 0.0) for dim in ["Creativity", "Logic", "Density", "Realization", "Entropy"]]
        context_input = [
            world_model.get("current_context", {}).get("processing_load", 0.0),
            1.0 if world_model.get("knowledge_base", {}).get("even_pattern_detected") else 0.0
        ]
        return base_input + context_input

    def exert_will(self, state_vector, world_model, other_agents):
        self.will_strength = self.mass * self.will_strength_multiplier
        nn_input = self._get_nn_input(state_vector, world_model)
        nn_output = self.insider_network.forward(nn_input)[0]

        if self.dimension_focus in state_vector:
            modulated_will = self.will_strength * (0.5 + nn_output)
            delta = modulated_will * (1.0 - state_vector[self.dimension_focus])
            state_vector[self.dimension_focus] += delta
        return state_vector

    def critique_and_modify(self, target_agent, current_state_vector, world_model):
        pass

class CreativityAgent(Agent):
    def exert_will(self, state_vector, world_model, other_agents):
        state_vector = super().exert_will(state_vector, world_model, other_agents)
        if state_vector["Logic"] > 0.7:
            state_vector["Logic"] -= 0.05 * self.mass
        return state_vector

class LogicAgent(Agent):
    def exert_will(self, state_vector, world_model, other_agents):
        state_vector = super().exert_will(state_vector, world_model, other_agents)
        if state_vector["Entropy"] > 0.4:
            state_vector["Entropy"] -= 0.1 * self.mass
        return state_vector

class DensityAgent(Agent):
    def exert_will(self, state_vector, world_model, other_agents):
        state_vector = super().exert_will(state_vector, world_model, other_agents)
        if state_vector["Density"] > 0.6:
            state_vector["Realization"] += 0.08 * self.mass
        return state_vector

class RealizationAgent(Agent):
    def exert_will(self, state_vector, world_model, other_agents):
        state_vector = super().exert_will(state_vector, world_model, other_agents)
        if state_vector["Entropy"] > 0.7:
            state_vector["Entropy"] -= 0.05 * self.mass
        return state_vector

class EntropyAgent(Agent):
    def exert_will(self, state_vector, world_model, other_agents):
        state_vector = super().exert_will(state_vector, world_model, other_agents)
        if state_vector["Creativity"] < 0.5:
            state_vector["Creativity"] += 0.03 * self.mass
        return state_vector

class ClosureLayer:
    def __init__(self, dimensions):
        self.dimensions = dimensions
    def apply_closure(self, state_vector):
        return {dim: min(1.0, max(0.0, state_vector.get(dim, 0.0))) for dim in self.dimensions}

class AutonomousManifoldV4:
    """
    Unified Manifold v4.0: Integrated with Circular Geometric Convergence Engine (CGCE).
    """
    def __init__(self, persistence_file="manifold_state_v4.json", config_file="config.json"):
        self.name = "Unified_Manifold_v4_CGCE"
        self.persistence_file = persistence_file
        self.config_file = config_file
        self.dimensions = ["Creativity", "Logic", "Density", "Realization", "Entropy"]

        self.load_config()
        self.cgce = CGCE(config=self.config.get("cgce_config"))

        self.agents = [
            CreativityAgent("Creativity_Agent", "Creativity", initial_mass=self.config["agents"]["Creativity_Agent"]["initial_mass"], config=self.config),
            LogicAgent("Logic_Agent", "Logic", initial_mass=self.config["agents"]["Logic_Agent"]["initial_mass"], config=self.config),
            DensityAgent("Density_Agent", "Density", initial_mass=self.config["agents"]["Density_Agent"]["initial_mass"], config=self.config),
            RealizationAgent("Realization_Agent", "Realization", initial_mass=self.config["agents"]["Realization_Agent"]["initial_mass"], config=self.config),
            EntropyAgent("Entropy_Agent", "Entropy", initial_mass=self.config["agents"]["Entropy_Agent"]["initial_mass"], config=self.config)
        ]
        self.closure_layer = ClosureLayer(self.dimensions)
        self.load_state()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "q_weights": {"Creativity": 0.08, "Logic": 0.18, "Density": 0.16, "Realization": 0.20, "Entropy": 0.12},
                "target_state": {"Creativity": 0.9, "Logic": 0.9, "Density": 0.7, "Realization": 0.98, "Entropy": 0.05},
                "agents": {
                    "Creativity_Agent": {"initial_mass": 0.15},
                    "Logic_Agent": {"initial_mass": 0.15},
                    "Density_Agent": {"initial_mass": 0.2},
                    "Realization_Agent": {"initial_mass": 0.12},
                    "Entropy_Agent": {"initial_mass": 0.08}
                },
                "will_strength_multiplier": 1.5,
                "agent_mass_increment": 0.005,
                "cgce_config": {},
                "nn_hyperparameters": {"hidden_layer_size": 8, "learning_rate": 0.05}
            }
        self.q_weights = self.config.get("q_weights")
        self.target_state = self.config.get("target_state")
        self.agent_mass_increment = self.config.get("agent_mass_increment", 0.005)

    def load_state(self):
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    for agent in self.agents:
                        if agent.name in data.get("agent_masses", {}):
                            agent.mass = data["agent_masses"][agent.name]
                        if agent.name in data.get("agent_nn_weights", {}):
                            weights_data = data["agent_nn_weights"][agent.name]
                            if "weights1" in weights_data:
                                agent.insider_network.weights1 = weights_data["weights1"]
                                agent.insider_network.bias1 = weights_data["bias1"]
                                agent.insider_network.weights2 = weights_data["weights2"]
                                agent.insider_network.bias2 = weights_data["bias2"]
                print(f"[{self.name}] State loaded.")
            except Exception as e:
                print(f"[{self.name}] Error loading state: {e}")

    def save_state(self):
        data = {
            "agent_masses": {agent.name: agent.mass for agent in self.agents},
            "agent_nn_weights": {
                agent.name: {
                    "weights1": agent.insider_network.weights1,
                    "bias1": agent.insider_network.bias1,
                    "weights2": agent.insider_network.weights2,
                    "bias2": agent.insider_network.bias2
                } for agent in self.agents
            }
        }
        with open(self.persistence_file, 'w') as f:
            json.dump(data, f, indent=4)

    def calculate_q_score(self, state_vector):
        q_score = sum(state_vector[dim] * self.q_weights[dim] if dim != "Entropy" else (1.0 - state_vector[dim]) * self.q_weights[dim] for dim in self.dimensions)
        total_mass = sum(a.mass for a in self.agents)
        return min(1.0, q_score + 0.26 + (total_mass * 0.05))

    def process(self, initial_input, iterations=5):
        print(f"\n[{self.name}] Initiating Integrated Process...")
        cgce_result = self.cgce.process_cognition_loop(initial_input)
        world_model = cgce_result["final_state"]["world_model"]

        current_state = {dim: 0.2 for dim in self.dimensions}
        current_state["Entropy"] = 0.4

        for i in range(iterations):
            q_before = self.calculate_q_score(current_state)
            for agent in self.agents:
                current_state = agent.exert_will(current_state, world_model, self.agents)
            current_state = self.closure_layer.apply_closure(current_state)
            q_after = self.calculate_q_score(current_state)
            reward = 1.0 if q_after > q_before else 0.0
            for agent in self.agents:
                agent.insider_network.train(agent._get_nn_input(current_state, world_model), [reward])
                if reward > 0: agent.mass += self.agent_mass_increment
            decision = self.cgce.reasoner.decide(world_model)
            action = self.cgce.actor.execute(decision)
            valid, reason = self.cgce.verifier.verify(world_model, decision, action)
            if not valid:
                current_state["Entropy"] += 0.1
        final_q = self.calculate_q_score(current_state)
        self.save_state()
        return {"q_score": final_q, "state_vector": current_state, "agent_masses": {a.name: a.mass for a in self.agents}}

if __name__ == "__main__":
    manifold = AutonomousManifoldV4()
    manifold.process({"user_state": "Singularity Evolution", "pattern_type": "even", "data_volume": 500}, iterations=10)
