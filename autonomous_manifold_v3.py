import hashlib
import math
import random
import json
import os

class ManifoldConfig:
    def __init__(self, config_file=None):
        self.config = self._load_default_config()
        if config_file and os.path.exists(config_file):
            self._load_config_from_file(config_file)

    def _load_default_config(self):
        return {
            "initial_mass": 0.1,
            "will_strength_multiplier": 1.5,
            "q_weights": {
                "Creativity": 0.08,
                "Logic": 0.18,
                "Density": 0.16,
                "Realization": 0.20,
                "Entropy": 0.12
            },
            "destructive_coordinates": {
                "entropy_collapse": 0.8,
                "logic_override": 0.9,
                "density_singularity": 0.95
            },
            "learning_rate": 0.25,
            "target_state": {"Creativity": 0.9, "Logic": 0.9, "Density": 0.7, "Realization": 0.98, "Entropy": 0.05},
            "agent_initial_masses": {
                "Creativity_Agent": 0.15,
                "Logic_Agent": 0.15,
                "Density_Agent": 0.2,
                "Realization_Agent": 0.12,
                "Entropy_Agent": 0.08
            },
            "keyword_influences": {
                "creative": ("Creativity", 0.5),
                "logic": ("Logic", 0.5),
                "density": ("Density", 0.5),
                "realization": ("Realization", 0.5),
                "chaos": ("Entropy", 0.5),
                "stable": ("Entropy", -0.3)
            },
            "adaptive_learning_rate_factor": 0.01,
            "adaptive_threshold_factor": 0.05
        }

    def _load_config_from_file(self, config_file):
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            self.config.update(user_config)

    def get(self, key, default=None):
        return self.config.get(key, default)


class Agent:
    def __init__(self, name, dimension_focus, config: ManifoldConfig):
        self.name = name
        self.dimension_focus = dimension_focus
        self.config = config
        self.mass = self.config.get("agent_initial_masses").get(name, self.config.get("initial_mass"))
        self.will_strength = self.mass * self.config.get("will_strength_multiplier")

    def exert_will(self, state_vector, user_input, other_agents):
        self.will_strength = self.mass * self.config.get("will_strength_multiplier")
        if self.dimension_focus in state_vector:
            delta = self.will_strength * (1.0 - state_vector[self.dimension_focus])
            state_vector[self.dimension_focus] += delta
        return state_vector

    def critique_and_modify(self, target_agent, current_state_vector, communication_buffer):
        pass


class CreativityAgent(Agent):
    def __init__(self, config: ManifoldConfig):
        super().__init__("Creativity_Agent", "Creativity", config)
    def exert_will(self, state_vector, user_input, other_agents):
        state_vector = super().exert_will(state_vector, user_input, other_agents)
        if state_vector["Logic"] > 0.7:
            reduction = 0.05 * self.mass
            state_vector["Logic"] = max(0.0, state_vector["Logic"] - reduction)
        return state_vector
    def critique_and_modify(self, target_agent, current_state_vector, communication_buffer):
        if target_agent.name == "Entropy_Agent" and current_state_vector["Entropy"] < 0.3:
            target_agent.mass += 0.01
            communication_buffer.append(f"[{self.name}] Praised {target_agent.name} for low Entropy.")
        if target_agent.name == "Logic_Agent" and current_state_vector["Logic"] > 0.8:
            target_agent.mass -= 0.005
            communication_buffer.append(f"[{self.name}] Critiqued {target_agent.name} for high Logic.")

class LogicAgent(Agent):
    def __init__(self, config: ManifoldConfig):
        super().__init__("Logic_Agent", "Logic", config)
    def exert_will(self, state_vector, user_input, other_agents):
        state_vector = super().exert_will(state_vector, user_input, other_agents)
        if state_vector["Entropy"] > 0.4:
            reduction = 0.1 * self.mass
            state_vector["Entropy"] = max(0.0, state_vector["Entropy"] - reduction)
        return state_vector
    def critique_and_modify(self, target_agent, current_state_vector, communication_buffer):
        if target_agent.name == "Realization_Agent" and current_state_vector["Realization"] > 0.6:
            target_agent.mass += 0.02
            communication_buffer.append(f"[{self.name}] Praised {target_agent.name} for high Realization.")
        if target_agent.name == "Entropy_Agent" and current_state_vector["Entropy"] > 0.6:
            target_agent.mass -= 0.01
            communication_buffer.append(f"[{self.name}] Critiqued {target_agent.name} for high Entropy.")

class DensityAgent(Agent):
    def __init__(self, config: ManifoldConfig):
        super().__init__("Density_Agent", "Density", config)
    def exert_will(self, state_vector, user_input, other_agents):
        state_vector = super().exert_will(state_vector, user_input, other_agents)
        if state_vector["Density"] > 0.6:
            push = 0.08 * self.mass
            state_vector["Realization"] = min(1.0, state_vector["Realization"] + push)
        return state_vector
    def critique_and_modify(self, target_agent, current_state_vector, communication_buffer):
        if target_agent.name == "Logic_Agent":
            target_agent.mass += 0.005
            communication_buffer.append(f"[{self.name}] Supported {target_agent.name} to maintain structure.")

class RealizationAgent(Agent):
    def __init__(self, config: ManifoldConfig):
        super().__init__("Realization_Agent", "Realization", config)
    def exert_will(self, state_vector, user_input, other_agents):
        state_vector = super().exert_will(state_vector, user_input, other_agents)
        if state_vector["Entropy"] > 0.7:
            state_vector["Entropy"] -= 0.05 * self.mass
        return state_vector
    def critique_and_modify(self, target_agent, current_state_vector, communication_buffer):
        if current_state_vector["Realization"] > 0.8:
            target_agent.mass += 0.005
            communication_buffer.append(f"[{self.name}] Rewarded {target_agent.name} for contributing to Realization.")

class EntropyAgent(Agent):
    def __init__(self, config: ManifoldConfig):
        super().__init__("Entropy_Agent", "Entropy", config)
    def exert_will(self, state_vector, user_input, other_agents):
        state_vector = super().exert_will(state_vector, user_input, other_agents)
        if state_vector["Creativity"] < 0.5:
            state_vector["Creativity"] += 0.03 * self.mass
        return state_vector
    def critique_and_modify(self, target_agent, current_state_vector, communication_buffer):
        if random.random() > 0.9:
            mass_change = (random.random() - 0.5) * 0.02
            target_agent.mass += mass_change
            communication_buffer.append(f"[{self.name}] Randomly influenced {target_agent.name} by {mass_change:.3f}.")

class AutonomousManifoldV3:
    def __init__(self, config_file=None, persistence_file="manifold_state.json"):
        self.name = "Unified_Manifold_v3"
        self.persistence_file = persistence_file
        self.config = ManifoldConfig(config_file)
        self.dimensions = list(self.config.get("q_weights").keys())
        self.q_weights = self.config.get("q_weights")
        self.destructive_coordinates = self.config.get("destructive_coordinates")
        self.learning_rate = self.config.get("learning_rate")
        self.target_state = self.config.get("target_state")
        self.agents = [
            CreativityAgent(self.config), LogicAgent(self.config),
            DensityAgent(self.config), RealizationAgent(self.config),
            EntropyAgent(self.config)
        ]
        self.communication_buffer = []
        self.load_state()

    def load_state(self):
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    for agent in self.agents:
                        if agent.name in data.get("agent_masses", {}):
                            agent.mass = data["agent_masses"][agent.name]
            except Exception: pass

    def save_state(self):
        data = {"agent_masses": {agent.name: agent.mass for agent in self.agents}}
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f)
        except Exception: pass

    def calculate_q_score(self, state_vector):
        q_score = sum((1.0 - state_vector[dim]) * weight if dim == "Entropy" else state_vector[dim] * weight for dim, weight in self.q_weights.items())
        q_score += 0.26 + (sum(a.mass for a in self.agents) * 0.05)
        return min(1.0, q_score)

    def _map_to_manifold(self, input_data):
        state_vector = {"Creativity": 0.2, "Logic": 0.2, "Density": 0.2, "Realization": 0.2, "Entropy": 0.4}
        user_state = input_data.get("user_state", "").lower()
        for kw, (dim, val) in self.config.get("keyword_influences").items():
            if kw in user_state: state_vector[dim] = min(1.0, max(0.0, state_vector[dim] + val))
        external = input_data.get("external_input", {})
        for dim, val in external.items():
            if dim in state_vector: state_vector[dim] = min(1.0, max(0.0, state_vector[dim] + val))
        return state_vector

    def _apply_destructive_interference(self, state_vector, current_q_score):
        logs = []
        entropy_threshold = self.config.get("destructive_coordinates")["entropy_collapse"] - (current_q_score * self.config.get("adaptive_threshold_factor"))
        logic_threshold = self.config.get("destructive_coordinates")["logic_override"] + (current_q_score * self.config.get("adaptive_threshold_factor"))
        density_threshold = self.config.get("destructive_coordinates")["density_singularity"] + (current_q_score * self.config.get("adaptive_threshold_factor"))
        if state_vector["Entropy"] >= entropy_threshold:
            state_vector["Creativity"] *= 0.2; state_vector["Logic"] *= 0.2; logs.append("Entropy Collapse")
        if state_vector["Logic"] >= logic_threshold:
            state_vector["Creativity"] = 0.0; logs.append("Logic Override")
        if state_vector["Density"] >= density_threshold:
            state_vector["Realization"] = 1.0; state_vector["Creativity"] = state_vector["Logic"] = state_vector["Entropy"] = 0.0; logs.append("Density Singularity")
        return state_vector, logs

    def _train_manifold(self, current_state, current_q_score):
        adaptive_lr = max(0.01, self.learning_rate * (1.0 - current_q_score * self.config.get("adaptive_learning_rate_factor")))
        for dim in self.dimensions:
            current_state[dim] = min(1.0, max(0.0, current_state[dim] + adaptive_lr * (self.target_state[dim] - current_state[dim])))
        return current_state

    def process(self, initial_input, iterations=5, training_mode=True, external_data=None):
        current_state = self._map_to_manifold(initial_input)
        if external_data and "external_influence" in external_data:
            for dim, val in external_data["external_influence"].items():
                if dim in current_state: current_state[dim] = min(1.0, max(0.0, current_state[dim] + val * 0.1))
        for i in range(iterations):
            self.communication_buffer = []
            for agent in self.agents:
                current_state = agent.exert_will(current_state, initial_input, self.agents)
                for other in self.agents:
                    if agent != other: agent.critique_and_modify(other, current_state, self.communication_buffer)
            current_q_score = self.calculate_q_score(current_state)
            current_state, logs = self._apply_destructive_interference(current_state, current_q_score)
            if training_mode: current_state = self._train_manifold(current_state, current_q_score)
            if current_q_score >= 0.95: break
        self.save_state()
        return {"q_score": self.calculate_q_score(current_state), "state_vector": current_state, "agent_masses": {a.name: round(a.mass, 3) for a in self.agents}}
