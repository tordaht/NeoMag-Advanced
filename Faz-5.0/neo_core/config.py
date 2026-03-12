from dataclasses import dataclass


@dataclass(frozen=True)
class NeoCoreConfig:
    world_width: float = 1600.0
    world_height: float = 900.0
    max_agents: int = 192
    initial_agents: int = 96
    tribe_count: int = 3

    base_observation_dim: int = 12
    trust_observation_dim: int = 4
    syntax_observation_dim: int = 4
    observation_dim: int = 20
    base_action_dim: int = 6
    syntax_bits: int = 4
    syntax_vocab_size: int = 16
    action_dim: int = 10
    latent_dim: int = 128
    gat_hidden_dim: int = 64
    gat_heads: int = 4
    fov_neighbors: int = 6
    fov_radius: float = 96.0
    entity_state_dim: int = 8
    social_feature_dim: int = 12

    max_speed: float = 2.4
    steering_gain: float = 0.18
    thrust_gain: float = 0.32
    damping: float = 0.92
    step_dt: float = 1.0
    base_energy: float = 100.0
    energy_drain: float = 0.065
    signal_cost: float = 0.02

    interaction_radius: float = 32.0
    trust_decay: float = 0.985
    trust_min: float = -2.0
    trust_max: float = 2.0
    trust_altruism_bonus: float = 0.75
    trust_mimic_penalty: float = -1.15
    same_tribe_mimic_factor: float = 0.35

    altruism_threshold: float = 0.55
    mimic_threshold: float = 0.55
    altruism_transfer_amount: float = 1.8
    altruism_transfer_tax: float = 1.15
    minimum_self_energy: float = 18.0
    signal_threshold: float = 0.35
    gumbel_tau: float = 0.65
    influence_beta: float = 0.35
    extrinsic_alpha: float = 1.0
    trust_eta: float = 0.25
    gossip_beta: float = 0.20
    alignment_lambda: float = 0.92
    gamma: float = 0.985

    trust_clip_scale: float = 2.0
    distance_norm: float = 96.0
    position_norm_x: float = 1600.0
    position_norm_y: float = 900.0


DEFAULT_CONFIG = NeoCoreConfig()
