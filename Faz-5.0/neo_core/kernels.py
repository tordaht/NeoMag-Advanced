import taichi as ti


@ti.kernel
def reset_agents(
    agent_count: ti.i32,
    initial_agents: ti.i32,
    world_width: ti.f32,
    world_height: ti.f32,
    base_energy: ti.f32,
    alive: ti.types.ndarray(dtype=ti.i32, ndim=1),
    tribe: ti.types.ndarray(dtype=ti.i32, ndim=1),
    pos: ti.types.ndarray(dtype=ti.f32, ndim=2),
    vel: ti.types.ndarray(dtype=ti.f32, ndim=2),
    energy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    actions: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rewards: ti.types.ndarray(dtype=ti.f32, ndim=1),
    observations: ti.types.ndarray(dtype=ti.f32, ndim=2),
    trust_matrix: ti.types.ndarray(dtype=ti.f32, ndim=2),
    nearest_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    nearest_dist: ti.types.ndarray(dtype=ti.f32, ndim=1),
    interaction_delta: ti.types.ndarray(dtype=ti.f32, ndim=1),
    received_syntax: ti.types.ndarray(dtype=ti.f32, ndim=2),
    syntax_code: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for i in range(agent_count):
        enabled = 1 if i < initial_agents else 0
        alive[i] = enabled
        tribe[i] = i % 3
        pos[i, 0] = (17.0 * i + 91.0) % world_width
        pos[i, 1] = (31.0 * i + 47.0) % world_height
        vel[i, 0] = 0.0
        vel[i, 1] = 0.0
        energy[i] = base_energy if enabled == 1 else 0.0
        rewards[i] = 0.0
        nearest_idx[i] = -1
        nearest_dist[i] = 1e6
        interaction_delta[i] = 0.0
        syntax_code[i] = 0
        for j in range(actions.shape[1]):
            actions[i, j] = 0.0
        for j in range(observations.shape[1]):
            observations[i, j] = 0.0
        for j in range(received_syntax.shape[1]):
            received_syntax[i, j] = 0.0
        for j in range(agent_count):
            trust_matrix[i, j] = 0.0


@ti.kernel
def decay_trust(
    agent_count: ti.i32,
    trust_decay: ti.f32,
    trust_matrix: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for i, j in ti.ndrange(agent_count, agent_count):
        trust_matrix[i, j] *= trust_decay


@ti.kernel
def advance_agents(
    agent_count: ti.i32,
    world_width: ti.f32,
    world_height: ti.f32,
    max_speed: ti.f32,
    damping: ti.f32,
    thrust_gain: ti.f32,
    steering_gain: ti.f32,
    energy_drain: ti.f32,
    signal_cost: ti.f32,
    alive: ti.types.ndarray(dtype=ti.i32, ndim=1),
    pos: ti.types.ndarray(dtype=ti.f32, ndim=2),
    vel: ti.types.ndarray(dtype=ti.f32, ndim=2),
    energy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    actions: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rewards: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(agent_count):
        rewards[i] = 0.0
        if alive[i] == 0:
            continue

        thrust = ti.math.clamp(actions[i, 0], 0.0, 1.0)
        steer = ti.math.clamp(actions[i, 1], -1.0, 1.0)
        metabolic = ti.math.clamp(actions[i, 2], 0.0, 1.0)
        signal = ti.math.clamp(actions[i, 3], 0.0, 1.0)

        vx = vel[i, 0] * damping + thrust_gain * thrust
        vy = vel[i, 1] * damping + steering_gain * steer
        speed = ti.sqrt(vx * vx + vy * vy)
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale

        vel[i, 0] = vx
        vel[i, 1] = vy
        pos[i, 0] = (pos[i, 0] + vx) % world_width
        pos[i, 1] = (pos[i, 1] + vy) % world_height

        drain = energy_drain * (1.0 + metabolic * 0.35 + signal)
        energy[i] -= drain + signal * signal_cost
        rewards[i] -= drain
        if energy[i] <= 0.0:
            alive[i] = 0
            energy[i] = 0.0
            rewards[i] -= 1.0


@ti.kernel
def apply_social_interactions(
    agent_count: ti.i32,
    interaction_radius: ti.f32,
    altruism_threshold: ti.f32,
    mimic_threshold: ti.f32,
    signal_threshold: ti.f32,
    trust_min: ti.f32,
    trust_max: ti.f32,
    trust_altruism_bonus: ti.f32,
    trust_mimic_penalty: ti.f32,
    same_tribe_mimic_factor: ti.f32,
    altruism_transfer_amount: ti.f32,
    altruism_transfer_tax: ti.f32,
    minimum_self_energy: ti.f32,
    alive: ti.types.ndarray(dtype=ti.i32, ndim=1),
    tribe: ti.types.ndarray(dtype=ti.i32, ndim=1),
    pos: ti.types.ndarray(dtype=ti.f32, ndim=2),
    energy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    actions: ti.types.ndarray(dtype=ti.f32, ndim=2),
    trust_matrix: ti.types.ndarray(dtype=ti.f32, ndim=2),
    nearest_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    nearest_dist: ti.types.ndarray(dtype=ti.f32, ndim=1),
    interaction_delta: ti.types.ndarray(dtype=ti.f32, ndim=1),
    received_syntax: ti.types.ndarray(dtype=ti.f32, ndim=2),
    syntax_code: ti.types.ndarray(dtype=ti.i32, ndim=1),
    rewards: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(agent_count):
        nearest_idx[i] = -1
        nearest_dist[i] = 1e6
        interaction_delta[i] = 0.0
        syntax_code[i] = 0
        for bit in range(received_syntax.shape[1]):
            received_syntax[i, bit] = 0.0
        if alive[i] == 0:
            continue

        best_j = -1
        best_dist = 1e6
        for j in range(agent_count):
            if i == j or alive[j] == 0:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = ti.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_j = j

        nearest_idx[i] = best_j
        nearest_dist[i] = best_dist
        if best_j < 0 or best_dist > interaction_radius:
            continue

        trust_value = trust_matrix[i, best_j]
        signal_strength = ti.math.clamp(actions[best_j, 3], 0.0, 1.0)
        mimic_strength = ti.math.clamp(actions[best_j, 4], 0.0, 1.0)
        altruism_strength = ti.math.clamp(actions[best_j, 5], 0.0, 1.0)
        code = 0
        if signal_strength > signal_threshold:
            for bit in ti.static(range(4)):
                bit_value = 1 if actions[best_j, 6 + bit] > 0.5 else 0
                received_syntax[i, bit] = float(bit_value)
                code += bit_value << bit
        syntax_code[i] = code

        if mimic_strength > mimic_threshold:
            tribe_factor = 1.0
            if tribe[i] == tribe[best_j]:
                tribe_factor = same_tribe_mimic_factor
            delta = trust_mimic_penalty * mimic_strength * tribe_factor
            trust_value = ti.math.clamp(trust_value + delta, trust_min, trust_max)
            interaction_delta[i] += delta

        if altruism_strength > altruism_threshold and energy[best_j] > minimum_self_energy:
            transfer = ti.min(altruism_transfer_amount * altruism_strength, ti.max(0.0, energy[best_j] - minimum_self_energy))
            if transfer > 0.0:
                ti.atomic_add(energy[i], transfer)
                ti.atomic_add(energy[best_j], -transfer * altruism_transfer_tax)
                delta = trust_altruism_bonus * altruism_strength
                trust_value = ti.math.clamp(trust_value + delta, trust_min, trust_max)
                interaction_delta[i] += delta
                rewards[best_j] += transfer * 0.05

        trust_matrix[i, best_j] = ti.math.clamp(trust_value, trust_min, trust_max)


@ti.kernel
def build_observations(
    agent_count: ti.i32,
    world_width: ti.f32,
    world_height: ti.f32,
    distance_norm: ti.f32,
    trust_scale: ti.f32,
    alive: ti.types.ndarray(dtype=ti.i32, ndim=1),
    tribe: ti.types.ndarray(dtype=ti.i32, ndim=1),
    pos: ti.types.ndarray(dtype=ti.f32, ndim=2),
    vel: ti.types.ndarray(dtype=ti.f32, ndim=2),
    energy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    trust_matrix: ti.types.ndarray(dtype=ti.f32, ndim=2),
    nearest_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    nearest_dist: ti.types.ndarray(dtype=ti.f32, ndim=1),
    interaction_delta: ti.types.ndarray(dtype=ti.f32, ndim=1),
    received_syntax: ti.types.ndarray(dtype=ti.f32, ndim=2),
    observations: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for i in range(agent_count):
        for k in range(observations.shape[1]):
            observations[i, k] = 0.0

        if alive[i] == 0:
            continue

        speed = ti.sqrt(vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1])
        row_sum = 0.0
        row_min = 1e6
        alive_neighbors = 0.0
        for j in range(agent_count):
            if i == j or alive[j] == 0:
                continue
            row_sum += trust_matrix[i, j]
            row_min = ti.min(row_min, trust_matrix[i, j])
            alive_neighbors += 1.0
        if alive_neighbors == 0.0:
            row_min = 0.0

        nearest = nearest_idx[i]
        trust_out = 0.0
        trust_in = 0.0
        nearest_same_tribe = 0.0
        nearest_alive = 0.0
        nearest_dist_norm = 1.0
        if nearest >= 0 and alive[nearest] == 1:
            nearest_alive = 1.0
            trust_out = trust_matrix[i, nearest]
            trust_in = trust_matrix[nearest, i]
            nearest_same_tribe = 1.0 if tribe[nearest] == tribe[i] else 0.0
            nearest_dist_norm = ti.math.clamp(nearest_dist[i] / distance_norm, 0.0, 1.0)

        observations[i, 0] = ti.math.clamp(energy[i] / 100.0, 0.0, 2.0)
        observations[i, 1] = ti.math.clamp(speed / 4.0, 0.0, 1.0)
        observations[i, 2] = ti.math.clamp(pos[i, 0] / world_width, 0.0, 1.0)
        observations[i, 3] = ti.math.clamp(pos[i, 1] / world_height, 0.0, 1.0)
        observations[i, 4] = ti.math.clamp(0.5 + vel[i, 0] / 8.0, 0.0, 1.0)
        observations[i, 5] = ti.math.clamp(0.5 + vel[i, 1] / 8.0, 0.0, 1.0)
        observations[i, 6] = 1.0 if tribe[i] == 0 else 0.0
        observations[i, 7] = 1.0 if tribe[i] == 1 else 0.0
        observations[i, 8] = 1.0 if tribe[i] == 2 else 0.0
        observations[i, 9] = nearest_dist_norm
        observations[i, 10] = nearest_alive
        observations[i, 11] = nearest_same_tribe
        observations[i, 12] = ti.math.clamp(trust_out / trust_scale, -1.0, 1.0)
        observations[i, 13] = ti.math.clamp(trust_in / trust_scale, -1.0, 1.0)
        observations[i, 14] = ti.math.clamp((row_sum / ti.max(alive_neighbors, 1.0)) / trust_scale, -1.0, 1.0)
        observations[i, 15] = ti.math.clamp(interaction_delta[i] / trust_scale, -1.0, 1.0)
        observations[i, 16] = received_syntax[i, 0]
        observations[i, 17] = received_syntax[i, 1]
        observations[i, 18] = received_syntax[i, 2]
        observations[i, 19] = received_syntax[i, 3]
