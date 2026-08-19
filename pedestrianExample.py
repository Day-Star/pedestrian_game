"""
pedestrianExample.py
---------------------
Standalone demonstration of the full AReS pedestrian-game inference pipeline,
reusing this repo's existing helpers from modelCall.py and pedestrianGame.py
(mirrors context/carlaExample.py, the equivalent demo for the sibling CARLA
project).

Pipeline:

  1. Load game parameters, ONNX model paths/sessions, and normalization data.
  2. Convert a placeholder global telemetry trace (robot pose + pedestrian
     positions + robot's applied controls) to robot-centric state/action
     histories, one per pedestrian.
  3. For each pedestrian, run the supervised -> decoder pipeline over a
     sliding window of that history to estimate a system-ID model, then
     check whether a freshly re-estimated ("candidate") model is more
     accurate than the model currently in use, evaluated over the same data
     window -- and reject the update if it is not.
  4. Assemble the final 17-D observation for the control/disturbance/critic
     networks from the winning per-pedestrian latents and error bounds.
  5. Run the disturbance ("worst case adversary"), critic, and actor
     networks, and apply the same q-value safety filter pedestrianGame.py's
     game loop uses to pick between the nominal and actor control.

This example uses hardcoded placeholder telemetry for illustration. A real
deployment should replace Section 2 with actual robot/pedestrian telemetry.
"""

import numpy as np

from modelCall import (
    build_ares_observation,
    build_critic_action,
    build_supervised_trajectory_input,
    denormalize_control,
    get_model_paths,
    get_onnx_session,
    load_game_parameters,
    normalize_control,
    run_actor,
    run_critic,
    run_decoder,
    run_disturbance,
    run_supervised,
)
from pedestrianGame import (
    build_system_id_model,
    compute_error_bounds,
    compute_relative_position,
    denormalize_latent,
)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# The exported actor/disturbance/critic nets expect exactly two pedestrians'
# worth of relative position / latent / disturbance-bound slots in the obs
# (pedestrianGame.py duplicates a single pedestrian's values to fill this
# when only one is present; here we demonstrate the non-duplicated path with
# two real pedestrians).
NUM_PEDESTRIANS = 2
LATENT_NORM = 5.0  # supervised model output range is [-5, 5], matches pedestrianGame.py

# =========================================================================
# 1. Load game parameters, ONNX sessions, and normalization data
# =========================================================================

params = load_game_parameters()
model_paths = get_model_paths(params)
io = params["onnx_io"]

actor_sess = get_onnx_session(model_paths["actor"])
critic_sess = get_onnx_session(model_paths["critic"])
disturbance_sess = get_onnx_session(model_paths["disturbance"])
supervised_sess = get_onnx_session(model_paths["supervised"])
decoder_sess = get_onnx_session(model_paths["decoder"])

ares_params = params.get("ares", {})
dist_bound = float(ares_params.get("disturbance_bounds", 0.4))
level_zero = float(ares_params.get("level_zero", -0.15))

try:
    with np.load("normalization/pedestrian_latent_cache.npz") as latent_cache:
        latent_min = latent_cache["latent_min"].astype(np.float32)
        latent_max = latent_cache["latent_max"].astype(np.float32)
except FileNotFoundError:
    latent_min = None
    latent_max = None


# =========================================================================
# 2. Placeholder global telemetry -> robot-centric state/action histories
# =========================================================================
# Global robot state [x, y, theta, v, omega] over 4 timesteps, sampled at
# the game's dt (0.1 s, see game_parameters.yaml: game.dt).
robot_global_history = np.array([
    [0.00, 0.00, 0.00, 1.00, 0.00],
    [0.10, 0.00, 0.02, 1.00, 0.02],
    [0.20, 0.00, 0.05, 1.05, 0.03],
    [0.31, 0.01, 0.09, 1.10, 0.04],
], dtype=np.float32)

# Global (x, y) position of each pedestrian over the same 4 timesteps.
ped_global_history = np.array([
    # pedestrian 0
    [[2.0, 1.0], [1.9, 1.05], [1.8, 1.1], [1.7, 1.15]],
    # pedestrian 1
    [[-1.5, -2.0], [-1.4, -1.9], [-1.3, -1.8], [-1.2, -1.7]],
], dtype=np.float32)

# The robot's own applied [angular accel, accel] control between consecutive
# timesteps (physical units) -- this is the "action" paired with each state
# transition in pedestrianGame.py's sysid pipeline (see main()'s
# action_histories, populated from applied_control).
action_history = np.array([
    [0.20, 0.00],
    [0.10, 0.50],
    [0.15, 0.50],
], dtype=np.float32)

# Convert to robot-centric per-pedestrian state histories:
# state_t = [theta_t, v_t, omega_t, rel_x_t, rel_y_t], matching
# pedestrianGame.py:310-312,474-479.
ped_state_histories = []
for ped_idx in range(NUM_PEDESTRIANS):
    states = []
    for t in range(4):
        rel_x, rel_y = compute_relative_position(robot_global_history[t], ped_global_history[ped_idx, t])
        theta, v, omega = robot_global_history[t, 2], robot_global_history[t, 3], robot_global_history[t, 4]
        states.append(np.array([theta, v, omega, rel_x, rel_y], dtype=np.float32))
    ped_state_histories.append(np.stack(states))  # (4, 5)

print("Robot-centric state histories (per pedestrian, 4x5):")
for ped_idx, states in enumerate(ped_state_histories):
    print(f"  pedestrian {ped_idx}:\n{states}")


# =========================================================================
# 3. Sliding-window system-ID estimation with accept/reject
# =========================================================================

def estimate_system_id(states, actions):
    """Run supervised -> (denormalize) -> decoder -> build_system_id_model
    for one window of states/actions. Mirrors pedestrianGame.py:401-414."""
    trajectory = build_supervised_trajectory_input(states, actions)
    latent_out = run_supervised(supervised_sess, trajectory, io["supervised"])[0]
    latent_for_decoder = latent_out
    if latent_min is not None and latent_max is not None:
        latent_for_decoder = denormalize_latent(latent_out, latent_min, latent_max, LATENT_NORM)
    params_vec = run_decoder(decoder_sess, latent_for_decoder, io["decoder"])[0]
    model = build_system_id_model(params_vec)
    return model, latent_out.astype(np.float32)


latents = []           # winning 3-D latent per pedestrian
error_bounds = []      # winning 2-D disturbance bound per pedestrian

for ped_idx in range(NUM_PEDESTRIANS):
    states = ped_state_histories[ped_idx]

    # Window A (steps 0-2, actions 0-1): establishes the initial model.
    # There is no prior model yet, so this is always accepted -- exactly
    # like `current_models[idx] is None` in pedestrianGame.py:405-409.
    states_a, actions_a = states[0:3], action_history[0:2]
    current_model, current_latent = estimate_system_id(states_a, actions_a)
    print(f"\nPedestrian {ped_idx}: initial model estimated from window A, latent={current_latent}")

    # Window B (steps 1-3, actions 1-2): re-estimate a candidate model from
    # the newer window and check whether it is more accurate than the
    # current model *on that same window*, per pedestrianGame.py:397-436.
    states_b, actions_b = states[1:4], action_history[1:3]
    candidate_model, candidate_latent = estimate_system_id(states_b, actions_b)

    candidate_error = compute_error_bounds(candidate_model, states_b, actions_b)
    current_error = compute_error_bounds(current_model, states_b, actions_b)
    candidate_metric = float(np.mean(candidate_error))
    current_metric = float(np.mean(current_error))

    if candidate_metric < current_metric:
        current_model, current_latent, winning_error = candidate_model, candidate_latent, candidate_error
        print(f"Pedestrian {ped_idx}: ACCEPTED new estimation over window B "
              f"(candidate={candidate_metric:.5f} < current={current_metric:.5f})")
    else:
        winning_error = current_error
        print(f"Pedestrian {ped_idx}: REJECTED new estimation over window B "
              f"(candidate={candidate_metric:.5f} >= current={current_metric:.5f}), keeping prior model")

    error_bound = np.clip(winning_error, 0.0, dist_bound).astype(np.float32)
    latents.append(current_latent)
    error_bounds.append(error_bound)

latents = np.concatenate(latents)             # (6,) = latent_ped0(3) + latent_ped1(3)
error_bounds = np.concatenate(error_bounds)   # (4,) = bound_ped0(2) + bound_ped1(2)


# =========================================================================
# 4. Assemble the final observation for the control/disturbance/critic nets
# =========================================================================
# Order matches build_ares_observation: [robot_state(3), ped_rel_positions(4),
# latents(6), disturbance_bounds(4)] = 17.
final_robot_state = robot_global_history[-1]
rel_positions = []
for ped_idx in range(NUM_PEDESTRIANS):
    rel_x, rel_y = compute_relative_position(final_robot_state, ped_global_history[ped_idx, -1])
    rel_positions.extend([rel_x, rel_y])

obs = build_ares_observation(
    robot_state=final_robot_state[2:5],
    ped_rel_positions=np.array(rel_positions, dtype=np.float32),
    latents=latents,
    disturbance_bounds=error_bounds,
    duplicate_single=False,
)

print(f"\nFinal observation (17,): {obs}")


# =========================================================================
# 5. Run the disturbance, critic, and actor networks, apply the safety filter
# =========================================================================
disturbance_action = run_disturbance(disturbance_sess, obs, io["disturbance"])[0]

nominal_control = np.array([0.10, 0.30], dtype=np.float32)  # placeholder nominal control
norm_nominal = normalize_control(nominal_control)
critic_action = build_critic_action(norm_nominal, disturbance_action)
q_value = float(run_critic(critic_sess, obs, critic_action, io["critic"])[0][0])

print(f"\nDisturbance action (4,): {disturbance_action}")
print(f"Nominal control (physical): {nominal_control}")
print(f"Critic value for nominal control: {q_value:.4f} (level_zero={level_zero})")

if q_value < level_zero:
    actor_out = run_actor(actor_sess, obs, io["actor"])[0]
    applied_control = denormalize_control(actor_out)
    print(f"q_value below level_zero -> using actor's safe control: {applied_control}")
else:
    applied_control = nominal_control
    print(f"q_value above level_zero -> using nominal control: {applied_control}")
