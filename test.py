import time

import numpy as np

from helperFunctions import mpc_nominal_control, simulate_robot_step, square_progress_to_pose
from modelCall import (
	build_critic_action,
	build_supervised_trajectory_input,
	get_model_paths,
	get_onnx_session,
	load_game_parameters,
	run_actor,
	run_critic,
	run_decoder,
	run_disturbance,
	run_supervised,
)


def test_model_inference() -> None:
	params = load_game_parameters()
	io = params.get("onnx_io", {})
	paths = get_model_paths(params)

	actor_sess = get_onnx_session(paths["actor"])
	critic_sess = get_onnx_session(paths["critic"])
	disturbance_sess = get_onnx_session(paths["disturbance"])
	supervised_sess = get_onnx_session(paths["supervised"])
	decoder_sess = get_onnx_session(paths["decoder"])

	obs_dim = int(io["actor"]["obs_dim"])
	action_dim = int(io["critic"]["action_dim"])
	disturbance_dim = int(io["disturbance"]["output_dim"])

	obs = np.zeros((1, obs_dim), dtype=np.float32)
	actor_out = run_actor(actor_sess, obs, io["actor"])
	disturbance_out = run_disturbance(disturbance_sess, obs, io["disturbance"])

	critic_action = build_critic_action(
		control=np.zeros(2, dtype=np.float32),
		disturbance=np.zeros(disturbance_dim, dtype=np.float32),
	).reshape(1, action_dim)
	critic_out = run_critic(critic_sess, obs, critic_action, io["critic"])

	states = np.zeros((3, 5), dtype=np.float32)
	actions = np.zeros((2, 2), dtype=np.float32)
	trajectory = build_supervised_trajectory_input(states, actions).reshape(1, -1)
	latent_out = run_supervised(supervised_sess, trajectory, io["supervised"])
	decoder_out = run_decoder(decoder_sess, latent_out, io["decoder"])

	print("Actor output shape:", actor_out.shape)
	print("Disturbance output shape:", disturbance_out.shape)
	print("Critic output shape:", critic_out.shape)
	print("Supervised latent shape:", latent_out.shape)
	print("Decoder output shape:", decoder_out.shape)


def test_mpc_runtime() -> None:
	params = load_game_parameters()
	nominal = params.get("nominal", {})
	target_speed = float(nominal.get("speed", 1.0))
	square_side = float(nominal.get("square_side", 4.0))

	robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
	progress = 0.0

	iterations = 10
	start = time.perf_counter()
	for _ in range(iterations):
		control, progress, info = mpc_nominal_control(
			robot_state,
			progress,
			path_type="line",
			params=nominal,
			target_speed=target_speed,
			square_side=square_side,
		)
		_ = control
	elapsed = time.perf_counter() - start

	avg = elapsed / iterations
	print(f"MPC runtime: {avg:.4f}s per call ({iterations} iterations)")


def test_mpc_square_tracking() -> None:
	params = load_game_parameters()
	game = params.get("game", {})
	nominal = params.get("nominal", {})
	perimeter = 4.0 * float(nominal.get("square_side", 4.0))
	target_speed = float(nominal.get("speed", 1.0))
	dt = float(game.get("dt", nominal.get("mpc", {}).get("dt", 0.1)))
	max_steps = int(game.get("max_episode_steps", 800))

	steps_needed = int(perimeter / max(target_speed * dt, 1e-6)) + 1
	steps = min(steps_needed, max_steps)

	robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
	progress = 0.0
	errors = []
	success_count = 0

	for _ in range(steps):
		control, progress, info = mpc_nominal_control(
			robot_state,
			progress,
			path_type="square",
			params=nominal,
			target_speed=target_speed,
			square_side=float(nominal.get("square_side", 4.0)),
		)
		robot_state = simulate_robot_step(robot_state, control, dt)
		x_ref, y_ref, _ = square_progress_to_pose(progress, float(nominal.get("square_side", 4.0)))
		errors.append(float(np.hypot(robot_state[0] - x_ref, robot_state[1] - y_ref)))
		if info.get("success"):
			success_count += 1

	avg_err = float(np.mean(errors)) if errors else 0.0
	max_err = float(np.max(errors)) if errors else 0.0

	print(
		"Square tracking: steps={}/{} avg_err={:.3f} max_err={:.3f} mpc_success={}".format(
			steps,
			steps_needed,
			avg_err,
			max_err,
			success_count,
		)
	)


if __name__ == "__main__":
	test_model_inference()
	test_mpc_runtime()
	test_mpc_square_tracking()
