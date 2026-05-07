import math
import os
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import pygame
import torch
from torch import nn

from helperFunctions import (
	build_safe_set_grid,
	marching_squares_segments,
	mpc_nominal_control,
	simulate_robot_step,
	startup_gui,
	wrap_angle,
)
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


class SystemIDNetwork(nn.Module):
	def __init__(self, input_dim: int = 7, output_dim: int = 2, hidden_layers: Tuple[int, int] = (8, 6)):
		super().__init__()
		layers = []
		prev_dim = input_dim
		for size in hidden_layers:
			layers.append(nn.Linear(prev_dim, size))
			layers.append(nn.ReLU())
			prev_dim = size
		layers.append(nn.Linear(prev_dim, output_dim))
		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)


def build_system_id_model(params: np.ndarray) -> SystemIDNetwork:
	model = SystemIDNetwork()
	flat = np.asarray(params, dtype=np.float32).reshape(-1)
	expected = sum(p.numel() for p in model.parameters())
	if flat.size != expected:
		raise ValueError(f"Decoder output size {flat.size} does not match expected {expected}")
	idx = 0
	with torch.no_grad():
		for param in model.parameters():
			count = param.numel()
			values = flat[idx:idx + count].reshape(param.shape)
			param.copy_(torch.from_numpy(values))
			idx += count
	model.eval()
	return model


def compute_error_bounds(
	model: SystemIDNetwork,
	states: np.ndarray,
	actions: np.ndarray,
) -> np.ndarray:
	errors = []
	with torch.no_grad():
		for idx in range(actions.shape[0]):
			inputs = np.concatenate([states[idx], actions[idx]], axis=0).astype(np.float32)
			pred = model(torch.from_numpy(inputs).unsqueeze(0)).squeeze(0).cpu().numpy()
			actual = states[idx + 1][3:5]
			errors.append(np.abs(pred - actual))
	return np.mean(np.asarray(errors), axis=0).astype(np.float32)*.1


def prompt_float_pair(prompt: str, default: Tuple[float, float]) -> Tuple[float, float]:
	while True:
		raw = input(f"{prompt} (default {default[0]}, {default[1]}): ").strip()
		if not raw:
			return default
		parts = raw.split(",")
		if len(parts) != 2:
			print("Please enter as x,y")
			continue
		try:
			return float(parts[0]), float(parts[1])
		except ValueError:
			print("Invalid number format")


def prompt_choice(prompt: str, options: Tuple[str, ...], default: str) -> str:
	options_lower = tuple(opt.lower() for opt in options)
	while True:
		raw = input(f"{prompt} ({'/'.join(options)}) [default {default}]: ").strip().lower()
		if not raw:
			return default
		if raw in options_lower:
			return raw
		print("Invalid selection")


def compute_relative_position(robot_state: np.ndarray, ped_pos: np.ndarray) -> Tuple[float, float]:
	dx = ped_pos[0] - robot_state[0]
	dy = ped_pos[1] - robot_state[1]
	theta = robot_state[2]
	cos_t = math.cos(theta)
	sin_t = math.sin(theta)
	x_rel = cos_t * dx + sin_t * dy
	y_rel = -sin_t * dx + cos_t * dy
	return x_rel, y_rel


def world_to_screen(pos: np.ndarray, arena_center: Tuple[float, float], ppm: float) -> Tuple[int, int]:
	x_px = arena_center[0] + pos[0] * ppm
	y_px = arena_center[1] - pos[1] * ppm
	return int(x_px), int(y_px)


def draw_dashed_circle(surface, color, center, radius, dash_len=10, gap_len=6, width=2):
	circumference = 2 * math.pi * radius
	if circumference <= 0:
		return
	dash_angle = dash_len / radius
	gap_angle = gap_len / radius
	angle = 0.0
	rect = pygame.Rect(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)
	while angle < 2 * math.pi:
		start = angle
		end = min(angle + dash_angle, 2 * math.pi)
		pygame.draw.arc(surface, color, rect, start, end, width)
		angle += dash_angle + gap_angle


def clamp_vector(vec: np.ndarray, max_mag: float) -> np.ndarray:
	norm = np.linalg.norm(vec)
	if norm == 0.0 or norm <= max_mag:
		return vec
	return vec * (max_mag / norm)


def denormalize_latent(latent: np.ndarray, latent_min: np.ndarray, latent_max: np.ndarray, latent_norm: float) -> np.ndarray:
	latent = np.asarray(latent, dtype=np.float32)
	span = latent_max - latent_min
	span = np.where(span == 0.0, 1.0, span)
	return ((latent + latent_norm) / (2.0 * latent_norm)) * span + latent_min


def build_safe_set_surface(
	safe_mask: np.ndarray,
	xs: np.ndarray,
	ys: np.ndarray,
	arena_center: Tuple[float, float],
	ppm: float,
	screen_size: Tuple[int, int],
) -> pygame.Surface:
	overlay = pygame.Surface(screen_size, pygame.SRCALPHA)
	if xs.size < 2 or ys.size < 2:
		return overlay

	dx = float(abs(xs[1] - xs[0]))
	dy = float(abs(ys[1] - ys[0]))
	half_w = max(int(dx * ppm / 2.0), 1)
	half_h = max(int(dy * ppm / 2.0), 1)
	safe_color = (60, 190, 90, 70)
	unsafe_color = (220, 80, 80, 50)

	for row, y in enumerate(ys):
		for col, x in enumerate(xs):
			px, py = world_to_screen(np.array([x, y], dtype=np.float32), arena_center, ppm)
			rect = pygame.Rect(px - half_w, py - half_h, 2 * half_w, 2 * half_h)
			color = safe_color if safe_mask[row, col] else unsafe_color
			pygame.draw.rect(overlay, color, rect)

	return overlay


def main() -> None:

	# Load parameters from YAML and set up variables
	params = load_game_parameters()
	game_params = params.get("game", {})
	ego_params = params.get("ego", {})
	nominal_params = params.get("nominal", {})
	ares_params = params.get("ares", {})
	pedestrian_params = params.get("pedestrian", {})
	io = params.get("onnx_io", {})

	# Game settings
	dt = float(game_params.get("dt", 0.1))
	max_steps = int(game_params.get("max_episode_steps", 800))
	arena_size = float(game_params.get("arena_size", 10.0))
	ppm = float(game_params.get("pixels_per_meter", 60.0))
	banner_height = int(game_params.get("banner_height_px", 80))
	draw_target = bool(game_params.get("draw_target", True))
	draw_safe_set = bool(game_params.get("draw_safe_set", True))
	target_pos = np.array(game_params.get("target_pos", [10.0, 10.0]), dtype=np.float32)
	arena_half = arena_size / 2.0

	# Ego Settings
	ego_max_speed = float(ego_params.get("max_speed", 2.0))
	ego_start_pos = np.array(ego_params.get("start_pos", [0.0, 0.0]), dtype=np.float32)
	ego_start_heading = float(ego_params.get("start_heading", 0.0))

	max_speed = float(pedestrian_params.get("max_speed", 1.5))
	other_ped_speed = np.array(pedestrian_params.get("other_pedestrian_speed", [0.0, 0.0]), dtype=np.float32)
	other_ped_start_defaults = pedestrian_params.get("other_pedestrian_start_pos", [[0.0, 0.0]])
	other_ped_start = other_ped_start_defaults[0] if other_ped_start_defaults else [0.0, 0.0]
	target_speed = float(nominal_params.get("speed", 1.0))
	square_side = float(nominal_params.get("square_side", 4.0))

	dist_bound = float(ares_params.get("disturbance_bounds", 0.5))
	level_zero = float(ares_params.get("level_zero", -0.15))
	duplicate_single = bool(ares_params.get("duplicate_single_pedestrian", True))
	safe_set_type = str(ares_params.get("safe_set_type", "local")).strip().lower()
	safe_set_resolution = int(ares_params.get("safe_set_resolution", 20))
	safe_set_display_type = str(ares_params.get("safe_set_display_type", "square")).strip().lower()
	safe_set_refresh_frames = int(ares_params.get("safe_set_refresh_frames", 1))
	pipeline_every_step = bool(ares_params.get("pipeline_every_step", True))
	pipeline_period = int(ares_params.get("pipeline_period_steps", 1))
	error_horizon = int(ares_params.get("error_horizon_steps", 3))
	latent_norm = 5.0
	latent_min = None
	latent_max = None
	latent_cache_path = os.path.join("normalization", "pedestrian_latent_cache.npz")
	if os.path.exists(latent_cache_path):
		cache = np.load(latent_cache_path)
		latent_min = cache.get("latent_min")
		latent_max = cache.get("latent_max")

	os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
	selection = startup_gui(params)
	if selection is None:
		return
	ped_init = selection["ped_init"]
	other_ped_init = selection.get("other_ped_init")
	ped_count = int(selection.get("ped_count", pedestrian_params.get("count", 1)))
	other_ped_mode = str(
		selection.get("other_ped_mode", pedestrian_params.get("other_pedestrian_mode", "npc"))
	).strip().lower()
	if ped_count not in (1, 2):
		ped_count = 1
	if other_ped_mode not in ("npc", "manual"):
		other_ped_mode = "npc"
	ego_init = selection.get("ego_init", ego_start_pos)
	online_estimation = selection["online_estimation"]
	path_choice = selection["path_choice"]
	use_joystick = selection["input_choice"] == "joystick"

	# Initialize pygame and set up the display
	pygame.init()
	arena_px = int(arena_size * ppm)
	width = arena_px
	height = arena_px + banner_height
	screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption("AReS Pedestrian Game")
	clock = pygame.time.Clock()
	font = pygame.font.SysFont("arial", 18)
	small_font = pygame.font.SysFont("arial", 14)

	arena_center = (arena_px / 2.0, banner_height + arena_px / 2.0)
	joystick_center = np.array([arena_px - 120, banner_height + arena_px - 120], dtype=np.float32)
	joystick_radius = 70.0
	dragging = False

	model_paths = get_model_paths(params)
	actor_sess = get_onnx_session(model_paths["actor"])
	critic_sess = get_onnx_session(model_paths["critic"])
	disturbance_sess = get_onnx_session(model_paths["disturbance"])
	supervised_sess = get_onnx_session(model_paths["supervised"]) if online_estimation else None
	decoder_sess = get_onnx_session(model_paths["decoder"]) if online_estimation else None

	robot_state = np.array(
		[float(ego_init[0]), float(ego_init[1]), ego_start_heading, 0.0, 0.0],
		dtype=np.float32,
	)
	if other_ped_init is None:
		other_ped_init = (float(other_ped_start[0]), float(other_ped_start[1]))
	ped_positions = [np.array([ped_init[0], ped_init[1]], dtype=np.float32)]
	if ped_count == 2:
		ped_positions.append(np.array([other_ped_init[0], other_ped_init[1]], dtype=np.float32))
	ped_positions = np.stack(ped_positions, axis=0)
	progress = 0.0
	nominal_control = np.zeros(2, dtype=np.float32)
	applied_control = np.zeros(2, dtype=np.float32)
	latents = np.zeros((ped_count, 3), dtype=np.float32)
	error_bounds = np.full(
		(ped_count, 2),
		dist_bound if not online_estimation else 0.0,
		dtype=np.float32,
	)
	current_models = [None for _ in range(ped_count)]
	current_error_metrics = [None for _ in range(ped_count)]

	history_len = max(error_horizon, 2)
	action_len = max(history_len - 1, 1)
	state_histories = [deque(maxlen=history_len) for _ in range(ped_count)]
	action_histories = [deque(maxlen=action_len) for _ in range(ped_count)]
	last_pipeline_steps = [-9999 for _ in range(ped_count)]
	for idx in range(ped_count):
		init_rel_x, init_rel_y = compute_relative_position(robot_state, ped_positions[idx])
		state_histories[idx].append(
			np.array([robot_state[2], robot_state[3], robot_state[4], init_rel_x, init_rel_y], dtype=np.float32)
		)

	elapsed_steps = 0
	frame_count = 0
	filter_active = False
	collision = False
	running = True
	paused = False
	accumulator = 0.0
	safe_set_grid_half = 5.0
	safe_set_overlay = None
	safe_set_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
	safe_set_xs = None
	safe_set_ys = None
	last_safe_set_frame = -9999

	while running:
		dt_real = clock.tick(60) / 1000.0
		accumulator += dt_real

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_j:
					use_joystick = not use_joystick
				elif event.key == pygame.K_p:
					paused = not paused
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if use_joystick:
					mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float32)
					if np.linalg.norm(mouse_pos - joystick_center) <= joystick_radius:
						dragging = True
			elif event.type == pygame.MOUSEBUTTONUP:
				dragging = False

		ped_vels = np.zeros((ped_count, 2), dtype=np.float32)
		if use_joystick:
			if dragging:
				mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float32)
				offset = mouse_pos - joystick_center
				offset[1] = -offset[1]
				offset = clamp_vector(offset, joystick_radius)
				ped_vels[0] = (offset / joystick_radius) * max_speed
		else:
			keys = pygame.key.get_pressed()
			if keys[pygame.K_LEFT]:
				ped_vels[0][0] -= max_speed
			if keys[pygame.K_RIGHT]:
				ped_vels[0][0] += max_speed
			if keys[pygame.K_UP]:
				ped_vels[0][1] += max_speed
			if keys[pygame.K_DOWN]:
				ped_vels[0][1] -= max_speed
			ped_vels[0] = clamp_vector(ped_vels[0], max_speed)

		if ped_count == 2:
			if other_ped_mode == "npc":
				ped_vels[1] = clamp_vector(other_ped_speed, max_speed)
			else:
				keys = pygame.key.get_pressed()
				if keys[pygame.K_a]:
					ped_vels[1][0] -= max_speed
				if keys[pygame.K_d]:
					ped_vels[1][0] += max_speed
				if keys[pygame.K_w]:
					ped_vels[1][1] += max_speed
				if keys[pygame.K_s]:
					ped_vels[1][1] -= max_speed
				ped_vels[1] = clamp_vector(ped_vels[1], max_speed)

		if paused:
			accumulator = 0.0
		else:
			while accumulator >= dt and running:
				accumulator -= dt
				elapsed_steps += 1

				rel_positions: List[float] = []
				for idx in range(ped_count):
					rel_x, rel_y = compute_relative_position(robot_state, ped_positions[idx])
					rel_positions.extend([rel_x, rel_y])

				if online_estimation:
					for idx in range(ped_count):
						if len(state_histories[idx]) >= 3 and len(action_histories[idx]) >= 2:
							if pipeline_every_step or (elapsed_steps - last_pipeline_steps[idx] >= pipeline_period):
								states = np.stack(list(state_histories[idx])[-3:], axis=0)
								actions = np.stack(list(action_histories[idx])[-2:], axis=0)
								trajectory = build_supervised_trajectory_input(states, actions)
								latent_out = run_supervised(supervised_sess, trajectory, io["supervised"])[0]
								latent_for_decoder = latent_out
								if latent_min is not None and latent_max is not None:
									latent_for_decoder = denormalize_latent(
										latent_out,
										latent_min,
										latent_max,
										latent_norm,
									)
								params_vec = run_decoder(decoder_sess, latent_for_decoder, io["decoder"])[0]
								candidate_model = build_system_id_model(params_vec)
								candidate_error = compute_error_bounds(candidate_model, states, actions)
								candidate_metric = float(np.mean(candidate_error))

								if current_models[idx] is None:
									current_models[idx] = candidate_model
									current_error_metrics[idx] = candidate_metric
									latents[idx] = latent_out.astype(np.float32)
									error_bounds[idx] = candidate_error
								else:
									current_error = compute_error_bounds(current_models[idx], states, actions)
									current_metric = float(np.mean(current_error))
									if candidate_metric < current_metric:
										current_models[idx] = candidate_model
										current_error_metrics[idx] = candidate_metric
										latents[idx] = latent_out.astype(np.float32)
										error_bounds[idx] = candidate_error
									else:
										current_error_metrics[idx] = current_metric
										error_bounds[idx] = current_error

								error_bounds[idx] = np.clip(error_bounds[idx], 0.0, dist_bound)
								last_pipeline_steps[idx] = elapsed_steps
				else:
					error_bounds[:] = dist_bound

				nominal_control, progress, _ = mpc_nominal_control(
					robot_state,
					progress,
					path_type=path_choice,
					params=nominal_params,
					target_speed=target_speed,
					square_side=square_side,
					target_pos=(float(target_pos[0]), float(target_pos[1])),
				)

				single_dup = duplicate_single and ped_count == 1
				obs = build_ares_observation(
					robot_state=np.array([robot_state[2], robot_state[3], robot_state[4]], dtype=np.float32),
					ped_rel_positions=np.array(rel_positions, dtype=np.float32),
					latents=latents.reshape(-1),
					disturbance_bounds=error_bounds.reshape(-1),
					duplicate_single=single_dup,
				)

				disturbance = run_disturbance(disturbance_sess, obs, io["disturbance"])[0]
				norm_nominal = normalize_control(nominal_control)
				critic_action = build_critic_action(norm_nominal, disturbance)
				q_value = run_critic(critic_sess, obs, critic_action, io["critic"])[0][0]

				if q_value < level_zero:
					actor_out = run_actor(actor_sess, obs, io["actor"])[0]
					applied_control = denormalize_control(actor_out)
					filter_active = True
				else:
					applied_control = nominal_control
					filter_active = False

				robot_state = simulate_robot_step(robot_state, applied_control, dt)
				ped_positions = ped_positions + ped_vels * dt
				for idx in range(ped_count):
					action_histories[idx].append(applied_control)
					new_rel_x, new_rel_y = compute_relative_position(robot_state, ped_positions[idx])
					state_histories[idx].append(
						np.array([robot_state[2], robot_state[3], robot_state[4], new_rel_x, new_rel_y], dtype=np.float32)
					)

				for idx in range(ped_count):
					dx = robot_state[0] - ped_positions[idx][0]
					dy = robot_state[1] - ped_positions[idx][1]
					if math.hypot(dx, dy) <= 0.95:
						collision = True
						running = False
						break
				if collision:
					break
				if abs(robot_state[0]) > arena_half or abs(robot_state[1]) > arena_half:
					running = False
					break
				if elapsed_steps >= max_steps:
					running = False
					break

		frame_count += 1
		if draw_safe_set and (frame_count - last_safe_set_frame >= max(safe_set_refresh_frames, 1)):
			grid_center = (float(robot_state[0]), float(robot_state[1]))
			grid_half = safe_set_grid_half
			if safe_set_type == "whole":
				grid_center = (0.0, 0.0)
				grid_half = arena_half
			single_dup = duplicate_single and ped_count == 1
			safe_set_xs, safe_set_ys, safe_set_mask = build_safe_set_grid(
				robot_state=robot_state,
				ped_positions=ped_positions,
				latents=latents.reshape(-1),
				error_bounds=error_bounds.reshape(-1),
				actor_sess=actor_sess,
				critic_sess=critic_sess,
				disturbance_sess=disturbance_sess,
				io=io,
				level_zero=level_zero,
				duplicate_single=single_dup,
				grid_center=grid_center,
				grid_half=grid_half,
				resolution=safe_set_resolution,
			)
			if safe_set_display_type in ("line", "lines"):
				safe_set_segments = marching_squares_segments(safe_set_mask, safe_set_xs, safe_set_ys)
				safe_set_overlay = None
			else:
				safe_set_overlay = build_safe_set_surface(
					safe_set_mask,
					safe_set_xs,
					safe_set_ys,
					arena_center,
					ppm,
					screen.get_size(),
				)
				safe_set_segments = []
			last_safe_set_frame = frame_count

		screen.fill((255, 255, 255))

		banner_color = (40, 180, 40)
		status_text = "Filter: Off"
		if collision:
			banner_color = (200, 40, 40)
			status_text = "Filter: Collision"
		elif paused:
			banner_color = (70, 120, 200)
			status_text = "Paused"
		elif filter_active:
			banner_color = (230, 140, 40)
			status_text = "Filter: On"

		pygame.draw.rect(screen, banner_color, pygame.Rect(0, 0, width, banner_height))

		if online_estimation:
			if ped_count == 2:
				err_lines = [
					f"Error bounds p1: ({error_bounds[0][0]:.2f}, {error_bounds[0][1]:.2f})",
					f"Error bounds p2: ({error_bounds[1][0]:.2f}, {error_bounds[1][1]:.2f})",
				]
			else:
				err_lines = [f"Error bounds: ({error_bounds[0][0]:.2f}, {error_bounds[0][1]:.2f})"]
		else:
			err_lines = ["Error bounds: disabled"]
		banner_lines = [status_text] + err_lines
		for idx, line in enumerate(banner_lines):
			text = font.render(line, True, (10, 10, 10))
			screen.blit(text, (12, 12 + idx * 22))

		key_lines = ["Keys: Esc quit | P pause | J toggle input"]
		if use_joystick:
			key_lines.append("Mouse drag joystick for ped1")
		else:
			key_lines.append("Arrows move ped1")
		if ped_count == 2:
			if other_ped_mode == "manual":
				key_lines.append("WASD move ped2")
			else:
				key_lines.append("Ped2 uses NPC velocity")
		for idx, line in enumerate(key_lines):
			text = small_font.render(line, True, (10, 10, 10))
			screen.blit(text, (width - text.get_width() - 12, 12 + idx * 18))

		arena_rect = pygame.Rect(0, banner_height, arena_px, arena_px)
		pygame.draw.rect(screen, (245, 245, 245), arena_rect)
		pygame.draw.rect(screen, (0, 0, 0), arena_rect, 2)

		if draw_safe_set:
			if safe_set_overlay is not None:
				screen.blit(safe_set_overlay, (0, 0))
			elif safe_set_segments:
				for seg_start, seg_end in safe_set_segments:
					start_px = world_to_screen(np.array(seg_start, dtype=np.float32), arena_center, ppm)
					end_px = world_to_screen(np.array(seg_end, dtype=np.float32), arena_center, ppm)
					pygame.draw.line(screen, (40, 150, 70), start_px, end_px, 2)

		if path_choice == "square":
			side = square_side
			half = side / 2.0
			square_points = [
				(-half, -half),
				(half, -half),
				(half, half),
				(-half, half),
				(-half, -half),
			]
			square_screen = [world_to_screen(np.array(p), arena_center, ppm) for p in square_points]
			pygame.draw.lines(screen, (90, 90, 90), False, square_screen, 2)
		#else:
			#line_start = world_to_screen(np.array([0.0, 0.0]), arena_center, ppm)
			#line_end = world_to_screen(np.array([arena_half, 0.0]), arena_center, ppm)
			#pygame.draw.line(screen, (90, 90, 90), line_start, line_end, 2)

		robot_px = world_to_screen(robot_state[:2], arena_center, ppm)
		ped_pixels = [world_to_screen(ped_positions[idx], arena_center, ppm) for idx in range(ped_count)]
		target_px = world_to_screen(target_pos, arena_center, ppm)
		collision_radius_px = int(1.0 * ppm)
		ped_colors = [(200, 50, 50), (255, 140, 0)]

		#draw_dashed_circle(screen, (0, 0, 0), robot_px, collision_radius_px, width=2)
		for idx, ped_px in enumerate(ped_pixels):
			draw_dashed_circle(screen, (0, 0, 0), ped_px, collision_radius_px, width=2)
			pygame.draw.circle(screen, ped_colors[idx], ped_px, 6)

		pygame.draw.circle(screen, (40, 180, 40), robot_px, 6)
		if draw_target:
			pygame.draw.circle(screen, (0, 0, 0), target_px, 6)

		heading = np.array([math.cos(robot_state[2]), math.sin(robot_state[2])], dtype=np.float32)
		arrow_scale = 0.6
		nominal_vec = heading * nominal_control[1] * arrow_scale
		safe_vec = heading * applied_control[1] * arrow_scale

		nominal_end = world_to_screen(robot_state[:2] + nominal_vec, arena_center, ppm)
		safe_end = world_to_screen(robot_state[:2] + safe_vec, arena_center, ppm)
		pygame.draw.line(screen, (40, 120, 200), robot_px, nominal_end, 3)
		if filter_active:
			pygame.draw.line(screen, (230, 140, 40), robot_px, safe_end, 3)

		if use_joystick:
			pygame.draw.circle(screen, (255, 255, 255), joystick_center.astype(int), int(joystick_radius))
			pygame.draw.circle(screen, (0, 0, 0), joystick_center.astype(int), 4)

		pygame.display.flip()

	pygame.quit()


if __name__ == "__main__":
	main()
