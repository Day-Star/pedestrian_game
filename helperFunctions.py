import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from dynamics.pedestrianDynamics import get_control_bounds
from modelCall import (
	build_ares_observation,
	build_critic_action,
	run_actor,
	run_critic,
	run_disturbance,
)


def wrap_angle(angle: float) -> float:
	return (angle + math.pi) % (2 * math.pi) - math.pi


def simulate_robot_step(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
	"""Propagate robot state in global frame by one step.

	State: [x, y, theta, v, omega]
	Control: [alpha (angular accel), a (linear accel)]
	"""
	x, y, theta, v, omega = state
	alpha, accel = control

	x_next = x + v * math.cos(theta) * dt
	y_next = y + v * math.sin(theta) * dt
	theta_next = wrap_angle(theta + omega * dt)
	v_next = v + accel * dt
	omega_next = omega + alpha * dt

	return np.array([x_next, y_next, theta_next, v_next, omega_next], dtype=np.float32)


def square_progress_to_pose(progress: float, side: float) -> Tuple[float, float, float]:
	perim = 4.0 * side
	p = progress % perim
	half = side / 2.0
	if p < side:
		return -half + p, -half, 0.0
	if p < 2.0 * side:
		return half, -half + (p - side), math.pi / 2.0
	if p < 3.0 * side:
		return half - (p - 2.0 * side), half, math.pi
	return -half, half - (p - 3.0 * side), -math.pi / 2.0


def build_reference_trajectory(
	progress: float,
	horizon: int,
	dt: float,
	path_type: str,
	speed: float,
	square_side: float,
	target_pos: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
	refs = []
	if target_pos is None:
		target_pos = (0.0, 0.0)
	for step in range(horizon):
		step_progress = progress + speed * dt * (step + 1)
		if path_type == "square":
			x_ref, y_ref, theta_ref = square_progress_to_pose(step_progress, square_side)
		elif path_type == "point":
			x_ref, y_ref, theta_ref = float(target_pos[0]), float(target_pos[1]), 0.0
		else:
			x_ref, y_ref, theta_ref = step_progress, 0.0, 0.0
		refs.append([x_ref, y_ref, theta_ref])
	return np.asarray(refs, dtype=np.float32)


def mpc_nominal_control(
	robot_state: np.ndarray,
	progress: float,
	path_type: str,
	params: Dict,
	target_speed: float,
	square_side: float,
	target_pos: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float, Dict]:
	"""Solve a lightweight MPC problem for line, square, or point tracking."""
	mpc = params.get("mpc", {})
	horizon = int(mpc.get("horizon", 8))
	dt = float(mpc.get("dt", 0.1))
	max_iter = int(mpc.get("max_iter", 40))
	ftol = float(mpc.get("ftol", 1.0e-4))
	eps = float(mpc.get("eps", 1.0e-3))

	weights = {
		"pos": float(mpc.get("weight_pos", 5.0)),
		"heading": float(mpc.get("weight_heading", 1.0)),
		"speed": float(mpc.get("weight_speed", 2.0)),
		"omega": float(mpc.get("weight_omega", 0.2)),
		"control": float(mpc.get("weight_control", 0.01)),
	}

	refs = build_reference_trajectory(progress, horizon, dt, path_type, target_speed, square_side, target_pos)
	control_bounds = get_control_bounds()
	bounds = [
		(-control_bounds[0], control_bounds[0]),
		(-control_bounds[1], control_bounds[1]),
	] * horizon

	def cost(u_flat: np.ndarray) -> float:
		u_seq = u_flat.reshape(horizon, 2)
		state = robot_state.astype(np.float64, copy=True)
		total = 0.0
		for k in range(horizon):
			state = simulate_robot_step(state, u_seq[k], dt)
			x_ref, y_ref, theta_ref = refs[k]
			pos_err = (state[0] - x_ref) ** 2 + (state[1] - y_ref) ** 2
			heading_err = wrap_angle(state[2] - theta_ref) ** 2
			speed_err = (state[3] - target_speed) ** 2
			omega_err = state[4] ** 2
			control_pen = u_seq[k][0] ** 2 + u_seq[k][1] ** 2
			total += (
				weights["pos"] * pos_err
				+ weights["heading"] * heading_err
				+ weights["speed"] * speed_err
				+ weights["omega"] * omega_err
				+ weights["control"] * control_pen
			)
		return total

	u0 = np.zeros((horizon, 2), dtype=np.float64).reshape(-1)
	result = minimize(
		cost,
		u0,
		method="L-BFGS-B",
		bounds=bounds,
		options={"maxiter": max_iter, "ftol": ftol, "eps": eps},
	)

	if not result.success:
		control = np.zeros(2, dtype=np.float32)
	else:
		control = result.x[:2].astype(np.float32)

	if path_type == "point":
		next_progress = progress
	else:
		next_progress = progress + target_speed * dt
	info = {"success": bool(result.success), "iterations": result.nit if result.success else 0}
	return control, next_progress, info


def build_safe_set_grid(
	robot_state: np.ndarray,
	ped_positions: np.ndarray,
	latents: np.ndarray,
	error_bounds: np.ndarray,
	actor_sess,
	critic_sess,
	disturbance_sess,
	io: Dict,
	level_zero: float,
	duplicate_single: bool,
	grid_center: Tuple[float, float],
	grid_half: float,
	resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	ped_positions = np.asarray(ped_positions, dtype=np.float32)
	if ped_positions.ndim == 1:
		ped_positions = ped_positions.reshape(1, 2)
	ped_count = ped_positions.shape[0]
	latents = np.asarray(latents, dtype=np.float32)
	error_bounds = np.asarray(error_bounds, dtype=np.float32)
	if latents.ndim == 2:
		latents = latents.reshape(-1)
	if error_bounds.ndim == 2:
		error_bounds = error_bounds.reshape(-1)
	if ped_count == 1:
		duplicate_single = True
	if resolution < 2:
		resolution = 2
	xs = np.linspace(grid_center[0] - grid_half, grid_center[0] + grid_half, resolution, dtype=np.float32)
	ys = np.linspace(grid_center[1] - grid_half, grid_center[1] + grid_half, resolution, dtype=np.float32)
	safe_mask = np.zeros((resolution, resolution), dtype=bool)

	theta = float(robot_state[2])
	cos_t = math.cos(theta)
	sin_t = math.sin(theta)
	base_state = np.array([robot_state[2], robot_state[3], robot_state[4]], dtype=np.float32)

	for row, y in enumerate(ys):
		for col, x in enumerate(xs):
			all_far = True
			too_close = False
			rel_positions: List[float] = []
			for ped_pos in ped_positions:
				dx = ped_pos[0] - x
				dy = ped_pos[1] - y
				dist = math.hypot(dx, dy)
				if dist <= 10.0:
					all_far = False
				if dist < 1.0:
					too_close = True
					break
				rel_x = cos_t * dx + sin_t * dy
				rel_y = -sin_t * dx + cos_t * dy
				rel_positions.extend([rel_x, rel_y])
			if too_close:
				continue
			if all_far:
				safe_mask[row, col] = True
				continue
			obs = build_ares_observation(
				robot_state=base_state,
				ped_rel_positions=np.asarray(rel_positions, dtype=np.float32),
				latents=latents,
				disturbance_bounds=error_bounds,
				duplicate_single=duplicate_single,
			)
			actor_out = run_actor(actor_sess, obs, io["actor"])[0]
			disturbance = run_disturbance(disturbance_sess, obs, io["disturbance"])[0]
			critic_action = build_critic_action(actor_out, disturbance)
			q_value = run_critic(critic_sess, obs, critic_action, io["critic"])[0][0]
			safe_mask[row, col] = q_value >= level_zero - 0.1

	return xs, ys, safe_mask


def marching_squares_segments(
	safe_mask: np.ndarray,
	xs: np.ndarray,
	ys: np.ndarray,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
	segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
	rows, cols = safe_mask.shape
	for row in range(rows - 1):
		y0 = float(ys[row])
		y1 = float(ys[row + 1])
		for col in range(cols - 1):
			x0 = float(xs[col])
			x1 = float(xs[col + 1])
			bl = bool(safe_mask[row, col])
			br = bool(safe_mask[row, col + 1])
			tr = bool(safe_mask[row + 1, col + 1])
			tl = bool(safe_mask[row + 1, col])

			edge_points = {}
			if tl != tr:
				edge_points["top"] = ((x0 + x1) / 2.0, y1)
			if tr != br:
				edge_points["right"] = (x1, (y0 + y1) / 2.0)
			if br != bl:
				edge_points["bottom"] = ((x0 + x1) / 2.0, y0)
			if bl != tl:
				edge_points["left"] = (x0, (y0 + y1) / 2.0)

			if len(edge_points) == 2:
				keys = list(edge_points.keys())
				segments.append((edge_points[keys[0]], edge_points[keys[1]]))
			elif len(edge_points) == 4 and tl == br and tr == bl and tl != tr:
				if tl:
					pairs = [("top", "left"), ("bottom", "right")]
				else:
					pairs = [("top", "right"), ("bottom", "left")]
				for first, second in pairs:
					segments.append((edge_points[first], edge_points[second]))

	return segments


def _parse_text_float(value: str, default: float) -> Optional[float]:
	text = value.strip()
	if text in ("", "-", ".", "-."):
		return default
	try:
		return float(text)
	except ValueError:
		return None


def _handle_text_input(text: str, key: str) -> str:
	if key == "backspace":
		return text[:-1]
	if key in "0123456789":
		return text + key
	if key == "." and "." not in text:
		return text + key
	if key == "-" and text == "":
		return "-"
	return text


def startup_gui(params: Dict) -> Optional[Dict]:
	import pygame

	default_ped_x, default_ped_y = 5.0, 0.0
	ped_params = params.get("pedestrian", {})
	other_defaults = ped_params.get("other_pedestrian_start_pos", [[0.0, 0.0]])
	other_default = other_defaults[0] if other_defaults else [0.0, 0.0]
	default_ped2_x = float(other_default[0]) if len(other_default) > 0 else 0.0
	default_ped2_y = float(other_default[1]) if len(other_default) > 1 else 0.0
	ego_defaults = params.get("ego", {}).get("start_pos", [0.0, 0.0])
	default_ego_x = float(ego_defaults[0]) if len(ego_defaults) > 0 else 0.0
	default_ego_y = float(ego_defaults[1]) if len(ego_defaults) > 1 else 0.0

	pygame.init()
	width, height = 520, 620
	screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption("AReS Startup")
	font = pygame.font.SysFont("arial", 18)
	small = pygame.font.SysFont("arial", 14)
	clock = pygame.time.Clock()

	ped_x_text = f"{default_ped_x}"
	ped_y_text = f"{default_ped_y}"
	ped2_x_text = f"{default_ped2_x}"
	ped2_y_text = f"{default_ped2_y}"
	ego_x_text = f"{default_ego_x}"
	ego_y_text = f"{default_ego_y}"
	active_field = None

	ped_count_options = ["1", "2"]
	other_mode_options = ["npc", "manual"]
	online_options = ["yes", "no"]
	path_options = ["point", "line", "square"]
	input_options = ["arrows", "joystick"]
	default_count = int(ped_params.get("count", 1))
	default_mode = str(ped_params.get("other_pedestrian_mode", "npc")).strip().lower()
	ped_count_idx = 0 if default_count != 2 else 1
	other_mode_idx = other_mode_options.index(default_mode) if default_mode in other_mode_options else 0
	online_idx = 0
	path_idx = 0
	input_idx = 0

	ped_x_box = pygame.Rect(170, 60, 120, 32)
	ped_y_box = pygame.Rect(320, 60, 120, 32)
	ego_x_box = pygame.Rect(170, 130, 120, 32)
	ego_y_box = pygame.Rect(320, 130, 120, 32)
	ped2_x_box = pygame.Rect(170, 200, 120, 32)
	ped2_y_box = pygame.Rect(320, 200, 120, 32)
	ped_count_rect = pygame.Rect(40, 250, 440, 40)
	other_mode_rect = pygame.Rect(40, 310, 440, 40)
	online_rect = pygame.Rect(40, 370, 440, 40)
	path_rect = pygame.Rect(40, 430, 440, 40)
	input_rect = pygame.Rect(40, 490, 440, 40)
	start_rect = pygame.Rect(180, 550, 160, 50)

	error_text = ""
	running = True
	while running:
		clock.tick(30)
		ped2_enabled = ped_count_options[ped_count_idx] == "2"
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				return None
			if event.type == pygame.MOUSEBUTTONDOWN:
				if ped_x_box.collidepoint(event.pos):
					active_field = "ped_x"
				elif ped_y_box.collidepoint(event.pos):
					active_field = "ped_y"
				elif ego_x_box.collidepoint(event.pos):
					active_field = "ego_x"
				elif ego_y_box.collidepoint(event.pos):
					active_field = "ego_y"
				elif ped2_enabled and ped2_x_box.collidepoint(event.pos):
					active_field = "ped2_x"
				elif ped2_enabled and ped2_y_box.collidepoint(event.pos):
					active_field = "ped2_y"
				elif ped_count_rect.collidepoint(event.pos):
					ped_count_idx = (ped_count_idx + 1) % len(ped_count_options)
				elif other_mode_rect.collidepoint(event.pos):
					other_mode_idx = (other_mode_idx + 1) % len(other_mode_options)
				elif online_rect.collidepoint(event.pos):
					online_idx = (online_idx + 1) % len(online_options)
				elif path_rect.collidepoint(event.pos):
					path_idx = (path_idx + 1) % len(path_options)
				elif input_rect.collidepoint(event.pos):
					input_idx = (input_idx + 1) % len(input_options)
				elif start_rect.collidepoint(event.pos):
					ped_x = _parse_text_float(ped_x_text, default_ped_x)
					ped_y = _parse_text_float(ped_y_text, default_ped_y)
					ped2_x = _parse_text_float(ped2_x_text, default_ped2_x)
					ped2_y = _parse_text_float(ped2_y_text, default_ped2_y)
					ego_x = _parse_text_float(ego_x_text, default_ego_x)
					ego_y = _parse_text_float(ego_y_text, default_ego_y)
					if ped_x is None or ped_y is None or ego_x is None or ego_y is None:
						error_text = "Invalid start position"
						continue
					if ped_count_options[ped_count_idx] == "2" and (ped2_x is None or ped2_y is None):
						error_text = "Invalid second pedestrian position"
						continue
					if math.hypot(ped_x - ego_x, ped_y - ego_y) < 2.0:
						error_text = "Pedestrian must be >= 2m from ego"
						continue
					if ped_count_options[ped_count_idx] == "2":
						if math.hypot(ped2_x - ego_x, ped2_y - ego_y) < 2.0:
							error_text = "Second pedestrian must be >= 2m from ego"
							continue
					pygame.quit()
					return {
						"ped_init": (ped_x, ped_y),
						"other_ped_init": (ped2_x, ped2_y),
						"ego_init": (ego_x, ego_y),
						"ped_count": int(ped_count_options[ped_count_idx]),
						"other_ped_mode": other_mode_options[other_mode_idx],
						"online_estimation": online_options[online_idx] == "yes",
						"path_choice": path_options[path_idx],
						"input_choice": input_options[input_idx],
					}
				else:
					active_field = None
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					pygame.quit()
					return None
				key = pygame.key.name(event.key)
				if active_field == "ped_x":
					ped_x_text = _handle_text_input(ped_x_text, key)
				elif active_field == "ped_y":
					ped_y_text = _handle_text_input(ped_y_text, key)
				elif active_field == "ped2_x":
					ped2_x_text = _handle_text_input(ped2_x_text, key)
				elif active_field == "ped2_y":
					ped2_y_text = _handle_text_input(ped2_y_text, key)
				elif active_field == "ego_x":
					ego_x_text = _handle_text_input(ego_x_text, key)
				elif active_field == "ego_y":
					ego_y_text = _handle_text_input(ego_y_text, key)

		screen.fill((245, 245, 245))
		screen.blit(font.render("Initial pedestrian location (x, y)", True, (20, 20, 20)), (40, 30))
		pygame.draw.rect(screen, (255, 255, 255), ped_x_box)
		pygame.draw.rect(screen, (255, 255, 255), ped_y_box)
		pygame.draw.rect(screen, (60, 60, 60), ped_x_box, 2 if active_field == "ped_x" else 1)
		pygame.draw.rect(screen, (60, 60, 60), ped_y_box, 2 if active_field == "ped_y" else 1)
		screen.blit(font.render(ped_x_text, True, (10, 10, 10)), (ped_x_box.x + 8, ped_x_box.y + 6))
		screen.blit(font.render(ped_y_text, True, (10, 10, 10)), (ped_y_box.x + 8, ped_y_box.y + 6))
		screen.blit(small.render("X", True, (60, 60, 60)), (ped_x_box.x - 18, ped_x_box.y + 8))
		screen.blit(small.render("Y", True, (60, 60, 60)), (ped_y_box.x - 18, ped_y_box.y + 8))

		screen.blit(font.render("Initial ego location (x, y)", True, (20, 20, 20)), (40, 100))
		pygame.draw.rect(screen, (255, 255, 255), ego_x_box)
		pygame.draw.rect(screen, (255, 255, 255), ego_y_box)
		pygame.draw.rect(screen, (60, 60, 60), ego_x_box, 2 if active_field == "ego_x" else 1)
		pygame.draw.rect(screen, (60, 60, 60), ego_y_box, 2 if active_field == "ego_y" else 1)
		screen.blit(font.render(ego_x_text, True, (10, 10, 10)), (ego_x_box.x + 8, ego_x_box.y + 6))
		screen.blit(font.render(ego_y_text, True, (10, 10, 10)), (ego_y_box.x + 8, ego_y_box.y + 6))
		screen.blit(small.render("X", True, (60, 60, 60)), (ego_x_box.x - 18, ego_x_box.y + 8))
		screen.blit(small.render("Y", True, (60, 60, 60)), (ego_y_box.x - 18, ego_y_box.y + 8))

		ped2_enabled = ped_count_options[ped_count_idx] == "2"
		ped2_label = "Second pedestrian location (x, y)"
		screen.blit(font.render(ped2_label, True, (20, 20, 20)), (40, 170))
		ped2_fill = (255, 255, 255) if ped2_enabled else (230, 230, 230)
		pygame.draw.rect(screen, ped2_fill, ped2_x_box)
		pygame.draw.rect(screen, ped2_fill, ped2_y_box)
		pygame.draw.rect(screen, (60, 60, 60), ped2_x_box, 2 if active_field == "ped2_x" else 1)
		pygame.draw.rect(screen, (60, 60, 60), ped2_y_box, 2 if active_field == "ped2_y" else 1)
		screen.blit(font.render(ped2_x_text, True, (10, 10, 10)), (ped2_x_box.x + 8, ped2_x_box.y + 6))
		screen.blit(font.render(ped2_y_text, True, (10, 10, 10)), (ped2_y_box.x + 8, ped2_y_box.y + 6))
		screen.blit(small.render("X", True, (60, 60, 60)), (ped2_x_box.x - 18, ped2_x_box.y + 8))
		screen.blit(small.render("Y", True, (60, 60, 60)), (ped2_y_box.x - 18, ped2_y_box.y + 8))

		pygame.draw.rect(screen, (230, 230, 230), ped_count_rect)
		pygame.draw.rect(screen, (230, 230, 230), other_mode_rect)
		pygame.draw.rect(screen, (230, 230, 230), online_rect)
		pygame.draw.rect(screen, (230, 230, 230), path_rect)
		pygame.draw.rect(screen, (230, 230, 230), input_rect)
		pygame.draw.rect(screen, (60, 60, 60), ped_count_rect, 1)
		pygame.draw.rect(screen, (60, 60, 60), other_mode_rect, 1)
		pygame.draw.rect(screen, (60, 60, 60), online_rect, 1)
		pygame.draw.rect(screen, (60, 60, 60), path_rect, 1)
		pygame.draw.rect(screen, (60, 60, 60), input_rect, 1)

		ped_count_label = f"Pedestrian count: {ped_count_options[ped_count_idx]}"
		other_mode_label = f"Other pedestrian mode: {other_mode_options[other_mode_idx].upper()}"
		online_label = f"Online model error estimation: {online_options[online_idx].upper()}"
		path_label = f"Robot trajectory: {path_options[path_idx].upper()}"
		input_label = f"Input mode: {input_options[input_idx].upper()}"
		screen.blit(font.render(ped_count_label, True, (20, 20, 20)), (ped_count_rect.x + 10, ped_count_rect.y + 10))
		screen.blit(font.render(other_mode_label, True, (20, 20, 20)), (other_mode_rect.x + 10, other_mode_rect.y + 10))
		screen.blit(font.render(online_label, True, (20, 20, 20)), (online_rect.x + 10, online_rect.y + 10))
		screen.blit(font.render(path_label, True, (20, 20, 20)), (path_rect.x + 10, path_rect.y + 10))
		screen.blit(font.render(input_label, True, (20, 20, 20)), (input_rect.x + 10, input_rect.y + 10))

		pygame.draw.rect(screen, (80, 170, 80), start_rect)
		screen.blit(font.render("Start", True, (255, 255, 255)), (start_rect.x + 48, start_rect.y + 12))

		if error_text:
			screen.blit(small.render(error_text, True, (160, 40, 40)), (40, 600))

		pygame.display.flip()

	pygame.quit()
	return None
