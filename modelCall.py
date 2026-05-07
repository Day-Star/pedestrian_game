import os
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import yaml

from dynamics.pedestrianDynamics import get_control_bounds, get_state_bounds


def load_game_parameters(path: str = "game_parameters.yaml") -> Dict:
	with open(path, "r", encoding="utf-8") as handle:
		data = yaml.safe_load(handle) or {}
	return data


def _resolve_model_path(base_dir: str, filename: str) -> str:
	return os.path.join(base_dir, filename)


def get_model_paths(params: Dict) -> Dict[str, str]:
	models = params.get("models", {})
	base_dir = models.get("base_dir", "models")
	model_set = params.get("model_set", "log")
	model_set_map = models.get(model_set, {})

	return {
		"actor": _resolve_model_path(base_dir, model_set_map.get("actor", "log_actor.onnx")),
		"critic": _resolve_model_path(base_dir, model_set_map.get("critic", "log_critic.onnx")),
		"disturbance": _resolve_model_path(base_dir, model_set_map.get("disturbance", "log_disturbance.onnx")),
		"supervised": _resolve_model_path(base_dir, models.get("supervised", "pedestrian_supervised.onnx")),
		"decoder": _resolve_model_path(base_dir, models.get("decoder", "pedestrian_decoder.onnx")),
	}


@lru_cache(maxsize=16)
def get_onnx_session(path: str) -> ort.InferenceSession:
	if not os.path.exists(path):
		raise FileNotFoundError(f"ONNX model not found: {path}")
	return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _ensure_batch(x: np.ndarray) -> np.ndarray:
	if x.ndim == 1:
		return x[None, :]
	return x


def build_supervised_trajectory_input(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
	"""
	Build the supervised model input from a short trajectory.

	Expected format: [delta_0, action_0, delta_1, action_1, ...]
	where delta_t = state_{t+1} - state_t.
	"""
	states = np.asarray(states, dtype=np.float32)
	actions = np.asarray(actions, dtype=np.float32)

	if states.ndim != 2 or actions.ndim != 2:
		raise ValueError("states and actions must be 2D arrays")
	if states.shape[0] != actions.shape[0] + 1:
		raise ValueError("states must have one more step than actions")

	deltas = states[1:] - states[:-1]
	parts = []
	for idx in range(actions.shape[0]):
		parts.append(deltas[idx])
		parts.append(actions[idx])
	trajectory = np.concatenate(parts, axis=0).astype(np.float32)
	return trajectory


def build_ares_observation(
	robot_state: np.ndarray,
	ped_rel_positions: np.ndarray,
	latents: np.ndarray,
	disturbance_bounds: np.ndarray,
	duplicate_single: bool = True,
) -> np.ndarray:
	"""
	Build the normalized AReS observation vector for actor/critic/disturbance.
	The models expect two pedestrians and include disturbance bounds in the obs.
	"""
	robot_state = np.asarray(robot_state, dtype=np.float32).reshape(-1)
	ped_rel_positions = np.asarray(ped_rel_positions, dtype=np.float32).reshape(-1)
	latents = np.asarray(latents, dtype=np.float32).reshape(-1)
	disturbance_bounds = np.asarray(disturbance_bounds, dtype=np.float32).reshape(-1)

	if duplicate_single:
		if ped_rel_positions.size == 2:
			ped_rel_positions = np.tile(ped_rel_positions, 2)
		if latents.size == 3:
			latents = np.tile(latents, 2)
		if disturbance_bounds.size == 2:
			disturbance_bounds = np.tile(disturbance_bounds, 2)

	if robot_state.size != 3 or ped_rel_positions.size != 4:
		raise ValueError("Expected robot_state size 3 and ped_rel_positions size 4")
	if latents.size != 6 or disturbance_bounds.size != 4:
		raise ValueError("Expected latents size 6 and disturbance_bounds size 4")

	state = np.concatenate([robot_state, ped_rel_positions], axis=0)
	state_bounds = get_state_bounds(num_pedestrians=2)
	normalized_state = state / state_bounds

	obs = np.concatenate([normalized_state, latents, disturbance_bounds], axis=0).astype(np.float32)
	return obs


def normalize_control(control: np.ndarray) -> np.ndarray:
	bounds = get_control_bounds()
	return np.asarray(control, dtype=np.float32) / bounds


def denormalize_control(control: np.ndarray) -> np.ndarray:
	bounds = get_control_bounds()
	return np.asarray(control, dtype=np.float32) * bounds


def run_supervised(session: ort.InferenceSession, trajectory: np.ndarray, io: Dict) -> np.ndarray:
	trajectory = _ensure_batch(np.asarray(trajectory, dtype=np.float32))
	output = session.run([io["output"]], {io["input"]: trajectory})[0]
	return output


def run_decoder(session: ort.InferenceSession, latent: np.ndarray, io: Dict) -> np.ndarray:
	latent = _ensure_batch(np.asarray(latent, dtype=np.float32))
	output = session.run([io["output"]], {io["input"]: latent})[0]
	return output


def run_actor(session: ort.InferenceSession, obs: np.ndarray, io: Dict) -> np.ndarray:
	obs = _ensure_batch(np.asarray(obs, dtype=np.float32))
	output = session.run([io["output"]], {io["input"]: obs})[0]
	return output


def run_disturbance(session: ort.InferenceSession, obs: np.ndarray, io: Dict) -> np.ndarray:
	obs = _ensure_batch(np.asarray(obs, dtype=np.float32))
	output = session.run([io["output"]], {io["input"]: obs})[0]
	return output


def run_critic(
	session: ort.InferenceSession,
	obs: np.ndarray,
	action: np.ndarray,
	io: Dict,
) -> np.ndarray:
	obs = _ensure_batch(np.asarray(obs, dtype=np.float32))
	action = _ensure_batch(np.asarray(action, dtype=np.float32))
	output = session.run([io["output"]], {io["inputs"]["obs"]: obs, io["inputs"]["action"]: action})[0]
	return output


def build_critic_action(control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
	control = np.asarray(control, dtype=np.float32).reshape(-1)
	disturbance = np.asarray(disturbance, dtype=np.float32).reshape(-1)
	return np.concatenate([control, disturbance], axis=0)
