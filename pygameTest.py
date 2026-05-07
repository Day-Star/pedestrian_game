import os
import time

import numpy as np
import pygame

from modelCall import load_game_parameters


def clamp_vector(vec: np.ndarray, max_mag: float) -> np.ndarray:
	norm = np.linalg.norm(vec)
	if norm <= max_mag or norm == 0.0:
		return vec
	return vec * (max_mag / norm)


def main() -> None:
	params = load_game_parameters()
	max_speed = float(params.get("pedestrian", {}).get("max_speed", 1.5))

	os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
	pygame.init()
	width, height = 800, 600
	screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption("Pedestrian Input Test")
	clock = pygame.time.Clock()

	font = pygame.font.SysFont("arial", 20)
	use_joystick = False
	dragging = False
	last_log = 0.0

	joystick_center = np.array([width - 150, height - 150], dtype=np.float32)
	joystick_radius = 80.0

	running = True
	while running:
		dt = clock.tick(60) / 1000.0
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_j:
					use_joystick = not use_joystick
				elif event.key == pygame.K_ESCAPE:
					running = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if use_joystick:
					mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float32)
					if np.linalg.norm(mouse_pos - joystick_center) <= joystick_radius:
						dragging = True
			elif event.type == pygame.MOUSEBUTTONUP:
				dragging = False

		velocity = np.zeros(2, dtype=np.float32)
		if use_joystick:
			if dragging:
				mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float32)
				offset = mouse_pos - joystick_center
				offset[1] = -offset[1]
				offset = clamp_vector(offset, joystick_radius)
				velocity = (offset / joystick_radius) * max_speed
		else:
			keys = pygame.key.get_pressed()
			if keys[pygame.K_LEFT]:
				velocity[0] -= max_speed
			if keys[pygame.K_RIGHT]:
				velocity[0] += max_speed
			if keys[pygame.K_UP]:
				velocity[1] += max_speed
			if keys[pygame.K_DOWN]:
				velocity[1] -= max_speed
			velocity = clamp_vector(velocity, max_speed)

		now = time.time()
		if now - last_log > 0.2:
			print(f"mode={'joystick' if use_joystick else 'arrows'} vel=({velocity[0]:+.2f}, {velocity[1]:+.2f})")
			last_log = now

		screen.fill((245, 245, 245))

		if use_joystick:
			pygame.draw.circle(screen, (255, 255, 255), joystick_center.astype(int), int(joystick_radius))
			pygame.draw.circle(screen, (0, 0, 0), joystick_center.astype(int), 4)

		info = [
			f"Mode: {'Joystick (press J to toggle)' if use_joystick else 'Arrow keys (press J to toggle)'}",
			f"Velocity: ({velocity[0]:+.2f}, {velocity[1]:+.2f}) m/s",
			f"Max speed: {max_speed:.2f} m/s",
		]
		for idx, line in enumerate(info):
			text = font.render(line, True, (20, 20, 20))
			screen.blit(text, (20, 20 + idx * 24))

		pygame.display.flip()

	pygame.quit()


if __name__ == "__main__":
	main()
