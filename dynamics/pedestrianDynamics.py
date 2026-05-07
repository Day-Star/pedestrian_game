import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_state_bounds(num_pedestrians=2):
    """
    Get physical bounds for the state space.
    
    For robot-centric coordinates:
    - Scout angle: [-pi, pi]
    - Scout speed: [0, 2] m/s (assuming max speed of 2 m/s)
    - Scout angular velocity: [-pi, pi] rad/s (assuming max turn rate of 180 deg/s)
    - Pedestrian relative positions: [-5, 5] m in both x and y (assuming pedestrians can be up to 5m away in any direction)

    Args:
        num_pedestrians (int): Number of pedestrians in the environment

    Returns:
        ndarray: Physical bounds for the state space
    """
    return np.array(
        [np.pi, 3.0, np.pi] + [5.0] * (2 * num_pedestrians), dtype=np.float32
    )

def get_control_bounds():
    """
    Get physical bounds for the control space.
    
    - Scout angular acceleration: [-pi/2, pi/2] rad/s^2 (assuming max angular accel of 90 deg/s^2)
    - Scout acceleration: [-2, 2] m/s^2 (assuming max accel of 2 m/s^2)

    Returns:
        ndarray: Physical bounds for the control space
    """
    return np.array([np.pi/2, 2.0], dtype=np.float32)

def get_vartheta_bounds(num_pedestrians=2):
    """
    Get physical bounds for the vartheta space (pedestrian velocities).
    
    - Pedestrian velocities in world frame: [-2, 2] m/s in both x and y (assuming pedestrians can move up to 2 m/s in any direction)

    Args:
        num_pedestrians (int): Number of pedestrians in the environment
    Returns:
        ndarray: Physical bounds for the vartheta space
    """
    return np.array([2.0] * (2 * num_pedestrians), dtype=np.float32)

def pedestrian_dynamics(x, u, vartheta, num_pedestrians=2, dt=0.1):
    """
    Compute the next state given current state, control, and vartheta.
    
    In robot-centric coordinates: robot at (0,0) facing +x direction.
    Robot motion affects relative pedestrian positions.

    Discrete-time dynamics (standard nonlinear representation, forward Euler):

        Let x_k = [theta_k, v_k, omega_k, p_{1,x,k}, p_{1,y,k}, ..., p_{n,x,k}, p_{n,y,k}]
        and u_k = [alpha_k, a_k], vartheta_k = [p1_vx, p1_vy, ...] (pedestrian world velocities).

        theta_{k+1} = theta_k + dt * omega_k
        v_{k+1} = v_k + dt * a_k
        omega_{k+1} = omega_k + dt * alpha_k

        For each pedestrian i with world velocity (p_{i,vx}, p_{i,vy}):

        p_{i,x,k+1} = p_{i,x,k} + dt * (omega_k * p_{i,y,k} + cos(theta_k)*p_{i,vx} + sin(theta_k)*p_{i,vy} - v_k)
        p_{i,y,k+1} = p_{i,y,k} + dt * (-omega_k * p_{i,x,k} - sin(theta_k)*p_{i,vx} + cos(theta_k)*p_{i,vy})

        In compact form: x_{k+1} = f(x_k, u_k, vartheta_k) where f is the nonlinear map above.

    Args:
        x (ndarray): Current state [Scout angle, Scout speed, scout angular velocity,
                            Pedestrian 1 x position, Pedestrian 1 y position, ...]
        u (ndarray): Control input [Scout angular acceleration, Scout acceleration]
        vartheta (ndarray): Pedestrian velocities in world frame [p1_vx, p1_vy, p2_vx, p2_vy, ...]
        num_pedestrians (int): Number of pedestrians in the environment
        dt (float): Time step for the simulation

    Returns:
        ndarray: Next state after applying dynamics
    """

    def state_derivative(state):
        """Compute dx/dt for the full robot-centric state vector."""
        th, v, om = state[0], state[1], state[2]
        cos_th = np.cos(th)
        sin_th = np.sin(th)
        derivs = [om, u[1], u[0]]  # d(theta)/dt, d(v)/dt, d(omega)/dt
        for ped_idx in range(num_pedestrians):
            pos_idx = 3 + ped_idx * 2
            vel_idx = ped_idx * 2
            p_x = state[pos_idx]
            p_y = state[pos_idx + 1]
            p_vx = vartheta[vel_idx]
            p_vy = vartheta[vel_idx + 1]
            # d(p_rel)/dt = [[0, omega], [-omega, 0]] * p_rel + R^T * (v_ped - v_scout_world)
            derivs.append(om * p_y + cos_th * p_vx + sin_th * p_vy - v)
            derivs.append(-om * p_x - sin_th * p_vx + cos_th * p_vy)
        return np.array(derivs)

    # RK4 integration
    k1 = state_derivative(x)
    k2 = state_derivative(x + 0.5 * dt * k1)
    k3 = state_derivative(x + 0.5 * dt * k2)
    k4 = state_derivative(x + dt * k3)

    # Resolve integration to get next state
    next_state = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Ensure the angle stays within [-pi, pi]
    next_state[0] = (next_state[0] + np.pi) % (2 * np.pi) - np.pi

    return next_state

def global_dynamics(x, u, vartheta, num_pedestrians=2, dt=0.1):
    """
    Used for validating pedestrian_dynamics by comparing results in global frame.

    Args:
        x (ndarray): Current state [Scout Position x, Scout Position y, Scout angle, Scout speed, scout angular velocity,
                            Pedestrian 1 x position, Pedestrian 1 y position, ...]
        u (ndarray): Control input [Scout angular acceleration, Scout acceleration]
        vartheta (ndarray): Pedestrian velocities in world frame [p1_vx, p1_vy, p2_vx, p2_vy, ...]
        num_pedestrians (int): Number of pedestrians in the environment
        dt (float): Time step for the simulation

    Returns:
        ndarray: Next state after applying dynamics
    """

    def state_derivative(state):
        """Compute dx/dt for the full global state vector."""
        th, v, om = state[2], state[3], state[4]
        derivs = [
            v * np.cos(th),  # dx_scout/dt
            v * np.sin(th),  # dy_scout/dt
            om,              # dtheta/dt
            u[1],            # dv/dt
            u[0],            # domega/dt
        ]
        for ped_idx in range(num_pedestrians):
            vel_idx = ped_idx * 2
            derivs.append(vartheta[vel_idx])
            derivs.append(vartheta[vel_idx + 1])
        return np.array(derivs)

    # RK4 integration
    k1 = state_derivative(x)
    k2 = state_derivative(x + 0.5 * dt * k1)
    k3 = state_derivative(x + 0.5 * dt * k2)
    k4 = state_derivative(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Reward function
def reward_fn(x, personal_radii=None):
    """
    Calculate reward based on distance to pedestrians.
    
    Args:
        x (ndarray): Current state [Scout velocity, Scout angular velocity,
                            Pedestrian 1 x position, Pedestrian 1 y position, ...]
        personal_radii (list or ndarray): Personal radius for each pedestrian.
                                          If None, defaults to 0.5 for all pedestrians.
    """
    # Determine number of pedestrians from state size
    num_pedestrians = (len(x) - 3) // 2
    
    # Default personal radii if not provided
    if personal_radii is None:
        personal_radii = [1.0] * num_pedestrians
    elif np.isscalar(personal_radii):
        personal_radii = [float(personal_radii)] * num_pedestrians
    else:
        personal_radii = [float(pr) for pr in personal_radii]
    
    # Calculate distances to all pedestrians
    min_distance = float('inf')
    for ped_idx in range(num_pedestrians):
        pos_idx = 3 + ped_idx * 2
        dist = np.sqrt(x[pos_idx]**2 + x[pos_idx + 1]**2) - personal_radii[ped_idx]
        min_distance = min(min_distance, float(dist))

    # # Adjust positive rewards to keep in line with negative distance rewards
    # if min_distance > 0:
    #     min_distance = min_distance * 0.3
    if min_distance < 0:
        min_distance = min_distance * 0.5

    # Return the minimum distance to any pedestrian
    return min_distance

# Test the dynamics functions
if __name__ == "__main__":
    
    # Example global state: [x_scout, y_scout, theta_scout, v_scout, omega_scout, p1_x, p1_y, p2_x, p2_y]
    x_global = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0])
    u = np.array([0.1, 0.5])  # [angular acceleration, acceleration]
    vartheta = np.array([0.5, 0.0, 0.0, 0.5])  # [p1_vx, p1_vy, p2_vx, p2_vy]
    dt = 0.1

    # Example robot-centric state: [theta_scout, v_scout, omega_scout, p1_x_rel, p1_y_rel, p2_x_rel, p2_y_rel]
    x_robot_centric = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0])

    # Simulation length
    sim_length = 30

    # Store trajectories for plotting
    global_trajectory = [x_global.copy()]
    robot_centric_trajectory = [x_robot_centric.copy()]

    for _ in range(sim_length):
        x_global = global_dynamics(x_global, u, vartheta, num_pedestrians=2, dt=dt)
        x_robot_centric = pedestrian_dynamics(x_robot_centric, u, vartheta, num_pedestrians=2, dt=dt)

        global_trajectory.append(x_global.copy())
        robot_centric_trajectory.append(x_robot_centric.copy())
    
    # Convert to numpy arrays for easier indexing
    global_trajectory = np.array(global_trajectory)
    robot_centric_trajectory = np.array(robot_centric_trajectory)

    # Calculate global position of pedestrian in robot-centric frame for validation
    # p_x_rel = R(theta) * (p_x - x_scout)
    # p_y_rel = R(theta) * (p_y - y_scout)

    # Validated robot trajectory
    val_robot_centric_trajectory = np.zeros_like(global_trajectory)

    for idx in range(len(global_trajectory)):
        x_scout, y_scout, theta_scout = global_trajectory[idx, 0], global_trajectory[idx, 1], global_trajectory[idx, 2]
        cos_theta = np.cos(theta_scout)
        sin_theta = np.sin(theta_scout)

        for ped_idx in range(2):
            p_x = global_trajectory[idx, 5 + ped_idx * 2]
            p_y = global_trajectory[idx, 6 + ped_idx * 2]

            p_x_rel = cos_theta * (p_x - x_scout) + sin_theta * (p_y - y_scout)
            p_y_rel = -sin_theta * (p_x - x_scout) + cos_theta * (p_y - y_scout)

            # Update robot-centric trajectory with calculated relative positions
            val_robot_centric_trajectory[idx, 3 + ped_idx * 2] = p_x_rel
            val_robot_centric_trajectory[idx, 4 + ped_idx * 2] = p_y_rel

    # Plot trajectories
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Global Frame Trajectory")

    # Scout
    color = plt.gca()._get_lines.get_next_color()
    plt.plot(global_trajectory[:, 0], global_trajectory[:, 1], label="Scout", color=color)
    plt.scatter(global_trajectory[0, 0],  global_trajectory[0, 1],  marker='o', color=color, zorder=5, s=60)
    plt.scatter(global_trajectory[-1, 0], global_trajectory[-1, 1], marker='*', color=color, zorder=5, s=120)

    # Pedestrian 1
    color = plt.gca()._get_lines.get_next_color()
    plt.scatter(global_trajectory[:, 5], global_trajectory[:, 6], label="Pedestrian 1", s=10, color=color)
    plt.scatter(global_trajectory[0, 5],  global_trajectory[0, 6],  marker='o', color=color, zorder=5, s=60)
    plt.scatter(global_trajectory[-1, 5], global_trajectory[-1, 6], marker='*', color=color, zorder=5, s=120)

    # Pedestrian 2
    color = plt.gca()._get_lines.get_next_color()
    plt.scatter(global_trajectory[:, 7], global_trajectory[:, 8], label="Pedestrian 2", s=10, color=color)
    plt.scatter(global_trajectory[0, 7],  global_trajectory[0, 8],  marker='o', color=color, zorder=5, s=60)
    plt.scatter(global_trajectory[-1, 7], global_trajectory[-1, 8], marker='*', color=color, zorder=5, s=120)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.title("Robot-Centric Frame Trajectory")

    def plot_with_endpoints(x, y, label):
        color = plt.gca()._get_lines.get_next_color()
        plt.plot(x, y, label=label, color=color)
        plt.scatter(x[0],  y[0],  marker='o', color=color, zorder=5, s=60)
        plt.scatter(x[-1], y[-1], marker='*', color=color, zorder=5, s=120)

    plot_with_endpoints(robot_centric_trajectory[:, 3], robot_centric_trajectory[:, 4], "Pedestrian 1")
    plot_with_endpoints(robot_centric_trajectory[:, 5], robot_centric_trajectory[:, 6], "Pedestrian 2")
    plot_with_endpoints(val_robot_centric_trajectory[:, 3], val_robot_centric_trajectory[:, 4], "Pedestrian Validation 1")
    plot_with_endpoints(val_robot_centric_trajectory[:, 5], val_robot_centric_trajectory[:, 6], "Pedestrian Validation 2")

    # Draw robot at origin
    ax = plt.gca()
    robot_circle = mpatches.Circle((0, 0), radius=0.2, color='white', ec='black', linewidth=2, zorder=6, label="Robot")
    ax.add_patch(robot_circle)

    plt.xlabel("X Position (Robot-Centric)")
    plt.ylabel("Y Position (Robot-Centric)")
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig("pedestrian_dynamics_validation.png")

    # Calculate reward across single pedestrian locations in x,y relative to robot
    plt.figure(figsize=(6, 5))
    x_rel = np.linspace(-5, 5, 100)
    y_rel = np.linspace(-5, 5, 100)
    X_rel, Y_rel = np.meshgrid(x_rel, y_rel)
    rewards = np.zeros_like(X_rel)
    for i in range(X_rel.shape[0]):
        for j in range(X_rel.shape[1]):
            state = np.array([0.0, 1.0, 0.0, X_rel[i, j], Y_rel[i, j]])  # [theta, v, omega, p_x_rel, p_y_rel]
            rewards[i, j] = reward_fn(state, personal_radii=1.0)

    plt.contourf(X_rel, Y_rel, rewards, levels=50, cmap='viridis')
    plt.colorbar(label='Reward')
    plt.xlabel("Pedestrian X Position (Relative to Robot)")
    plt.ylabel("Pedestrian Y Position (Relative to Robot)")
    plt.title("Reward Landscape for Single Pedestrian")
    plt.scatter(0, 0, marker='o', color='red', label="Robot", zorder=5, s=60)
    plt.legend()
    plt.savefig("reward_landscape.png")
    plt.show()