import matplotlib.pyplot as plt
import numpy as np

# Initial State
num_robots = 20
arena_size = (100, 100)
steps = 100
num_obstacles = 5
obstacle_radius = 5
happiness_threshold = 10  # Distance at which robots feel 'happy' due to other robots

# Initialize
np.random.seed(0)
positions = np.random.rand(num_robots, 2) * arena_size

# Initialize obstacles
obstacles = np.random.rand(num_obstacles, 2) * arena_size

# Updating robot positions
def update_positions(positions, arena_size):
    # Random movement
    movement = np.random.rand(num_robots, 2) - 0.5
    new_positions = positions + movement

    # Boundary conditions
    new_positions = np.clip(new_positions, 0, arena_size[0])

    return new_positions

# Calculating happiness
def calculate_happiness(positions, obstacles, obstacle_radius, happiness_threshold):
    happiness = np.zeros(num_robots)
    for i in range(num_robots):
        # Distance to other robots
        distances = np.linalg.norm(positions - positions[i], axis=1)
        neighbor_happiness = np.sum(distances < happiness_threshold) - 1  # Subtract self

        # Distance to nearest obstacle
        min_dist_to_obstacle = np.min(np.linalg.norm(obstacles - positions[i], axis=1))
        obstacle_happiness = max(0, min_dist_to_obstacle - obstacle_radius)

        # Combine happiness measures
        happiness[i] = neighbor_happiness + obstacle_happiness
    return happiness

# Simulate
for _ in range(steps):
    positions = update_positions(positions, arena_size)
    happiness = calculate_happiness(positions, obstacles, obstacle_radius, happiness_threshold)

plt.figure(figsize=(8, 8))
plt.scatter(positions[:, 0], positions[:, 1], c=happiness, cmap='viridis')
for obstacle in obstacles:
    circle = plt.Circle(obstacle, obstacle_radius, color='red', alpha=0.5)
    plt.gca().add_patch(circle)
plt.colorbar(label='Happiness Level')
plt.xlim(0, arena_size[0])
plt.ylim(0, arena_size[1])
plt.title('Robot Swarm Simulation with Obstacles and Neighbors')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.grid(True)
plt.show()
