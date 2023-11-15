import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Initial States
num_robots = 20
arena_size = (100, 100)
num_obstacles = 5
obstacle_radius = 5
happiness_threshold = 10  # Distance at which robots feel 'happy'
simulation_steps = 100  # Total number of steps in the simulation

# Initializing Positions of Robots
np.random.seed(np.random.randint(0, 1000))
positions = np.random.rand(num_robots, 2) * arena_size
print (positions)

# Initializing obstacles
obstacles = np.random.rand(num_obstacles, 2) * arena_size

# Making robots move
def update_positions(positions, arena_size, obstacles, obstacle_radius):
    movement = np.random.rand(num_robots, 2)*3 - 1.5

    for i in range(num_robots):
        for obstacle in obstacles:
            # Calculate distance to the obstacle
            distance = np.linalg.norm(positions[i] - obstacle)
            
            # If too close to the obstacle, adjust movement away from it
            if distance < obstacle_radius * 2:  # boundary of the circle
                avoidance_direction = positions[i] - obstacle
                avoidance_direction /= np.linalg.norm(avoidance_direction)
                movement[i] += avoidance_direction * 0.5  # strength of avoidance
    
    new_positions = positions + movement
    # So the robots don't go outside the map
    new_positions = np.clip(new_positions, 0, arena_size[0])
    return new_positions

# calculating robot happiness
def calculate_happiness(positions, obstacles, obstacle_radius, happiness_threshold):
    happiness = np.zeros(num_robots)
    for i in range(num_robots):
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        neighbor_happiness = np.sum(distances < happiness_threshold) - 1
        obstacle_distances = np.sqrt(np.sum((obstacles - positions[i])**2, axis=1))
        min_dist_to_obstacle = np.min(obstacle_distances)
        obstacle_happiness = max(0, min_dist_to_obstacle - obstacle_radius)
        happiness[i] = neighbor_happiness + obstacle_happiness
    return happiness

# animation stuffs
fig, ax = plt.subplots(figsize=(16, 12))
scat = ax.scatter([], [], c=[], cmap='viridis', vmin=0, vmax=20)
colorbar = fig.colorbar(scat, ax=ax)
colorbar.set_label('Happiness Level')
ax.set_xlim(0, arena_size[0])
ax.set_ylim(0, arena_size[1])
for obstacle in obstacles:
    circle = plt.Circle(obstacle, obstacle_radius, color='red', alpha=0.5)
    ax.add_patch(circle)

def init():
    global positions, happiness
    positions = np.random.rand(num_robots, 2) * arena_size
    happiness = calculate_happiness(positions, obstacles, obstacle_radius, happiness_threshold)
    scat.set_offsets(positions)
    scat.set_array(happiness)
    return scat,

def update(frame):
    global positions, happiness
    positions = update_positions(positions, arena_size, obstacles, obstacle_radius)
    happiness = calculate_happiness(positions, obstacles, obstacle_radius, happiness_threshold)
    scat.set_offsets(positions)
    scat.set_array(happiness)
    return scat,

ani = FuncAnimation(fig, update, frames=simulation_steps, init_func=init, blit=True, interval=10)

# Saving animation as a GIF
ani.save('robot_swarm_simulation.gif', writer='pillow', fps=10)

print('animation saved')
