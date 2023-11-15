# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:11:44 2023

@author: rb
"""

import matplotlib.pyplot as plt
import numpy as np

# Settings
num_robots = 20
arena_size = (100, 100)
steps = 100
happiness_threshold = 10  # Distance at which robots feel 'happy'

# Initialize robots' positions
np.random.seed(0)
positions = np.random.rand(num_robots, 2) * arena_size

# Function to update positions
def update_positions(positions, arena_size):
    # Random movement
    movement = np.random.rand(num_robots, 2) - 0.5
    new_positions = positions + movement

    # Boundary conditions
    new_positions = np.clip(new_positions, 0, arena_size[0])

    return new_positions

# Function to calculate happiness
def calculate_happiness(positions, threshold):
    happiness = np.zeros(num_robots)
    for i in range(num_robots):
        # Calculate distance to all other robots
        distances = np.linalg.norm(positions - positions[i], axis=1)
        # Robots are happy if they have neighbors within the threshold distance
        happiness[i] = np.sum(distances < threshold) - 1  # Subtract self
    return happiness

# Simulate
for _ in range(steps):
    positions = update_positions(positions, arena_size)
    happiness = calculate_happiness(positions, happiness_threshold)

# Plot final positions
plt.figure(figsize=(8, 8))
plt.scatter(positions[:, 0], positions[:, 1], c=happiness, cmap='viridis')
plt.colorbar(label='Happiness Level')
plt.xlim(0, arena_size[0])
plt.ylim(0, arena_size[1])
plt.title('Robot Swarm Simulation')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.grid(True)
plt.show()
