import pygame
import numpy as np
import scenario_render

# Define constants for the game window
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
CELL_SIZE = 30

# Define colors
COLOR_MAP = {
    -1: (0, 0, 0),   # Black for inactive cells
    0: (255, 255, 255), # White for blank cells
    1: (255, 0, 0),  # Red for the learned agent
    2: (0, 0, 255),  # Blue for other agents
    3: (0, 0, 255),  # Blue for another agent
    10: (255, 215, 0), # Gold for the apple
    11: (255, 0 ,0),
    12: (0, 0, 255),  
    13: (0, 0, 255)
}

def render_environment(observations):
    pygame.init()
    
    # Set the window size based on grid dimensions
    cell_size = 50  # Each cell will be 50x50 pixels
    grid_height = 10
    grid_width = 16
    screen = pygame.display.set_mode((grid_width * cell_size, grid_height * cell_size))

    # Color mapping for different cell values
    # COLOR_MAP = {
    #     -1: (128, 128, 128), # Inactive cells (gray)
    #      0: (255, 255, 255), # Blank cells (white)
    #      1: (0, 255, 0),     # Learned agent (green)
    #      2: (255, 0, 0),     # Random agent (red)
    #      3: (255, 255, 0),   # Another agent (yellow)
    #     10: (0, 0, 255),     # Apple (blue)
    # }

    # Main render loop for each step
    for step_idx, obs in enumerate(observations):
        screen.fill((0, 0, 0))  # Clear the screen before rendering each step
        print(f"Rendering step {step_idx + 1}:")
        
        for i, row in enumerate(obs):  # Iterate through rows of the grid
            for j, cell_value in enumerate(row):  # Iterate through each cell in the row
                # Debug print to track cell values
                # print(f"Cell[{i}][{j}] = {cell_value}")

                # try:
                color = COLOR_MAP.get(int(cell_value), (255, 255, 255))  # Get color for cell
                # except ValueError:
                    # print(f"Invalid cell value at ({i}, {j}): {cell_value}")
                    # continue

                pygame.draw.rect(screen, color, pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))

        pygame.display.flip()  # Update the display

        # Delay to see the environment change with each step (adjust as needed)
        pygame.time.delay(500)  # 1000 ms delay per step

        # Allow user to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
    
    # Quit pygame after rendering all steps
    pygame.quit()

# render_environment(observations)

observation = scenario_render.getObservation()
print(observation)
# Sample observations string (replace this with your actual observations)

# Call the function to render the environment
render_environment(observation)
