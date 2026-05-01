import pygame
import numpy as np
import math
import sys

from main import TacticalPatrolPlanner

def run_simulation():
    pygame.init()
    
    # Grid settings
    GRID_W, GRID_H = 20, 20
    CELL_SIZE_PX = 30
    screen = pygame.display.set_mode((GRID_W * CELL_SIZE_PX, GRID_H * CELL_SIZE_PX))
    pygame.display.set_caption("SS-NBV Tactical Patrol Simulator")
    clock = pygame.time.Clock()

    # 1. Initialize the Planner
    planner = TacticalPatrolPlanner(grid_width=GRID_W, grid_height=GRID_H, cell_size_cm=20)
    
    # 2. Build a Mock Room (Fake Onion Peeling result)
    # 0 = Free Space, 1 = Wall
    planner.occupancy_grid[5:15, 10] = 1 # A vertical wall in the middle
    planner.occupancy_grid[5, 5:10] = 1  # A horizontal wall
    
    # 3. Add a Mock YOLO Semantic Target (A high-risk door)
    planner.mark_semantic_target(18, 18, "door")

    # Robot State
    robot_x, robot_y = 2, 2

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- ALGORITHM LOGIC ---
        
        # 1. Age the room (Staleness increases)
        planner.update_staleness()
        
        # 2. Generate Candidate Nodes (Every 2nd free block for speed)
        candidates = []
        for x in range(0, GRID_W, 2):
            for y in range(0, GRID_H, 2):
                if planner.occupancy_grid[x][y] == 0:
                    candidates.append((x, y))
                    
        # 3. Run Utility Equation to find Next Best View
        next_node, best_yaw = planner.get_next_best_view(robot_x, robot_y, candidates)
        
        if next_node:
            robot_x, robot_y = next_node
            
            # 4. Sweep the camera (Reset staleness of what the robot can see now)
            # In a real bot, you use Bresenham here to reset only visible blocks.
            # For this visualizer, we'll do a simple radius reset.
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    cx, cy = robot_x + dx, robot_y + dy
                    if 0 <= cx < GRID_W and 0 <= cy < GRID_H:
                        if planner.occupancy_grid[cx][cy] == 0:
                            planner.staleness_grid[cx][cy] = 0.0

        # --- DRAWING THE SCREEN ---
        screen.fill((255, 255, 255))
        
        for x in range(GRID_W):
            for y in range(GRID_H):
                rect = pygame.Rect(x * CELL_SIZE_PX, y * CELL_SIZE_PX, CELL_SIZE_PX, CELL_SIZE_PX)
                
                # Draw Walls
                if planner.occupancy_grid[x][y] == 1:
                    pygame.draw.rect(screen, (50, 50, 50), rect)
                else:
                    # Draw Staleness (Heatmap: White -> Red as it ages)
                    stale_val = min(255, int(planner.staleness_grid[x][y] * 5))
                    color = (255, 255 - stale_val, 255 - stale_val)
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, (200, 200, 200), rect, 1) # Grid lines

        # Draw Robot
        bot_center = (int((robot_x + 0.5) * CELL_SIZE_PX), int((robot_y + 0.5) * CELL_SIZE_PX))
        pygame.draw.circle(screen, (0, 0, 255), bot_center, int(CELL_SIZE_PX * 0.4))
        
        # Draw Yaw Angle (Camera Direction)
        yaw_rad = math.radians(best_yaw)
        end_x = bot_center[0] + int(math.cos(yaw_rad) * CELL_SIZE_PX)
        end_y = bot_center[1] + int(math.sin(yaw_rad) * CELL_SIZE_PX)
        pygame.draw.line(screen, (0, 255, 0), bot_center, (end_x, end_y), 3)

        pygame.display.flip()
        
        # Delay so you can actually watch it think and move
        pygame.time.delay(500) 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()