import pygame
import sys
import math

from main import TacticalPatrolPlanner 

def run_simulation():
    pygame.init()
    
    # Increased grid size to allow for complex rooms
    GRID_W, GRID_H = 30, 30
    CELL_SIZE_PX = 25 # Slightly smaller pixels to fit the screen
    screen = pygame.display.set_mode((GRID_W * CELL_SIZE_PX, GRID_H * CELL_SIZE_PX))
    pygame.display.set_caption("SS-NBV Tactical Patrol Simulator - Complex Environment")

    planner = TacticalPatrolPlanner(grid_width=GRID_W, grid_height=GRID_H, cell_size_cm=20)
    
    # ==========================================
    # BUILDING THE COMPLEX ENVIRONMENT
    # ==========================================
    
    # 1. Top-Left Room (with a narrow door at the bottom)
    planner.occupancy_grid[5:15, 8] = 1      # Right wall of the room
    planner.occupancy_grid[15, 0:5] = 1      # Bottom wall (leaves gap at x=5,6,7 for door)
    
    # 2. Bottom-Right Server Room (L-shaped trap)
    planner.occupancy_grid[20:28, 20] = 1    # Left wall
    planner.occupancy_grid[20, 20:30] = 1    # Top wall (door is at the bottom right)
    
    # 3. Central Obstacle (Desks / Pillars blocking Line of Sight)
    planner.occupancy_grid[12:16, 14:18] = 1 
    
    # 4. A random dead-end wall segment to test the raycasting
    planner.occupancy_grid[2:8, 22] = 1
    
    # ==========================================
    # SEMANTIC TARGETS (YOLO Mock Data)
    # ==========================================
    # Mark the doorway of the Top-Left room as a high-risk portal
    planner.mark_semantic_target(6, 15, "door")
    planner.mark_semantic_target(7, 15, "door")

    # Robot Starting State (Bottom Left Corner)
    robot_x, robot_y = 2, 28
    current_path = [] # Tracks the robot's physical movement sequence

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- ALGORITHM & MOVEMENT LOGIC ---
        
        # 1. The Room is always aging
        planner.update_staleness()
        
        # 2. STATE MACHINE: Are we driving or thinking?
        if len(current_path) == 0:
            # THINKING STATE: We reached our destination. Pick a new one.
            candidates = []
            for x in range(0, GRID_W, 2):
                for y in range(0, GRID_H, 2):
                    if planner.occupancy_grid[x][y] == 0:
                        candidates.append((x, y))
                        
            next_node, best_yaw = planner.get_next_best_view(robot_x, robot_y, candidates)
            
            if next_node:
                current_path = planner.a_star((robot_x, robot_y), next_node)
                # Once it decides where to go, do one final sweep at the optimal target angle
                planner.sweep_camera(robot_x, robot_y, best_yaw, fov=90)
        
        else:
            # DRIVING STATE: Take one physical step
            next_step = current_path.pop(0)
            
            # Calculate physical yaw based on movement direction using atan2
            dx = next_step[0] - robot_x
            dy = next_step[1] - robot_y
            current_yaw = math.degrees(math.atan2(dy, dx)) % 360
            
            robot_x, robot_y = next_step
            
            # TRUE CAMERA SWEEP: Raycast only what the camera physically sees
            planner.sweep_camera(robot_x, robot_y, current_yaw, fov=90)
        
        # --- DRAWING THE SCREEN ---
        screen.fill((255, 255, 255))
        
        for x in range(GRID_W):
            for y in range(GRID_H):
                rect = pygame.Rect(x * CELL_SIZE_PX, y * CELL_SIZE_PX, CELL_SIZE_PX, CELL_SIZE_PX)
                if planner.occupancy_grid[x][y] == 1:
                    pygame.draw.rect(screen, (30, 30, 30), rect)
                else:
                    stale_val = min(255, int(planner.staleness_grid[x][y] * 3))
                    color = (255, 255 - stale_val, 255 - stale_val)
                    pygame.draw.rect(screen, color, rect)

        # Draw Semantic Door (Orange)
        pygame.draw.rect(screen, (255, 165, 0), pygame.Rect(6 * CELL_SIZE_PX, 15 * CELL_SIZE_PX, CELL_SIZE_PX, CELL_SIZE_PX))
        pygame.draw.rect(screen, (255, 165, 0), pygame.Rect(7 * CELL_SIZE_PX, 15 * CELL_SIZE_PX, CELL_SIZE_PX, CELL_SIZE_PX))

        # VISUALIZATION UPGRADE: Draw the intended path line
        if len(current_path) > 0:
            path_pixels = [(int((px + 0.5) * CELL_SIZE_PX), int((py + 0.5) * CELL_SIZE_PX)) for px, py in current_path]
            # Add the current robot pos to the start of the line so it connects
            path_pixels.insert(0, (int((robot_x + 0.5) * CELL_SIZE_PX), int((robot_y + 0.5) * CELL_SIZE_PX)))
            if len(path_pixels) > 1:
                pygame.draw.lines(screen, (255, 0, 255), False, path_pixels, 2) # Magenta line

        # Draw Robot (Blue Circle)
        bot_center = (int((robot_x + 0.5) * CELL_SIZE_PX), int((robot_y + 0.5) * CELL_SIZE_PX))
        pygame.draw.circle(screen, (0, 100, 255), bot_center, int(CELL_SIZE_PX * 0.4))

        pygame.display.flip()
        
        # Reduced delay so the robot drives smoothly instead of jumping
        pygame.time.delay(50) 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()