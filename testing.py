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

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- ALGORITHM LOGIC ---
        planner.update_staleness()
        
        # Generate Candidate Nodes (Only checking free space)
        candidates = []
        for x in range(0, GRID_W, 2):
            for y in range(0, GRID_H, 2):
                if planner.occupancy_grid[x][y] == 0:
                    candidates.append((x, y))
                    
        # Find Next Best View
        next_node, best_yaw = planner.get_next_best_view(robot_x, robot_y, candidates)
        
        if next_node:
            # TELEPORTING (We will replace this with A* in Part 2)
            robot_x, robot_y = next_node
            
            # Simulated Camera Sweep (Clearing staleness based on rough Line of Sight)
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    cx, cy = robot_x + dx, robot_y + dy
                    if 0 <= cx < GRID_W and 0 <= cy < GRID_H:
                        if planner.occupancy_grid[cx][cy] == 0:
                            planner.staleness_grid[cx][cy] = 0.0

        # --- DRAWING THE SCREEN ---
        screen.fill((255, 255, 255))
        
        for x in range(GRID_W):
            for y in range(GRID_H):
                rect = pygame.Rect(x * CELL_SIZE_PX, y * CELL_SIZE_PX, CELL_SIZE_PX, CELL_SIZE_PX)
                
                # Draw Walls (Solid Black)
                if planner.occupancy_grid[x][y] == 1:
                    pygame.draw.rect(screen, (30, 30, 30), rect)
                else:
                    # Draw Staleness Heatmap (White -> Red)
                    stale_val = min(255, int(planner.staleness_grid[x][y] * 3)) # Multiplier controls visual fade speed
                    color = (255, 255 - stale_val, 255 - stale_val)
                    pygame.draw.rect(screen, color, rect)
                    # Optional: uncomment to draw grid lines
                    # pygame.draw.rect(screen, (220, 220, 220), rect, 1) 

        # Draw the High-Risk Doorway in Orange (for visualization)
        pygame.draw.rect(screen, (255, 165, 0), pygame.Rect(6 * CELL_SIZE_PX, 15 * CELL_SIZE_PX, CELL_SIZE_PX, CELL_SIZE_PX))
        pygame.draw.rect(screen, (255, 165, 0), pygame.Rect(7 * CELL_SIZE_PX, 15 * CELL_SIZE_PX, CELL_SIZE_PX, CELL_SIZE_PX))

        # Draw Robot (Blue Circle)
        bot_center = (int((robot_x + 0.5) * CELL_SIZE_PX), int((robot_y + 0.5) * CELL_SIZE_PX))
        pygame.draw.circle(screen, (0, 100, 255), bot_center, int(CELL_SIZE_PX * 0.4))
        
        # Draw Camera Direction (Green Line)
        yaw_rad = math.radians(best_yaw)
        end_x = bot_center[0] + int(math.cos(yaw_rad) * CELL_SIZE_PX * 1.5)
        end_y = bot_center[1] + int(math.sin(yaw_rad) * CELL_SIZE_PX * 1.5)
        pygame.draw.line(screen, (0, 255, 0), bot_center, (end_x, end_y), 4)

        pygame.display.flip()
        pygame.time.delay(400) # Slowed down slightly so you can watch the logic

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()