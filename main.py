import numpy as np
import math
import heapq

class TacticalPatrolPlanner:
    def __init__(self, grid_width, grid_height, cell_size_cm=20):
        # 1. The Dual Grids
        self.occupancy_grid = np.zeros((grid_width, grid_height), dtype=int)
        self.staleness_grid = np.zeros((grid_width, grid_height), dtype=float)
        self.semantic_multipliers = np.ones((grid_width, grid_height), dtype=float)
        
        # Hyperparameters for S(n) equation
        self.alpha = 1.0  # Weight for pure visibility
        self.beta = 2.5   # Weight for staleness (paranoia)
        self.gamma = 0.5  # Weight for travel cost (laziness)
        self.camera_range_cells = int(300 / cell_size_cm) # Assume 3 meter camera range
        
        # 2. Pre-compute the Ray Table (Optimization)
        self.ray_table = self._precompute_ray_table()

    def _precompute_ray_table(self):
        """Generates Bresenham steps for 360 degrees out to camera range."""
        table = {}
        for angle in range(360):
            rad = math.radians(angle)
            end_x = int(self.camera_range_cells * math.cos(rad))
            end_y = int(self.camera_range_cells * math.sin(rad))
            table[angle] = self._bresenham(0, 0, end_x, end_y)
        return table

    def _bresenham(self, x0, y0, x1, y1):
        """Standard integer-only line algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def update_staleness(self):
        """Called every second to age the map based on YOLO semantics."""
        free_space_mask = (self.occupancy_grid == 0)
        self.staleness_grid[free_space_mask] += (1.0 * self.semantic_multipliers[free_space_mask])

    def mark_semantic_target(self, x, y, label):
        """Called if YOLO detects a door/window during mapping."""
        if label in ["door", "window"]:
            self.semantic_multipliers[x][y] = 5.0 # Ages 5x faster

    def evaluate_candidate_node(self, cx, cy):
        """Shoots rays from a node, runs 90-deg sliding window, returns max score & angle."""
        ray_scores = np.zeros(360)
        
        # 1. Shoot Rays
        for angle in range(360):
            score = 0
            for offset_x, offset_y in self.ray_table[angle]:
                check_x, check_y = cx + offset_x, cy + offset_y
                
                # Boundary and collision check
                if (0 <= check_x < self.occupancy_grid.shape[0] and 
                    0 <= check_y < self.occupancy_grid.shape[1]):
                    if self.occupancy_grid[check_x][check_y] == 1:
                        break # Wall hit, stop ray
                    score += self.staleness_grid[check_x][check_y]
            ray_scores[angle] = score
            
        # 2. Sliding Window (90 degrees FOV)
        max_fov_score = 0
        best_yaw = 0
        
        for i in range(360):
            # Wrap around sum for 90 degrees
            indices = [(i + j) % 360 for j in range(90)]
            window_score = np.sum(ray_scores[indices])
            
            if window_score > max_fov_score:
                max_fov_score = window_score
                best_yaw = (i + 45) % 360 # Center of the 90deg window
                
        return max_fov_score, best_yaw

    def get_next_best_view(self, current_x, current_y, candidate_nodes):
        """The S(n) Utility Equation to pick the next destination."""
        best_node = None
        best_score = -float('inf')
        best_yaw = 0
        
        for (nx, ny) in candidate_nodes:
            # Calculate Information/Staleness Gain
            v_stale, optimal_yaw = self.evaluate_candidate_node(nx, ny)
            
            # Mock A* distance (Euclidean for simplicity in this example)
            travel_cost = math.dist((current_x, current_y), (nx, ny))
            
            # The S(n) Equation
            utility = (self.alpha * v_stale) - (self.gamma * travel_cost)
            
            if utility > best_score:
                best_score = utility
                best_node = (nx, ny)
                best_yaw = optimal_yaw
                
        return best_node, best_yaw
    
    def a_star(self, start, goal):
        """Calculates the shortest grid path from start to goal."""
        # Standard 8-way movement (Horizontal, Vertical, Diagonal)
        neighbors = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: math.dist(start, goal)} # Euclidean distance heuristic
        
        # Priority queue: stores tuples of (fscore, (x, y))
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            
            if current == goal:
                # Target reached, reconstruct path backwards
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse() # Flip it so it goes Start -> Goal
                return path

            close_set.add(current)
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Boundary check
                if not (0 <= neighbor[0] < self.occupancy_grid.shape[0] and 
                        0 <= neighbor[1] < self.occupancy_grid.shape[1]):
                    continue
                    
                # Wall check
                if self.occupancy_grid[neighbor[0]][neighbor[1]] == 1:
                    continue
                    
                # Diagonal movement cost is slightly higher (sqrt(2))
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = gscore[current] + cost
                
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                    
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + math.dist(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                    
        return [] # Return empty list if no path is possible