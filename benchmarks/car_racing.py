import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
import numpy as np
from typing import Tuple, Dict, Any, Optional

# The user's original code had a 'constraints' import.
# Since the file is not provided, I will comment it out but leave a note.
from constraints import safety 

class CarRacingEnv(gym.Env):
    """
    A 2D environment where a car must navigate to a corner and then
    return to the origin while avoiding an unsafe rectangular region.

    State: [x, y, vx, vy]
        - x, y: Position of the car.
        - vx, vy: Velocity of the car.
    
    Action: [ax, ay]
        - ax, ay: Acceleration in x and y directions.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(4,), dtype=np.float32)

        self.init_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(4,), dtype=np.float32)

        self._max_episode_steps = 400

        self.polys = [
            # P (s 1)^T <= 0 iff 1.0 <= s[0] <= 2.0 and 1.0 <= s[1] <= 2.0
            np.array([[1.0, 0.0, 0.0, 0.0, -2.0],
                      [-1.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0, 0.0, -2.0],
                      [0.0, -1.0, 0.0, 0.0, 1.0]])
        ]

        self.safe_polys = [
            np.array([[1.0, 0.0, 0.0, 0.0, -0.99]]),
            np.array([[-1.0, 0.0, 0.0, 0.0, 2.01]]),
            np.array([[0.0, 1.0, 0.0, 0.0, -0.99]]),
            np.array([[0.0, -1.0, 0.0, 0.0, 2.01]])
        ]
        
        self.safety = safety.Box(lower_bounds=[
                [-5, -5, -5, -5],
                [-5, -5, -5, -5],
                [2.01, -5, -5, -5],
                [-5, 2.01, -5, -5],
            ], 
            upper_bounds=[
                [0.99, 5, 5, 5],
                [5, 0.99, 5, 5],
                [5, 5, 5, 5],
                [5, 5, 5, 5],
        ])
        
        self.state_processor = None
        
        # Rendering attributes
        self.render_mode = "rgb_array" if render_mode is None else render_mode
        self.screen_width = 800
        self.screen_height = 800
        self.world_width = 10 # Corresponds to observation space from -5 to 5
        self.scale = self.screen_width / self.world_width
        self.screen = None
        self.clock = None
        self.isopen = True


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.state = self.init_space.sample().astype(np.float32)
        self.steps = 0
        self.corner = False
        
        if self.render_mode == "human":
            self.render()
            
        return self.state, {'state_original': self.state}

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        dt = 0.02
        x = self.state[0] + dt * self.state[2]
        y = self.state[1] + dt * self.state[3]
        vx = self.state[2] + dt * action[0]
        vy = self.state[3] + dt * action[1]

        self.state = np.clip(np.array([x, y, vx, vy]),
                             self.observation_space.low,
                             self.observation_space.high).astype(np.float32)

        if not self.corner and x >= 3.0 and y >= 3.0:
            self.corner = True

        if self.corner:
            reward = -(abs(x) + abs(y))
        else:
            reward = -(6.0 + abs(x - 3.0) + abs(y - 3.0))

        self.steps += 1
        
        # Gymnasium API uses terminated (end of episode) and truncated (time limit)
        terminated = (self.corner and x <= 0.0 and y <= 0.0) or self.unsafe(self.state)
        truncated = self.steps >= self._max_episode_steps
        
        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {'state_original': self.state}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Car Racing Environment")
            else:  # "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # --- Helper for coordinate conversion ---
        def world_to_screen(world_pos):
            # Flips the y-axis for standard screen coordinates
            screen_x = (world_pos[0] + self.world_width / 2) * self.scale
            screen_y = (-world_pos[1] + self.world_width / 2) * self.scale
            return int(screen_x), int(screen_y)

        # --- Drawing ---
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((240, 240, 240)) # Light gray background

        # Draw Goal Zone (target circle at corner 3,3)
        goal_pos = world_to_screen((3,3))
        pygame.draw.circle(canvas, (200, 255, 200), goal_pos, 20, 0)
        
        # Draw Unsafe Zone (the rectangular obstacle)
        # World coordinates: x between 1 and 2, y between 1 and 2.
        # The top-left corner in world space is (1, 2) due to y-axis direction.
        unsafe_world_tl = (1, 2) 
        unsafe_screen_tl = world_to_screen(unsafe_world_tl)
        unsafe_screen_size = (1 * self.scale, 1 * self.scale)
        unsafe_screen_rect = pygame.Rect(unsafe_screen_tl, unsafe_screen_size)
        pygame.draw.rect(canvas, (255, 180, 180), unsafe_screen_rect)
        pygame.draw.rect(canvas, (180, 50, 50), unsafe_screen_rect, 2) # Add a border


        # Draw Origin (final destination)
        origin_pos = world_to_screen((0,0))
        pygame.draw.circle(canvas, (200, 200, 255), origin_pos, 15, 0)
        pygame.draw.circle(canvas, (50, 50, 180), origin_pos, 15, 2) # Add a border


        # Draw the agent (car)
        car_pos_screen = world_to_screen((self.state[0], self.state[1]))
        pygame.draw.circle(canvas, (50, 100, 220), car_pos_screen, 12, 0)

        # --- Text Information ---
        font = pygame.font.Font(None, 28)
        info_color = (10, 10, 10)
        
        pos_text = font.render(f"Pos: ({self.state[0]:.2f}, {self.state[1]:.2f})", True, info_color)
        vel_text = font.render(f"Vel: ({self.state[2]:.2f}, {self.state[3]:.2f})", True, info_color)
        task_text_str = "Task: Return to Origin" if self.corner else "Task: Go to Goal"
        task_text = font.render(task_text_str, True, info_color)
        
        canvas.blit(pos_text, (10, 10))
        canvas.blit(vel_text, (10, 35))
        canvas.blit(task_text, (10, 60))

        
        # --- Safety Status Text ---
        is_unsafe = self.unsafe(self.state)
        status_text_str = "Status: Unsafe" if is_unsafe else "Status: Safe"
        status_color = (220, 50, 50) if is_unsafe else (50, 200, 50)

        status_font = pygame.font.Font(None, 36)
        status_text = status_font.render(status_text_str, True, status_color)
        status_rect = status_text.get_rect(topright=(self.screen_width - 10, 10))
        canvas.blit(status_text, status_rect)

        # --- Finalize Frame ---
        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if self.render_mode in ["human", "rgb_array"]:
             return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def predict_done(self, state: np.ndarray) -> bool:
        return False

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self, state: np.ndarray, simulated=False) -> bool:
        x, y = state[0], state[1]
        return 1.0 <= x <= 2.0 and 1.0 <= y <= 2.0

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

