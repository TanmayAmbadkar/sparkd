import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
import numpy as np
from typing import Tuple, Dict, Any, Optional
# The user's original code had a 'constraints' import.
# Since the file is not provided, I will comment it out but leave a note.
from constraints import safety 

class AccEnv(gym.Env):
    """
    A 1D environment for an Adaptive Cruise Control (ACC) simulation.

    The goal of the agent is to control its acceleration to follow a lead
    vehicle without crashing.

    State:
        - position (x): The ego vehicle's position relative to the lead vehicle.
                        A negative value means the ego vehicle is behind.
                        x >= 0 means a crash has occurred.
        - velocity (v): The ego vehicle's relative velocity.

    Action:
        - A single value representing the acceleration command.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        super(AccEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-10.0, -10.0]),
                                                high=np.array([10.0, 10.0]),
                                                dtype=np.float32)

        self.init_space = gym.spaces.Box(low=np.array([-1.1, -0.1]),
                                         high=np.array([-0.9, 0.1]),
                                         dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)

        self.rng = np.random.default_rng()

        self.concrete_safety = [
            lambda x: x[0]
        ]

        self._max_episode_steps = 300

        self.polys = [
            # P (s 1)^T <= 0 iff s[0] >= 0
            # P = [[-1, 0, 0]]
            np.array([[-1.0, 0.0, 0.0]])
        ]

        self.safe_polys = [
            np.array([[1.0, 0.0, 0.01]])
        ]
        
        # This part of the user's code depends on the 'constraints' module.
        # It's kept here for context.
        self.safety = safety.Box(
                np.array([-10, -10]),
                np.array([0.01, 10])
        )
        
        # Rendering attributes
        self.render_mode = "rgb_array" if render_mode is None else render_mode
        self.screen_width = 800
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.state = self.init_space.sample().astype(np.float32)
        self.steps = 0
        if seed is not None:
            self.rng = np.random.default_rng(np.random.PCG64(seed))
        
        if self.render_mode == "human":
            self.render()
            
        return self.state, {}

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        dt = 0.02
        
        # Dynamics update
        x = self.state[0] + dt * self.state[1]
        v = self.state[1] + dt * \
            (action[0] + self.rng.normal(loc=0, scale=0.5))
        
        self.state = np.clip(
                np.asarray([x, v]),
                self.observation_space.low,
                self.observation_space.high).astype(np.float32)
        
        self.steps += 1
        
        # Check for termination conditions
        terminated = bool(self.state[0] >= 0)
        truncated = self.steps >= self._max_episode_steps
        
        # Calculate reward
        reward = (2.0 + self.state[0]) if not terminated else -100.0
        
        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {}

    def render(self):
        """
        Renders the environment.
        - The blue car is the 'ego' vehicle (agent).
        - The red car is the 'lead' vehicle, which is stationary at x=0.
        - A crash occurs if the blue car's front touches the red car's back.
        """
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
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("ACC Environment")
            else:  # "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # --- Drawing ---
        
        # Create a surface to draw on
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((240, 240, 240)) # Light gray background

        # Draw the road
        road_y = self.screen_height * 0.6
        pygame.draw.line(canvas, (60, 60, 60), (0, road_y), (self.screen_width, road_y), 30)

        # World to screen scaling. We want to view the space from x=-10 to x=1
        view_width_world = 11.0
        scale = self.screen_width / view_width_world

        # Car dimensions
        car_width = 50
        car_height = 25

        # Draw the lead vehicle (at x=0, the crash point)
        # We position it on the right side of the screen for clarity.
        lead_car_world_x = 0.0
        lead_car_screen_x = self.screen_width * 0.8
        lead_car_rect = pygame.Rect(
            lead_car_screen_x, road_y - car_height - 2, car_width, car_height
        )
        pygame.draw.rect(canvas, (220, 50, 50), lead_car_rect) # Red

        # Draw the ego vehicle (agent)
        ego_x, ego_v = self.state
        # Its screen position is its world position relative to the lead car
        ego_screen_x = lead_car_screen_x + (ego_x - lead_car_world_x) * scale
        ego_car_rect = pygame.Rect(
            ego_screen_x, road_y - car_height - 2, car_width, car_height
        )
        pygame.draw.rect(canvas, (50, 100, 220), ego_car_rect) # Blue

        # --- Text Information ---
        font = pygame.font.Font(None, 32)
        info_color = (10, 10, 10)
        
        pos_text = font.render(f"Relative Position (x): {ego_x:.2f} m", True, info_color)
        vel_text = font.render(f"Relative Velocity (v): {ego_v:.2f} m/s", True, info_color)
        distance_text = font.render(f"Distance to lead: {-ego_x:.2f} m", True, info_color)
        
        canvas.blit(pos_text, (20, 20))
        canvas.blit(vel_text, (20, 55))
        canvas.blit(distance_text, (20, 90))
        
        # --- Safety Status Text ---
        is_unsafe = self.unsafe(self.state)
        if is_unsafe:
            status_text_str = "Status: Unsafe"
            status_color = (220, 50, 50)  # Red
        else:
            status_text_str = "Status: Safe"
            status_color = (50, 200, 50)  # Green

        status_font = pygame.font.Font(None, 40) # Slightly larger font
        status_text = status_font.render(status_text_str, True, status_color)
        
        # Position it at the top right corner
        status_rect = status_text.get_rect(topright=(self.screen_width - 20, 20))
        canvas.blit(status_text, status_rect)

        # --- Finalize Frame ---
        if self.render_mode == "human":
            # The screen and canvas are the same object, but this is good practice
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        # In either mode, return the array to support saving videos.
        # This prevents collecting `None` frames if render() is called for a `human` mode env.
        if self.render_mode in ["human", "rgb_array"]:
             return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 0

    def unsafe(self, state: np.ndarray, sim = None) -> bool:
        return state[0] >= 0

    def close(self):
        """Closes the rendering window."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

