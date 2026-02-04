import numpy as np
import time
from typing import Optional, Tuple, Any

from src.shield import VDK_Runtime
from src.vdk_shield import VDK_Shield

class Shield:
    """
    Construct a shield from a neural policy and a safety layer.
    """

    def __init__(
            self,
            shield_policy: Any,
            unsafe_policy: Any = None,
            means: Optional[np.ndarray] = None, 
            stds: Optional[np.ndarray] = None
    ):
        self.shield = shield_policy
        self.agent = unsafe_policy
        self.shield_times = 0
        self.backup_times = 0
        self.agent_times = 0
        self.total_time = 0.
        self.means = means
        self.stds = stds

    def __call__(self, state: np.ndarray, action: Optional[np.ndarray] = None, evaluate: bool = False, **kwargs: Any) -> Tuple[np.ndarray, str, float, np.ndarray]:
        start = time.time()
        if action is not None:
            proposed_action = action
        else:
            # Pass evaluate flag to the policy (agent)
            try:
                proposed_action = self.agent(state, evaluate=evaluate, **kwargs)
            except TypeError:
                # Fallback if agent doesn't support evaluate kwarg (e.g. simple function)
                proposed_action = self.agent(state, **kwargs)
            
        if self.means is not None:
            state = (state - self.means) / self.stds

        
        if self.shield.unsafe(state, proposed_action):
            act, shielded  = self.shield(state, action=proposed_action, **kwargs) # Pass kwargs (e.g. use_slacks) to shield
            self.shield_times += 1 if shielded else 0
            self.backup_times += 1 if not shielded else 0
            shielded = "SHIELD" if shielded else "BACKUP"
        else:
            act = proposed_action
            shielded = "NEURAL"
            self.agent_times += 1
        end = time.time()
        self.total_time += end - start
        
        # print(f"Shield: {shielded}, Action: {act}, Time: {end - start:.4f}s")
        return act, shielded, np.linalg.norm(act - proposed_action), proposed_action

    def report(self) -> Tuple[int, int, int, float]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self) -> None:
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0


class ShieldPolicy:
    """
    Wrapper for VDK_Runtime to match the interface expected by the Shield class.
    """
    def __init__(self, runtime: VDK_Runtime):
        self.shield_runtime = runtime
        
    def __call__(self, state: np.ndarray, action: Optional[np.ndarray] = None, base_action: Optional[np.ndarray] = None, **kwargs: Any) -> Tuple[np.ndarray, bool]:
        # Handle positional or keyword action
        u_rl = action if action is not None else base_action
        if u_rl is None:
             raise ValueError("ShieldPolicy requires an action to shield (u_rl).")
             
        # Assume state is already normalized if Shield class handles normalization
        u_safe, status = self.shield_runtime.solve_shield(state, u_rl)
        
        shielded = False
        if status == 'feasible':
            if np.linalg.norm(u_safe - u_rl) > 1e-4:
                shielded = True
        elif status == 'relaxed':
            shielded = True
        elif status == 'infeasible':
            shielded = True # Should backup?
            
        return u_safe, shielded

    def unsafe(self, state: np.ndarray, action: np.ndarray, **kwargs: Any) -> bool:
        _, shielded = self.__call__(state, action, **kwargs)
        return shielded
