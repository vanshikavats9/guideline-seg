import json
import os
import random
from pathlib import Path

STEP_COST          = 0.02   # penalty per extra loop
EARLY_STOP_PENALTY = 2.0    # penalty for stopping while still dirty
FINAL_BONUS        = 1.0    # bonus for reaching zero issues
MIN_ITERS          = 2      # force at least this many loops
MAX_ITERS          = 4      # never exceed this many loops
EPSILON            = 0.02   # epsilon-greedy exploration rate
ALPHA              = 0.3    # learning rate
GAMMA              = 0.9    # discount factor
N_CROWD_BUCKETS    = 3      # few/medium/crowded buckets

# actions
STOP     = 0
CONTINUE = 1

def crowd_bucket(obj_cnt: int) -> int:
    # discretise the scene by object count:
    # 0 = few   (<=3)
    # 1 = medium(4-7)
    # 2 = crowded(>=8)
    if obj_cnt <= 3:
        return 0
    if obj_cnt <= 7:
        return 1
    return 2

class RefineAgent:
    def __init__(self, state_path):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # load or initialise Q-table: states = 0-5, actions = {STOP, CONTINUE}
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text())
            self.Q = {int(s): {int(a): v for a, v in d.items()} 
                      for s, d in raw.items()}
        else:
            # biased initialization: CONTINUE is slightly preferred when dirty
            self.Q = {
                s: {
                    STOP:     (-0.5 if s % 2 == 1 else 0.0),
                    CONTINUE: ( 0.1 if s % 2 == 1 else 0.0)
                }
                for s in range(N_CROWD_BUCKETS * 2)
            }
        self.alpha   = ALPHA
        self.gamma   = GAMMA
        self.epsilon = EPSILON

    def act(self, issue_cnt: int, obj_cnt: int, iteration: int) -> int:
        """
        decide whether to STOP (0) or CONTINUE (1)
        - always CONTINUE if iteration < MIN_ITERS
        - always STOP if iteration >= MAX_ITERS
        - o/w epsilon-greedy on Q[state]
        """
        state = crowd_bucket(obj_cnt) * 2 + (1 if issue_cnt > 0 else 0)

        if iteration < MIN_ITERS:
            return CONTINUE
        if iteration >= MAX_ITERS:
            return STOP

        if random.random() < self.epsilon:
            return random.choice([STOP, CONTINUE])
        # exploit
        # return the action with the highest Q-value for the current state
        return max(self.Q[state], key=self.Q[state].get)

    def update(self, prev_issues: int, new_issues: int, obj_cnt: int, action: int) -> None:
        # q-learning update
        prev_state = crowd_bucket(obj_cnt) * 2 + (1 if prev_issues > 0 else 0)
        new_state  = crowd_bucket(obj_cnt) * 2 + (1 if new_issues > 0 else 0)

        # compute reward
        if action == CONTINUE:
            delta  = prev_issues - new_issues
            reward = delta - STEP_COST
            if new_issues == 0 and delta > 0:
                reward += FINAL_BONUS
        else:  # STOP
            reward = -EARLY_STOP_PENALTY if prev_issues > 0 else 0.0

        # bellman update
        best_next = max(self.Q[new_state].values())
        self.Q[prev_state][action] += self.alpha * (
            reward + self.gamma * best_next
            - self.Q[prev_state][action]
        )
        self._persist()

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.Q))
        os.replace(tmp_path, self.state_path)
