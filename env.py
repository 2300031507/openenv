import json
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    SCALE_UP = "scale_up"
    RESTART_SERVICE = "restart_service"
    CLEAR_LOGS = "clear_logs"
    FIX_DB_INDEX = "fix_db_index"
    ANALYZE_LOGS = "analyze_logs"
    ROLLBACK = "rollback"
    DO_NOTHING = "do_nothing"

class Action(BaseModel):
    action_type: ActionType
    target: str = "main_service"
    params: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    service_health: float = Field(..., ge=0.0, le=1.0)
    cpu_usage: float = Field(..., ge=0.0, le=1.0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    db_latency_ms: float
    disk_usage: float = Field(..., ge=0.0, le=1.0)
    logs: List[str]
    active_incidents: List[str]

class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    reason: str

class CloudIncidentEnv:
    def __init__(self, task_config: Dict[str, Any] = None):
        self.task_config = task_config or {}
        self.reset()

    def reset(self) -> Observation:
        # Initial state based on task config
        self._service_health = self.task_config.get("initial_health", 0.8)
        self._cpu_usage = self.task_config.get("initial_cpu", 0.4)
        self._error_rate = self.task_config.get("initial_error", 0.05)
        self._db_latency = self.task_config.get("initial_db_latency", 50.0)
        self._disk_usage = self.task_config.get("initial_disk", 0.3)
        self._logs = self.task_config.get("initial_logs", ["System check: OK"])
        self._active_incidents = self.task_config.get("initial_incidents", [])
        self._step_count = 0
        self._done = False
        return self.state()

    def state(self) -> Observation:
        return Observation(
            service_health=round(self._service_health, 2),
            cpu_usage=round(self._cpu_usage, 2),
            error_rate=round(self._error_rate, 2),
            db_latency_ms=round(self._db_latency, 2),
            disk_usage=round(self._disk_usage, 2),
            logs=self._logs[-5:], # Return last 5 logs
            active_incidents=self._active_incidents
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            return self.state(), 0.0, True, {"msg": "Environment already done"}

        self._step_count += 1
        reward_val = 0.1 # Baseline small reward for surviving a step
        info = {"action_taken": action.action_type}

        # Logic for each action
        if action.action_type == ActionType.SCALE_UP:
            if "high_cpu" in self._active_incidents:
                self._cpu_usage = max(0.2, self._cpu_usage - 0.4)
                self._service_health = min(1.0, self._service_health + 0.2)
                self._active_incidents.remove("high_cpu")
                # Varied rewards based on how high the CPU was
                reward_val = 0.5 + (0.3 * self._cpu_usage) 
                self._logs.append("Scaling up: Resources increased, CPU stabilized.")
            else:
                reward_val = 0.05 # Tiny reward for checking, but mostly penalty for waste
                self._logs.append("Scaling up: No CPU pressure detected. Resource waste.")

        elif action.action_type == ActionType.RESTART_SERVICE:
            if "memory_leak" in self._active_incidents:
                self._service_health = min(1.0, self._service_health + 0.3)
                self._error_rate = max(0.0, self._error_rate - 0.1)
                self._active_incidents.remove("memory_leak")
                # Varied reward based on health improvement
                reward_val = 0.6 + (0.2 * (1.0 - self._service_health))
                self._logs.append("Restarting: Memory cleared, health improved.")
            else:
                self._service_health = max(0.0, self._service_health - 0.1)
                reward_val = 0.0
                self._logs.append("Restarting: Service interrupted without cause.")

        elif action.action_type == ActionType.CLEAR_LOGS:
            if "disk_full" in self._active_incidents:
                # Varied reward based on disk pressure before clearing
                reward_val = 0.4 + (0.4 * self._disk_usage)
                self._disk_usage = 0.1
                self._active_incidents.remove("disk_full")
                self._logs.append("Logs cleared: Disk space reclaimed.")
            else:
                reward_val = 0.05
                self._logs.append("Logs cleared: No disk pressure.")

        elif action.action_type == ActionType.FIX_DB_INDEX:
            if "slow_queries" in self._active_incidents:
                # Varied reward based on latency
                reward_val = 0.5 + min(0.4, self._db_latency / 1000.0)
                self._db_latency = 20.0
                self._active_incidents.remove("slow_queries")
                self._logs.append("Index fixed: Database latency normalized.")
            else:
                reward_val = 0.0
                self._logs.append("Index fixed: No slow queries detected.")

        elif action.action_type == ActionType.ANALYZE_LOGS:
            reward_val = 0.2 # Analysis is good for debugging
            self._logs.append(f"Analysis: Found {len(self._active_incidents)} issues.")

        # Environmental decay/progression
        if "high_cpu" in self._active_incidents:
            self._service_health -= 0.05
            self._error_rate += 0.02
        if "disk_full" in self._active_incidents:
            self._error_rate += 0.05
            self._service_health -= 0.1

        # Check for failure/success
        if self._service_health <= 0.0:
            self._done = True
            reward_val = 0.0 # Final failure reward
            self._logs.append("CRITICAL: Service Down.")
        elif not self._active_incidents:
            self._done = True
            reward_val = 0.8 # Success reward
            self._logs.append("SUCCESS: All incidents resolved.")

        # Final reward clipping to strictly 0.0 to 1.0
        reward_val = max(0.0, min(1.0, float(reward_val)))
        
        # Clip state values
        self._service_health = max(0.0, min(1.0, self._service_health))
        self._cpu_usage = max(0.0, min(1.0, self._cpu_usage))
        self._error_rate = max(0.0, min(1.0, self._error_rate))
        self._disk_usage = max(0.0, min(1.0, self._disk_usage))

        if self._step_count >= self.task_config.get("max_steps", 10):
            self._done = True

        return self.state(), reward_val, self._done, info
