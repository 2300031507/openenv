import os
import sys
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Add the parent directory to sys.path to allow imports of env and tasks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CloudIncidentEnv, Action, Observation
from tasks.easy import TASK_CONFIG as easy_task
from tasks.medium import TASK_CONFIG as medium_task
from tasks.hard import TASK_CONFIG as hard_task

app = FastAPI(title="CloudIncidentEnv API")

TASKS_MAP = {
    "easy": easy_task,
    "easy_cpu_spike": easy_task,
    "medium": medium_task,
    "medium_storage_pressure": medium_task,
    "hard": hard_task,
    "hard_cascading_failure": hard_task
}

current_task_id = os.getenv("TASK_ID", "easy")
env = CloudIncidentEnv(TASKS_MAP.get(current_task_id, easy_task))

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "Environment is LIVE"}

@app.post("/reset")
def reset_env(request: ResetRequest = Body(default=None)):
    try:
        global env
        if request and request.task_id and request.task_id in TASKS_MAP:
            env = CloudIncidentEnv(TASKS_MAP[request.task_id])
        
        observation = env.reset()
        return {"observation": observation.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_env(action_data: Dict[str, Any] = Body(...)):
    try:
        if "action" in action_data:
            action_data = action_data["action"]
            
        action = Action(**action_data)
        observation, reward, done, info = env.step(action)
        
        return {
            "observation": observation.model_dump(),
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    """Entry point for the server."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
