import os
import json
from fastapi import FastAPI, Body, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Relative imports to ensure they work in both local and Docker environments
from env import CloudIncidentEnv, Action, Observation
from tasks.easy import TASK_CONFIG as easy_task
from tasks.medium import TASK_CONFIG as medium_task
from tasks.hard import TASK_CONFIG as hard_task
from inference import run_inference

app = FastAPI(title="CloudIncidentEnv API")

# Helper to map task IDs to configs
TASKS_MAP = {
    "easy": easy_task,
    "easy_cpu_spike": easy_task,
    "medium": medium_task,
    "medium_storage_pressure": medium_task,
    "hard": hard_task,
    "hard_cascading_failure": hard_task
}

# Global environment instance, initialized with the default task
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
        # If a specific task_id is requested in the reset, re-initialize the environment
        if request and request.task_id and request.task_id in TASKS_MAP:
            env = CloudIncidentEnv(TASKS_MAP[request.task_id])
        
        observation = env.reset()
        # Ensure we return a dictionary for JSON serialization
        return {"observation": observation.model_dump()}
    except Exception as e:
        print(f"Reset Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_env(action_data: Dict[str, Any] = Body(...)):
    try:
        # Manually parse the action to be more resilient to input formats
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
        print(f"Step Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.on_event("startup")
async def startup_event():
    # Run the evaluation once on startup to print logs for judges
    print("--- AUTO-EVALUATION STARTUP ---")
    target_task_id = os.getenv("TASK_ID", "easy")
    selected_task = TASKS_MAP.get(target_task_id, easy_task)
    # Run in background so the server can finish starting and bind to port 7860
    import threading
    thread = threading.Thread(target=run_inference, args=(selected_task,))
    thread.start()

if __name__ == "__main__":
    import uvicorn
    # Use the port required by Hugging Face Spaces (7860)
    uvicorn.run(app, host="0.0.0.0", port=7860)
