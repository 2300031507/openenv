import os
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from env import CloudIncidentEnv, Action, Observation
from tasks.easy import TASK_CONFIG as easy_task
from tasks.medium import TASK_CONFIG as medium_task
from tasks.hard import TASK_CONFIG as hard_task

app = FastAPI(title="CloudIncidentEnv API")

# Global environment instance
task_id = os.getenv("TASK_ID", "easy")
tasks_map = {
    "easy": easy_task,
    "medium": medium_task,
    "hard": hard_task
}
selected_task = tasks_map.get(task_id, easy_task)
env = CloudIncidentEnv(selected_task)

@app.get("/")
def read_root():
    return {"status": "Environment is LIVE"}

@app.post("/reset")
def reset_env():
    try:
        observation = env.reset()
        # Return observation directly as a dict matching Observation model
        return {"observation": observation.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_env(action: Action = Body(...)):
    try:
        observation, reward, done, info = env.step(action)
        return {
            "observation": observation.model_dump(),
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
