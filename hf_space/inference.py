import os
import sys
import json
from openai import OpenAI
from env import CloudIncidentEnv, Action, ActionType
from tasks.easy import TASK_CONFIG as easy_task
from tasks.medium import TASK_CONFIG as medium_task
from tasks.hard import TASK_CONFIG as hard_task
from tasks.extra import TASK_CONFIG as extra_task
from graders.grader import grade

def run_inference(task_config: dict):
    # Setup from environment variables
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    
    is_mock = False
    if not hf_token or hf_token == "MOCK":
        print("Note: HF_TOKEN missing or set to MOCK. Running with Mock Agent.")
        is_mock = True
    
    # Initialize OpenAI client
    client = None
    if not is_mock:
        client = OpenAI(base_url=api_base_url, api_key=hf_token)
    
    env = CloudIncidentEnv(task_config)
    obs = env.reset()
    
    task_name = task_config["id"]
    env_name = "CloudIncidentEnv-v1"
    
    # [START] task=<task_name> env=<env_name> model=<model_name>
    print(f"[START] task={task_name} env={env_name} model={model_name if not is_mock else 'mock-agent'}")
    
    steps = 0
    rewards = []
    done = False
    
    while not done and steps < task_config.get("max_steps", 10):
        error_msg = "null"
        action = None

        if is_mock:
            # Simple heuristic mock agent for perfect logs
            if "high_cpu" in obs.active_incidents:
                action = Action(action_type=ActionType.SCALE_UP)
            elif "disk_full" in obs.active_incidents:
                action = Action(action_type=ActionType.CLEAR_LOGS)
            elif "slow_queries" in obs.active_incidents:
                action = Action(action_type=ActionType.FIX_DB_INDEX)
            elif "memory_leak" in obs.active_incidents:
                action = Action(action_type=ActionType.RESTART_SERVICE)
            else:
                action = Action(action_type=ActionType.DO_NOTHING)
        else:
            # Build prompt for the LLM agent
            prompt = f"""You are an expert Cloud Reliability Engineer. 
            Your goal is to resolve infrastructure incidents efficiently.
            
            Current State:
            - Health: {obs.service_health}
            - CPU: {obs.cpu_usage}
            - Errors: {obs.error_rate}
            - Latency: {obs.db_latency_ms}ms
            - Disk: {obs.disk_usage}
            - Recent Logs: {obs.logs}
            - Active Incidents: {obs.active_incidents}
            
            Available Actions:
            - scale_up: Increase resources (fix high CPU)
            - restart_service: Clear temporary issues/memory leaks
            - clear_logs: Reclaim disk space
            - fix_db_index: Resolve database latency
            - analyze_logs: Gather more information
            - do_nothing: Wait for system stabilization
            
            Respond with ONLY a JSON object: {{"action_type": "one_of_the_above", "target": "main_service"}}"""
            
            try:
                # Get action from model
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=100
                )
                
                action_content = response.choices[0].message.content
                action_data = json.loads(action_content)
                action = Action(**action_data)
            except Exception as e:
                # Fallback to safe action
                action = Action(action_type=ActionType.DO_NOTHING)
                error_msg = str(e).replace("\n", " ")
            
        # Execute step
        obs, reward, done, info = env.step(action)
        steps += 1
        rewards.append(reward)
        
        # [STEP] step= action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        print(f"[STEP] step={steps} action={action.action_type.value} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
    
    # Grading
    score = grade(obs, steps, task_config["max_steps"])
    success = score >= 0.8
    
    # [END] success=<true|false> steps= score= rewards=<r1,r2,...>
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    # Standard evaluation script entry point
    target_task_id = os.getenv("TASK_ID", "easy")
    
    tasks_map = {
        "easy": easy_task,
        "medium": medium_task,
        "hard": hard_task,
        "extra": extra_task,
        "disk_cleanup": extra_task
    }
    
    selected_task = tasks_map.get(target_task_id, easy_task)
    run_inference(selected_task)
