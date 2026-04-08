TASK_CONFIG = {
    "id": "easy_cpu_spike",
    "name": "CPU Spike Response",
    "description": "A sudden CPU spike is affecting service health. Scale up to stabilize.",
    "initial_health": 0.7,
    "initial_cpu": 0.9,
    "initial_error": 0.05,
    "initial_db_latency": 50.0,
    "initial_disk": 0.3,
    "initial_incidents": ["high_cpu"],
    "initial_logs": ["Alert: CPU usage over 90% on main_service"],
    "max_steps": 5
}

def grade_task(final_observation, steps_taken, max_steps):
    """
    Returns a score between 0.0 and 1.0 for Easy Task.
    """
    if not final_observation.active_incidents and final_observation.service_health > 0.5:
        efficiency = (max_steps - steps_taken) / max_steps if max_steps > 0 else 0
        score = 0.8 + (0.2 * efficiency)
    else:
        incidents_penalty = len(final_observation.active_incidents) * 0.2
        score = max(0.0, min(1.0, final_observation.service_health - incidents_penalty))
    
    return round(float(score), 2)
