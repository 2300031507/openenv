TASK_CONFIG = {
    "id": "easy_disk_cleanup",
    "name": "Disk Cleanup",
    "description": "The disk is almost full. Clear logs to reclaim space.",
    "initial_health": 0.8,
    "initial_cpu": 0.3,
    "initial_error": 0.0,
    "initial_db_latency": 40.0,
    "initial_disk": 0.9,
    "initial_incidents": ["disk_full"],
    "initial_logs": ["Alert: Disk usage is 90%"],
    "max_steps": 5
}

def grade_task(final_observation, steps_taken, max_steps):
    """
    Returns a score between 0.0 and 1.0 for Extra Task.
    """
    if not final_observation.active_incidents and final_observation.service_health > 0.5:
        efficiency = (max_steps - steps_taken) / max_steps if max_steps > 0 else 0
        score = 0.8 + (0.2 * efficiency)
    else:
        incidents_penalty = len(final_observation.active_incidents) * 0.2
        score = max(0.0, min(1.0, final_observation.service_health - incidents_penalty))
    
    return round(float(score), 2)
