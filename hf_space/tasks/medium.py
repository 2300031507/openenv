TASK_CONFIG = {
    "id": "medium_storage_pressure",
    "name": "Storage and CPU Pressure",
    "description": "Both disk usage and CPU are high. Fix both to restore full health.",
    "initial_health": 0.6,
    "initial_cpu": 0.85,
    "initial_error": 0.1,
    "initial_db_latency": 55.0,
    "initial_disk": 0.95,
    "initial_incidents": ["high_cpu", "disk_full"],
    "initial_logs": ["Alert: High CPU usage", "Alert: Disk usage critical"],
    "max_steps": 10
}

def grade_task(final_observation, steps_taken, max_steps):
    """
    Returns a score between 0.0 and 1.0 for Medium Task.
    """
    if not final_observation.active_incidents and final_observation.service_health > 0.5:
        efficiency = (max_steps - steps_taken) / max_steps if max_steps > 0 else 0
        score = 0.8 + (0.2 * efficiency)
    else:
        incidents_penalty = len(final_observation.active_incidents) * 0.2
        score = max(0.0, min(1.0, final_observation.service_health - incidents_penalty))
    
    return round(float(score), 2)
