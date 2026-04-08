TASK_CONFIG = {
    "id": "hard_cascading_failure",
    "name": "Cascading Database and Memory Failure",
    "description": "Database latency is high and a memory leak is suspected. Resolve all issues before service crash.",
    "initial_health": 0.5,
    "initial_cpu": 0.4,
    "initial_error": 0.2,
    "initial_db_latency": 250.0,
    "initial_disk": 0.2,
    "initial_incidents": ["slow_queries", "memory_leak"],
    "initial_logs": ["Alert: DB latency > 200ms", "Alert: Error rate climbing", "Service logs: OutOfMemoryError in worker-7"],
    "max_steps": 15
}

def grade_task(final_observation, steps_taken, max_steps):
    """
    Returns a score between 0.0 and 1.0 for Hard Task.
    """
    if not final_observation.active_incidents and final_observation.service_health > 0.5:
        efficiency = (max_steps - steps_taken) / max_steps if max_steps > 0 else 0
        score = 0.8 + (0.2 * efficiency)
    else:
        incidents_penalty = len(final_observation.active_incidents) * 0.2
        score = max(0.0, min(1.0, final_observation.service_health - incidents_penalty))
    
    return round(float(score), 2)
