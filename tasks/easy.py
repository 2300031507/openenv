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
