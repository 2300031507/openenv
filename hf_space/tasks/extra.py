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
