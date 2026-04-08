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
