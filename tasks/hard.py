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
