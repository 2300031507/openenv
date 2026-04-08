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
