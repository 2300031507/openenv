def grade_task(final_observation, steps_taken, max_steps):
    """
    Returns a score between 0.0 and 1.0.
    """
    # Success condition: All incidents resolved and health > 0.5
    if not final_observation.active_incidents and final_observation.service_health > 0.5:
        # Efficiency bonus: more steps remaining = higher score
        efficiency = (max_steps - steps_taken) / max_steps if max_steps > 0 else 0
        score = 0.8 + (0.2 * efficiency)
    else:
        # Partial score based on remaining incidents and final health
        incidents_penalty = len(final_observation.active_incidents) * 0.2
        # Ensure score doesn't drop below 0 or above 1
        score = max(0.0, min(1.0, final_observation.service_health - incidents_penalty))
    
    return round(float(score), 2)
