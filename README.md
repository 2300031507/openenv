---
title: Cloud Infrastructure Incident Response
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Cloud Infrastructure Incident Response (CIIR) RL Environment

An advanced Meta OpenEnv submission for a real-world Cloud Reliability Engineering (CRE) simulation.

## Overview
This environment simulates a cloud infrastructure under stress. It challenges an RL agent (LLM-based) to act as a Reliability Engineer, responding to alerts, mitigating service degradation, and resolving cascading failures.

## Action Space
The agent can take the following actions:
- `scale_up`: Increases server capacity to handle CPU spikes.
- `restart_service`: Resets state to clear memory leaks or transient errors.
- `clear_logs`: Reclaims disk space by purging old logs.
- `fix_db_index`: Optimizes database performance to reduce latency.
- `analyze_logs`: Provides additional insights into active incidents.
- `do_nothing`: Wait for the system to stabilize.

## Observation Space
The environment provides the following telemetry (Pydantic models):
- `service_health` (0.0 to 1.0): Overall system uptime and responsiveness.
- `cpu_usage` (0.0 to 1.0): Current load on compute resources.
- `error_rate` (0.0 to 1.0): Frequency of failed user requests.
- `db_latency_ms`: Response time for database queries.
- `disk_usage` (0.0 to 1.0): Storage occupancy.
- `logs`: Last 5 system messages.
- `active_incidents`: List of ongoing issues (e.g., `high_cpu`, `disk_full`).

## Tasks
1. **Easy: CPU Spike Response**
   - Goal: Recognize a CPU alert and scale resources.
2. **Medium: Storage and CPU Pressure**
   - Goal: Manage multiple resources simultaneously without exhausting the budget/steps.
3. **Hard: Cascading Database and Memory Failure**
   - Goal: Identify root causes (OOM, slow queries) and resolve them before a full service outage.

## Reward System
- **Partial Progress**: Rewards for resolving individual incidents (e.g., +0.4 for scaling up during a spike).
- **Penalties**: Penalties for unnecessary or harmful actions (e.g., -0.2 for restarting a healthy service).
- **Final Score**: A deterministic grader in `graders/grader.py` calculates a score from 0.0 to 1.0 based on final health and efficiency.

## Setup Instructions

### Local Development
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```bash
   export API_BASE_URL="https://api-inference.huggingface.co/v1/"
   export MODEL_NAME="meta-llama/Llama-3-70b-instruct"
   export HF_TOKEN="your_huggingface_token"
   ```
4. Run the environment:
   ```bash
   python inference.py
   ```

### Docker
1. Build the image:
   ```bash
   docker build -t openenv-cloud-incident .
   ```
2. Run with environment variables:
   ```bash
   docker run -e HF_TOKEN="your_token" openenv-cloud-incident
   ```

## Requirements
- Python 3.10+
- OpenAI Client
- Pydantic
- Hugging Face Token (for inference)
