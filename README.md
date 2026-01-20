# rlvr-bash-terminal-bench

RLVR (Reinforcement Learning with Verifiable Rewards) dataset for bash scripting, generated from [Terminal-Bench](https://github.com/terminal-bench/terminal-bench) tasks.

## Stats

| Metric | Value |
|--------|-------|
| Total samples | 1,120 |
| Unique tasks | 88 |
| Avg samples/task | 12.7 |
| Average reward | 0.249 |
| Perfect solutions (reward=1.0) | 10.4% |
| Partial solutions (0<reward<1) | 28.8% |
| Zero reward | 60.8% |
| Tasks fully solved | 13.6% |

## Format

```json
{
  "task_id": "string",
  "prompt": "string",
  "completion": "string (bash script)",
  "reward": 0.0-1.0,
  "pass_rate": 0.0-1.0,
  "test_passed": int,
  "test_failed": int,
  "test_total": int,
  "model": "gemini-3-flash-preview",
  "temperature": 1.0
}
```

## Bash Script Generation

- **Model**: Gemini 3 Flash Preview
- **Verification**: Docker containers with pytest test suites from Terminal-Bench
- **Reward**: `test_passed / test_total` (fraction of pytest tests passed after script execution)

## Verification Infrastructure

Each task archive (`task_archives/<task_id>.tar.gz`) contains:

```
├── Dockerfile           # Execution environment
├── run-tests.sh         # Pytest entrypoint
├── task.yaml            # Task metadata
├── solution.sh          # Reference solution (excluded from dataset)
└── tests/
    └── test_outputs.py  # Pytest assertions
```

Verification flow:
1. Build Docker image from task's Dockerfile
2. Execute candidate bash script in container
3. Run pytest against `tests/test_outputs.py`
4. Parse pass/fail counts → reward

## Caveats

1. **Small scale**: 1,120 samples across 88 tasks.

2. **Single model source**: All completions from Gemini 3 Flash.

3. **High failure rate**: 60.8% of samples have zero reward. Terminal-Bench tasks are difficult system administration challenges (kernel builds, networking, Docker).

4. **Task difficulty varies widely**: Tasks range from "hello-world" to "build-linux-kernel-qemu". Reward comparisons across tasks are not meaningful.

5. **Non-deterministic verification**: Some tasks involve stochastic elements. Re-running verification may yield slightly different results.

6. **No ground truth solutions**: Per Terminal-Bench's request, reference solutions are not included to preserve benchmark integrity.

## Files

- `rlvr-bash-terminal-bench.jsonl` - Final dataset (1,120 samples across 88 tasks)
- `candidates.jsonl` - Raw script generations
- `verified.jsonl` - Verified results