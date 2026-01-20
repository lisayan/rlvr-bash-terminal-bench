#!/usr/bin/env python3
"""
Verify candidate bash scripts using Terminal-Bench's actual test infrastructure.

Optimized version with parallelization:
- Pre-builds Docker images per task
- Runs sample verification in parallel within each task
- Configurable number of workers (default: 4 for MacBook)
"""

import json
import subprocess
import os
import tempfile
import tarfile
import re
from pathlib import Path
from collections import defaultdict
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread-safe file writing
write_lock = threading.Lock()


def parse_pytest_output(output: str) -> dict:
    """Parse pytest output to extract pass/fail counts."""
    passed = failed = errors = 0

    passed_match = re.search(r'(\d+) passed', output)
    failed_match = re.search(r'(\d+) failed', output)
    error_match = re.search(r'(\d+) error', output)

    if passed_match:
        passed = int(passed_match.group(1))
    if failed_match:
        failed = int(failed_match.group(1))
    if error_match:
        errors = int(error_match.group(1))

    total = passed + failed + errors
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0.0
    }


def strip_markdown_fences(content: str) -> str:
    """Remove markdown code fences from completion."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]
        content = "\n".join(lines)
    if content.rstrip().endswith("```"):
        content = content.rstrip()[:-3].rstrip()
    return content


def build_task_image(task_id: str, archive_path: str, work_dir: Path) -> tuple[bool, str, str]:
    """Build Docker image for a task. Returns (success, image_tag, error_msg)."""
    image_tag = f"tb-{task_id}"

    # Check if image already exists
    result = subprocess.run(
        ["docker", "images", "-q", image_tag],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        return True, image_tag, ""

    # Extract archive
    task_dir = work_dir / task_id
    task_dir.mkdir(exist_ok=True)

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(task_dir)
    except Exception as e:
        return False, "", f"Failed to extract archive: {e}"

    # Build image
    try:
        result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for complex builds
        )
    except subprocess.TimeoutExpired:
        return False, "", "Docker build timed out after 600 seconds"

    if result.returncode != 0:
        return False, "", f"Docker build failed: {result.stderr[:1000]}"

    return True, image_tag, ""


def verify_sample(item: dict, image_tag: str, task_dir: Path, test_timeout: int) -> dict:
    """Verify a single sample using pre-built Docker image."""
    script_content = strip_markdown_fences(item["completion"])

    # Create temp dir for this sample's solution
    with tempfile.TemporaryDirectory() as sample_dir:
        sample_path = Path(sample_dir)

        # Copy task files to sample dir
        for f in task_dir.iterdir():
            if f.is_file():
                shutil.copy(f, sample_path / f.name)
            elif f.is_dir():
                shutil.copytree(f, sample_path / f.name)

        # Write candidate's solution
        solution_path = sample_path / "solution.sh"
        with open(solution_path, "w") as f:
            if not script_content.strip().startswith("#!"):
                f.write("#!/bin/bash\nset -e\n")
            f.write(script_content)
        os.chmod(solution_path, 0o755)

        try:
            # Run solution
            solution_result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{sample_path}:/app",
                    "-w", "/app",
                    image_tag,
                    "bash", "-c", "chmod +x solution.sh && ./solution.sh"
                ],
                capture_output=True,
                text=True,
                timeout=test_timeout
            )

            # Run tests
            test_result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{sample_path}:/app",
                    "-v", f"{sample_path}/tests:/tests:ro",
                    "-e", "TEST_DIR=/tests",
                    "-w", "/app",
                    image_tag,
                    "bash", "-c", "chmod +x run-tests.sh && ./run-tests.sh"
                ],
                capture_output=True,
                text=True,
                timeout=test_timeout
            )

            # Parse results
            test_output = test_result.stdout + "\n" + test_result.stderr
            pytest_results = parse_pytest_output(test_output)

            return {
                **item,
                "reward": pytest_results["pass_rate"],
                "test_passed": pytest_results["passed"],
                "test_failed": pytest_results["failed"],
                "test_errors": pytest_results["errors"],
                "test_total": pytest_results["total"],
                "pass_rate": pytest_results["pass_rate"],
                "solution_exit_code": solution_result.returncode,
                "test_exit_code": test_result.returncode,
                "solution_stdout": solution_result.stdout[:3000],
                "solution_stderr": solution_result.stderr[:3000],
                "test_stdout": test_result.stdout[:3000],
                "test_stderr": test_result.stderr[:3000],
            }

        except subprocess.TimeoutExpired:
            return {
                **item,
                "reward": 0.0,
                "test_passed": 0,
                "test_failed": 0,
                "test_total": 0,
                "pass_rate": 0.0,
                "error": "timeout"
            }
        except Exception as e:
            return {
                **item,
                "reward": 0.0,
                "test_passed": 0,
                "test_failed": 0,
                "test_total": 0,
                "pass_rate": 0.0,
                "error": str(e)
            }


def verify_samples_parallel(samples: list, image_tag: str, task_dir: Path,
                            timeout: int, num_workers: int) -> list:
    """Verify multiple samples in parallel."""
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(verify_sample, sample, image_tag, task_dir, timeout): sample
            for sample in samples
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                sample = futures[future]
                results.append({
                    **sample,
                    "reward": 0.0,
                    "test_passed": 0,
                    "test_failed": 0,
                    "test_total": 0,
                    "pass_rate": 0.0,
                    "error": f"executor_error: {str(e)}"
                })

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="candidates_gemini_v2.jsonl")
    parser.add_argument("--output", default="verified_bash_rl.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4, good for MacBook)")
    parser.add_argument("--skip-truncated", action="store_true",
                        help="Skip candidates with finish_reason=length (truncated)")
    args = parser.parse_args()

    print(f"Using {args.workers} parallel workers")

    # Load candidates
    with open(args.input) as f:
        candidates = [json.loads(line) for line in f]

    if args.limit:
        candidates = candidates[:args.limit]

    # Skip truncated candidates if requested
    if args.skip_truncated:
        original_count = len(candidates)
        candidates = [c for c in candidates if c.get("finish_reason") != "length"]
        skipped = original_count - len(candidates)
        print(f"Skipping {skipped} truncated candidates (finish_reason=length)")

    # Group by task
    by_task = defaultdict(list)
    for c in candidates:
        by_task[c["task_id"]].append(c)

    print(f"Verifying {len(candidates)} candidates across {len(by_task)} tasks")

    # Resume support: load already-completed tasks
    completed_tasks = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                data = json.loads(line)
                completed_tasks.add(data["task_id"])
        if completed_tasks:
            print(f"Resuming: {len(completed_tasks)} tasks already completed, skipping them")

    # Load task metadata
    task_meta = {}
    if os.path.exists("terminal_bench_tasks.jsonl"):
        with open("terminal_bench_tasks.jsonl") as f:
            for line in f:
                t = json.loads(line)
                task_meta[t["task_id"]] = t

    # Create work directory for task extractions
    work_dir = Path(tempfile.mkdtemp(prefix="tb_verify_"))
    print(f"Work dir: {work_dir}")

    verified = []
    total_passed = 0

    try:
        with open(args.output, "a") as out_f:  # Append mode for resume
            for task_idx, (task_id, samples) in enumerate(by_task.items()):
                # Skip already-completed tasks (resume support)
                if task_id in completed_tasks:
                    print(f"  [{task_idx+1}/{len(by_task)}] {task_id}: already completed, skipping")
                    continue

                archive_path = f"task_archives/{task_id}.tar.gz"
                if not os.path.exists(archive_path):
                    print(f"  [{task_idx+1}/{len(by_task)}] {task_id}: archive not found, skipping")
                    for s in samples:
                        result = {**s, "reward": 0.0, "error": "archive_not_found",
                                  "test_passed": 0, "test_total": 0, "pass_rate": 0.0}
                        out_f.write(json.dumps(result) + "\n")
                        verified.append(result)
                    continue

                # Build image for this task
                print(f"  [{task_idx+1}/{len(by_task)}] {task_id}: building image...")
                success, image_tag, error = build_task_image(task_id, archive_path, work_dir)

                if not success:
                    print(f"    Build failed: {error[:100]}")
                    for s in samples:
                        result = {**s, "reward": 0.0, "error": "build_failed",
                                  "test_passed": 0, "test_total": 0, "pass_rate": 0.0}
                        out_f.write(json.dumps(result) + "\n")
                        verified.append(result)
                    continue

                # Get timeout for this task
                timeout = task_meta.get(task_id, {}).get("max_test_timeout_sec", args.task_timeout)
                task_dir = work_dir / task_id

                # Verify samples in parallel
                print(f"    Verifying {len(samples)} samples in parallel...")
                results = verify_samples_parallel(
                    samples, image_tag, task_dir, timeout, args.workers
                )

                # Write results and count passes
                task_passed = 0
                for result in results:
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    verified.append(result)

                    if result.get("test_passed", 0) > 0:
                        task_passed += 1
                        total_passed += 1

                print(f"    {task_passed}/{len(samples)} samples passed at least 1 test")

    finally:
        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)

    # Summary
    total = len(verified)
    perfect = sum(1 for v in verified if v.get("pass_rate", 0) == 1.0)
    avg_reward = sum(v.get("reward", 0) for v in verified) / total if total else 0

    print(f"\n=== Results ===")
    print(f"Total: {total}")
    print(f"Passed at least 1 test: {total_passed} ({100*total_passed/total:.1f}%)")
    print(f"Perfect (all tests): {perfect} ({100*perfect/total:.1f}%)")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
