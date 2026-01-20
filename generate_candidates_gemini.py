from openai import OpenAI
import json
import os
from tqdm import tqdm  # optional progress bar: pip install tqdm

# Setup client with timeout
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/",
    timeout=120.0  # 2 minute timeout per request
)

MODEL = "gemini-3-flash-preview"           # Change to gemini-2.5-flash-lite or gemini-3-flash-preview if desired
SAMPLES_PER_PROMPT = 16
TEMPERATURE = 1.0                    # High for diversity → good RL negatives/partials
MAX_TOKENS = 8192
OUTPUT_FILE = "candidates_gemini_v2.jsonl"

# Load your prompts
with open("terminal_bench_tasks.jsonl", "r") as f:
    prompts = [json.loads(line) for line in f]

# Check for existing progress (resume support)
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            completed.add((data["task_id"], data["sample_id"]))
    print(f"Resuming: {len(completed)} candidates already generated")

# Open file in append mode for incremental writes
with open(OUTPUT_FILE, "a") as outfile:
    for p in tqdm(prompts, desc="Generating completions"):
        prompt_text = p["prompt"]
        task_id = p["task_id"]

        for i in range(SAMPLES_PER_PROMPT):
            # Skip if already completed
            if (task_id, i) in completed:
                continue

            try:
                # Retry up to 3 times if truncated due to length
                for attempt in range(3):
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": "You are an expert bash scripter. Output ONLY the complete bash script to solve the task. No explanations, comments, or markdown."},
                            {"role": "user", "content": prompt_text}
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    script = response.choices[0].message.content.strip()
                    finish_reason = response.choices[0].finish_reason

                    if finish_reason != "length":
                        break
                    print(f"  {task_id} [{i+1}/{SAMPLES_PER_PROMPT}] truncated, retrying ({attempt+1}/3)")

                candidate = {
                    "task_id": task_id,
                    "prompt": prompt_text,
                    "sample_id": i,
                    "completion": script,
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "finish_reason": finish_reason
                }
                # Write immediately (incremental)
                outfile.write(json.dumps(candidate) + "\n")
                outfile.flush()

                print(f"  {task_id} [{i+1}/{SAMPLES_PER_PROMPT}] len={len(script)} finish={finish_reason}")
            except Exception as e:
                print(f"Error on {task_id} sample {i}: {e}")
                continue  # skip bad ones

# Count final results
with open(OUTPUT_FILE, "r") as f:
    count = sum(1 for _ in f)
print(f"Generated {count} candidates → saved to {OUTPUT_FILE}")