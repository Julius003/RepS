from data.mbpp_plus import MBPPPlus
import torch
from transformer_lens.model_bridge import TransformerBridge
import sys, os, json, tempfile, subprocess


# One solved example so the base model sees the expected format.
FEW_SHOT = '''\
You are an expert Python programmer, and here is your task: Write a function to find the minimum of two numbers.
Your code should pass these tests:

assert minimum(1, 2) == 1
assert minimum(-5, -4) == -5
assert minimum(0, 0) == 0

[BEGIN]
```python
def minimum(a, b):
    return a if a < b else b
```
[DONE]
'''
TEST_TIMEOUT_S = 3

def build_prompt(sample: dict) -> str:
    tests = "\n".join(sample["test_list"])
    task = (
        f"You are an expert Python programmer, and here is your task: {sample['prompt']}\n"
        f"Your code should pass these tests:\n\n"
        f"{tests}\n\n"
        f"[BEGIN]\n"
        f"```python\n"
    )
    return FEW_SHOT + task

def extract_completion(prompt: str, answer) -> str:
    return answer[len(prompt):]

def extract_code(completion: str) -> str:
    for stop in ("```", "[DONE]"):
        idx = completion.find(stop)
        if idx != -1:
            completion = completion[:idx]
    return completion.strip()


def run_tests(code: str, test_list: list[str]) -> tuple[bool, str]:
    """Run code + asserts in a separate process. Returns (passed, error_output)."""
    program = code + "\n\n" + "\n".join(test_list) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_S,
        )
        return result.returncode == 0, result.stderr[-500:]
    except subprocess.TimeoutExpired:
        return False, "Timeout"

def last_done_task_id(path):
    last = None
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tid = json.loads(line)["task_id"]
                except (json.JSONDecodeError, KeyError):
                    continue  # ignore a half-written line
                if last is None or tid > last:
                    last = tid
    return last


# Given datasets like MBPP+ don't come with incorrect solutions
# -> generate own incorrect solutions by calling an LLM 
#    and seeing if the tests fail
def generate_incorrect_solutions(ds: MBPPPlus, out_path: str = "incorrect_solutions.jsonl"):
    models = ["Qwen/Qwen2.5-Coder-7B"]
    bridge = TransformerBridge.boot_transformers(models[0], device="cpu", dtype=torch.bfloat16)

    # Load already generated solutions
    last = last_done_task_id(out_path)

    with torch.inference_mode(), open(out_path, "a") as out:
        for sample in ds:
            if last is not None and sample["task_id"] <= last: 
                print(f"{sample["task_id"]}: skipped")
                continue
            prompt = build_prompt(sample)
            answer = bridge.generate(
                prompt,
                max_new_tokens=512,
                do_sample=False,
                stop_at_eos=True,
                return_type="str",
                verbose=False,
                    )
            completion = extract_completion(prompt, answer)
            code = extract_code(completion)

            # Test generated solution
            success, error_message = run_tests(code, sample["test_list"])
            print(f"{sample["task_id"]}: success={success}")
            if not success:
                # the generated code does not work => keep this implementation as faulty code
                out.write(json.dumps({
                    "task_id": sample["task_id"],
                    "prompt": sample["prompt"],
                    "incorrect_code": code,
                    "error": error_message,
                }) + "\n")
                out.flush()
        print("Done")
        return


if __name__ == "__main__":
    generate_incorrect_solutions(MBPPPlus(), "data/incorrect_solutions.jsonl")
