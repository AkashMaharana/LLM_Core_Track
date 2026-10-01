# !pip install -q transformers accelerate bitsandbytes
# This script demonstrates the evaluation of two LLMs (GPT-5.1 and Qwen2.5-Coder-7B) in translating Python code to optimized C++ code.
# It needs to be executed in Google Colab Envirobnment with GPU enabled.

# Colab Link : https://colab.research.google.com/drive/1n7Fc4nNZHf_NzJDBImV_xQXyrayUFZ8W#scrollTo=bUHGqVau8mcY  

import subprocess
import time

import torch

from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


def main():

    client = OpenAI()

    # ---------------------------------------------------------
    # Challenges
    # ---------------------------------------------------------

    challenges = [
        {
            "name": "Numeric Loop",
            "code": """
total = 0

for i in range(10_000_000):
    total += (i * i) % 97

print(total)
"""
        },

        {
            "name": "Prime Count",
            "code": """
import math

count = 0

for number in range(2, 200_000):
    is_prime = True

    limit = int(math.sqrt(number)) + 1

    for divisor in range(2, limit):
        if number % divisor == 0:
            is_prime = False
            break

    if is_prime:
        count += 1

print(count)
"""
        },

        {
            "name": "String Processing",
            "code": """
text = "distributed systems and artificial intelligence " * 500_000

count = 0

for word in text.split():
    if word.startswith("a"):
        count += 1

print(count)
"""
        },

        {
            "name": "Dynamic Programming",
            "code": """
numbers = [
    ((i * 37) % 1000)
    for i in range(3000)
]

dp = [1] * len(numbers)

for i in range(len(numbers)):
    for j in range(i):
        if numbers[j] < numbers[i]:
            dp[i] = max(dp[i], dp[j] + 1)

print(max(dp))
"""
        }
    ]

    # =========================================================
    # Load Qwen Coder
    # =========================================================

    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print("Qwen loaded.")
    print(
        "Memory:",
        round(
            qwen_model.get_memory_footprint() / 1024**3,
            2
        ),
        "GB"
    )

    results = []

    # =========================================================
    # Evaluate
    # =========================================================

    for challenge_index, challenge in enumerate(challenges):

        print("\n" + "=" * 70)
        print(challenge["name"])
        print("=" * 70)

        python_code = challenge["code"]

        prompt = f"""
Convert the following Python program into optimized C++.

Requirements:
- Preserve exactly the same behavior.
- The output must be identical to the Python program.
- Optimize the implementation where appropriate.
- Use appropriate C++ data types.
- Avoid integer overflow.
- Use C++17.
- Return only valid C++ source code.
- Do not include Markdown code fences.
- Do not include explanations.

Python:

{python_code}
"""

        # -----------------------------------------------------
        # Run original Python once
        # -----------------------------------------------------

        python_file = f"python_{challenge_index}.py"

        with open(python_file, "w") as file:
            file.write(python_code)

        python_start = time.perf_counter()

        python_result = subprocess.run(
            ["python", python_file],
            capture_output=True,
            text=True
        )

        python_time = time.perf_counter() - python_start

        expected_output = python_result.stdout.strip()

        # =====================================================
        # MODEL 1 — GPT-5.1
        # =====================================================

        print("\nGPT-5.1")

        start = time.perf_counter()

        response = client.responses.create(
            model="gpt-5.1",
            input=prompt
        )

        gpt_generation_time = (
            time.perf_counter() - start
        )

        gpt_code = response.output_text.strip()

        gpt_code = (
            gpt_code
            .replace("```cpp", "")
            .replace("```c++", "")
            .replace("```", "")
            .strip()
        )

        gpt_cpp_file = f"gpt_{challenge_index}.cpp"
        gpt_executable = f"gpt_{challenge_index}"

        with open(gpt_cpp_file, "w") as file:
            file.write(gpt_code)

        compile_result = subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                gpt_cpp_file,
                "-o",
                gpt_executable
            ],
            capture_output=True,
            text=True
        )

        gpt_compiled = compile_result.returncode == 0
        gpt_correct = False
        gpt_speedup = None

        if gpt_compiled:

            start = time.perf_counter()

            execution = subprocess.run(
                [f"./{gpt_executable}"],
                capture_output=True,
                text=True
            )

            cpp_time = time.perf_counter() - start

            gpt_correct = (
                execution.stdout.strip()
                == expected_output
            )

            if gpt_correct:
                gpt_speedup = python_time / cpp_time

        print("Compiled:", gpt_compiled)
        print("Correct:", gpt_correct)
        print(
            "Generation time:",
            round(gpt_generation_time, 2),
            "sec"
        )

        if gpt_speedup:
            print(
                "C++ speedup:",
                round(gpt_speedup, 2),
                "x"
            )

        # =====================================================
        # MODEL 2 — QWEN CODER
        # =====================================================

        print("\nQwen2.5-Coder-7B")

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(qwen_model.device)

        start = time.perf_counter()

        with torch.no_grad():

            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=1000,

                # Greedy decoding for repeatability
                do_sample=False
            )

        qwen_generation_time = (
            time.perf_counter() - start
        )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        qwen_code = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        qwen_code = (
            qwen_code
            .replace("```cpp", "")
            .replace("```c++", "")
            .replace("```", "")
            .strip()
        )

        qwen_cpp_file = f"qwen_{challenge_index}.cpp"
        qwen_executable = f"qwen_{challenge_index}"

        with open(qwen_cpp_file, "w") as file:
            file.write(qwen_code)

        compile_result = subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                qwen_cpp_file,
                "-o",
                qwen_executable
            ],
            capture_output=True,
            text=True
        )

        qwen_compiled = (
            compile_result.returncode == 0
        )

        qwen_correct = False
        qwen_speedup = None

        if qwen_compiled:

            start = time.perf_counter()

            execution = subprocess.run(
                [f"./{qwen_executable}"],
                capture_output=True,
                text=True
            )

            cpp_time = time.perf_counter() - start

            qwen_correct = (
                execution.stdout.strip()
                == expected_output
            )

            if qwen_correct:
                qwen_speedup = python_time / cpp_time

        print("Compiled:", qwen_compiled)
        print("Correct:", qwen_correct)

        print(
            "Generation time:",
            round(qwen_generation_time, 2),
            "sec"
        )

        if qwen_speedup:
            print(
                "C++ speedup:",
                round(qwen_speedup, 2),
                "x"
            )

        # -----------------------------------------------------
        # Save result
        # -----------------------------------------------------

        results.append({
            "challenge": challenge["name"],

            "gpt_compiled": gpt_compiled,
            "gpt_correct": gpt_correct,
            "gpt_generation": gpt_generation_time,

            "qwen_compiled": qwen_compiled,
            "qwen_correct": qwen_correct,
            "qwen_generation": qwen_generation_time,
        })

    # =========================================================
    # Final Evaluation
    # =========================================================

    print("\n\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    total = len(results)

    gpt_compile_count = sum(
        result["gpt_compiled"]
        for result in results
    )

    gpt_correct_count = sum(
        result["gpt_correct"]
        for result in results
    )

    qwen_compile_count = sum(
        result["qwen_compiled"]
        for result in results
    )

    qwen_correct_count = sum(
        result["qwen_correct"]
        for result in results
    )

    print("\nGPT-5.1")

    print(
        "Compile Rate:",
        f"{gpt_compile_count}/{total}"
    )

    print(
        "Correctness Rate:",
        f"{gpt_correct_count}/{total}"
    )

    print("\nQwen2.5-Coder-7B")

    print(
        "Compile Rate:",
        f"{qwen_compile_count}/{total}"
    )

    print(
        "Correctness Rate:",
        f"{qwen_correct_count}/{total}"
    )


if __name__ == "__main__":
    main()