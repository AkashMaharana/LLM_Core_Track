# !pip install -q transformers accelerate bitsandbytes
# This script demonstrates the evaluation of a LLM (Qwen2.5-Coder-7B) in translating Python code to optimized C++ code.
# It needs to be executed in Google Colab Envirobnment with GPU enabled.

import subprocess

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


def main():

    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # ---------------------------------------------------------
    # Load model in 4-bit
    # ---------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    challenges = [
        {
            "name": "Numeric Loop",
            "code": """
total = 0

for i in range(1_000_000):
    total += (i * i) % 97

print(total)
"""
        },

        {
            "name": "Prime Count",
            "code": """
import math

count = 0

for number in range(2, 100_000):
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
text = "distributed systems and artificial intelligence " * 10000

count = 0

for word in text.split():
    if word.startswith("a"):
        count += 1

print(count)
"""
        }
    ]

    total_problems = len(challenges)

    pass_1_count = 0
    pass_5_count = 0

    # =========================================================
    # Evaluate each challenge
    # =========================================================

    for challenge_index, challenge in enumerate(
        challenges
    ):

        print("\n" + "=" * 70)
        print(challenge["name"])
        print("=" * 70)

        python_code = challenge["code"]

        # -----------------------------------------------------
        # Run Python reference implementation
        # -----------------------------------------------------

        python_file = (
            f"reference_{challenge_index}.py"
        )

        with open(python_file, "w") as file:
            file.write(python_code)

        python_result = subprocess.run(
            ["python", python_file],
            capture_output=True,
            text=True
        )

        expected_output = (
            python_result.stdout.strip()
        )

        print(
            "Expected output:",
            expected_output
        )

        # -----------------------------------------------------
        # Build prompt
        # -----------------------------------------------------

        prompt = f"""
Convert the following Python program into optimized C++17.

Requirements:
- Preserve exactly the same behavior.
- Produce exactly the same output.
- Use appropriate C++ data types.
- Avoid integer overflow.
- Return only valid C++ source code.
- Do not include Markdown fences.
- Do not include explanations.

Python:

{python_code}
"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(model.device)

        candidate_results = []

        # =====================================================
        # Generate five candidates
        # =====================================================

        for candidate_index in range(5):

            print(
                f"\nCandidate "
                f"{candidate_index + 1}"
            )

            # ---------------------------------------------
            # Sampling creates diversity
            # ---------------------------------------------

            with torch.no_grad():

                output = model.generate(
                    **inputs,
                    max_new_tokens=800,

                    # Sampling ON
                    do_sample=True,

                    # Moderate randomness
                    temperature=0.8,

                    # Ignore very unlikely tail tokens
                    top_p=0.9
                )

            generated_tokens = output[0][
                inputs["input_ids"].shape[1]:
            ]

            cpp_code = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            cpp_code = (
                cpp_code
                .replace("```cpp", "")
                .replace("```c++", "")
                .replace("```", "")
                .strip()
            )

            cpp_file = (
                f"problem_{challenge_index}_"
                f"candidate_{candidate_index}.cpp"
            )

            executable = (
                f"problem_{challenge_index}_"
                f"candidate_{candidate_index}"
            )

            with open(cpp_file, "w") as file:
                file.write(cpp_code)

            # ---------------------------------------------
            # Compile
            # ---------------------------------------------

            compile_result = subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    cpp_file,
                    "-o",
                    executable
                ],
                capture_output=True,
                text=True
            )

            compiled = (
                compile_result.returncode == 0
            )

            if not compiled:

                print("Compilation: FAIL")

                candidate_results.append(False)

                continue

            # ---------------------------------------------
            # Execute
            # ---------------------------------------------

            execution_result = subprocess.run(
                [f"./{executable}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            actual_output = (
                execution_result.stdout.strip()
            )

            correct = (
                actual_output
                == expected_output
            )

            print(
                "Compilation: PASS"
            )

            print(
                "Correct:",
                correct
            )

            candidate_results.append(
                correct
            )

        # =====================================================
        # pass@1 / pass@5
        # =====================================================

        pass_1 = candidate_results[0]

        pass_5 = any(
            candidate_results
        )

        if pass_1:
            pass_1_count += 1

        if pass_5:
            pass_5_count += 1

        print("\nRESULT")

        print(
            "Candidate correctness:",
            candidate_results
        )

        print(
            "pass@1:",
            pass_1
        )

        print(
            "pass@5:",
            pass_5
        )

    # =========================================================
    # Aggregate results
    # =========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        "pass@1:",
        f"{pass_1_count}/{total_problems}",
        f"({pass_1_count / total_problems * 100:.1f}%)"
    )

    print(
        "pass@5:",
        f"{pass_5_count}/{total_problems}",
        f"({pass_5_count / total_problems * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()