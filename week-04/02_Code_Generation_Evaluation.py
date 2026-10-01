from openai import OpenAI
from dotenv import load_dotenv

import statistics
import subprocess
import sys
import time

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"


def main():

    client = OpenAI(
        base_url=endpoint
    )

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
    for i in range(5000)
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

    results = []

    for index, challenge in enumerate(challenges):

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
        # Generate C++
        # -----------------------------------------------------

        generation_start = time.perf_counter()

        response = client.responses.create(
            model=deployment_name,
            input=prompt
        )

        generation_time = (
            time.perf_counter() - generation_start
        )

        cpp_code = response.output_text.strip()

        cpp_code = cpp_code.replace("```cpp", "")
        cpp_code = cpp_code.replace("```c++", "")
        cpp_code = cpp_code.replace("```", "")
        cpp_code = cpp_code.strip()

        python_file = f"challenge_{index}.py"
        cpp_file = f"challenge_{index}.cpp"
        executable = f"challenge_{index}"

        with open(python_file, "w") as file:
            file.write(python_code)

        with open(cpp_file, "w") as file:
            file.write(cpp_code)

        # -----------------------------------------------------
        # Compile C++
        # -----------------------------------------------------

        compile_result = subprocess.run(
            [
                "clang++",
                "-O3",
                "-std=c++17",
                cpp_file,
                "-o",
                executable
            ],
            capture_output=True,
            text=True
        )

        compiled = compile_result.returncode == 0

        if not compiled:

            print("Compilation: FAILED")
            print(compile_result.stderr)

            results.append({
                "name": challenge["name"],
                "compiled": False,
                "correct": False,
                "generation_time": generation_time,
                "python_time": None,
                "cpp_time": None,
                "speedup": None
            })

            continue

        print("Compilation: PASSED")

        # -----------------------------------------------------
        # Run once for correctness
        # -----------------------------------------------------

        python_result = subprocess.run(
            [sys.executable, python_file],
            capture_output=True,
            text=True
        )

        cpp_result = subprocess.run(
            [f"./{executable}"],
            capture_output=True,
            text=True
        )

        python_output = python_result.stdout.strip()
        cpp_output = cpp_result.stdout.strip()

        correct = python_output == cpp_output

        print("Python output:", python_output)
        print("C++ output:   ", cpp_output)
        print("Correct:      ", correct)

        # -----------------------------------------------------
        # Performance only matters when output is correct
        # -----------------------------------------------------

        python_time = None
        cpp_time = None
        speedup = None

        if correct:

            python_times = []
            cpp_times = []

            # Run three times to reduce measurement noise.
            for _ in range(3):

                start = time.perf_counter()

                subprocess.run(
                    [sys.executable, python_file],
                    capture_output=True
                )

                python_times.append(
                    time.perf_counter() - start
                )

                start = time.perf_counter()

                subprocess.run(
                    [f"./{executable}"],
                    capture_output=True
                )

                cpp_times.append(
                    time.perf_counter() - start
                )

            python_time = statistics.median(
                python_times
            )

            cpp_time = statistics.median(
                cpp_times
            )

            speedup = python_time / cpp_time

            print(
                "Python median:",
                round(python_time, 4),
                "sec"
            )

            print(
                "C++ median:",
                round(cpp_time, 4),
                "sec"
            )

            print(
                "Speedup:",
                round(speedup, 2),
                "x"
            )

        results.append({
            "name": challenge["name"],
            "compiled": compiled,
            "correct": correct,
            "generation_time": generation_time,
            "python_time": python_time,
            "cpp_time": cpp_time,
            "speedup": speedup
        })

    # =========================================================
    # Evaluation Summary
    # =========================================================

    print("\n\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    compiled_count = sum(
        result["compiled"]
        for result in results
    )

    correct_count = sum(
        result["correct"]
        for result in results
    )

    total = len(results)

    print(
        f"Compile Rate: "
        f"{compiled_count}/{total} "
        f"({compiled_count / total * 100:.1f}%)"
    )

    print(
        f"Correctness Rate: "
        f"{correct_count}/{total} "
        f"({correct_count / total * 100:.1f}%)"
    )

    print("\nIndividual Results")

    for result in results:

        print("\n", result["name"])

        print(
            "  Compiled:",
            result["compiled"]
        )

        print(
            "  Correct:",
            result["correct"]
        )

        print(
            "  Generation latency:",
            round(
                result["generation_time"],
                2
            ),
            "sec"
        )

        if result["speedup"] is not None:

            print(
                "  C++ speedup:",
                round(
                    result["speedup"],
                    2
                ),
                "x"
            )


if __name__ == "__main__":
    main()