from openai import OpenAI
from dotenv import load_dotenv

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

    python_code = """
total = 0

for i in range(10_000_000):
    total += (i * i) % 97

print(total)
"""

    # ---------------------------------------------------------
    # 1. Ask GPT-5.1 to convert Python → optimized C++
    # ---------------------------------------------------------

    prompt = f"""
Convert the following Python program into optimized C++.

Requirements:
- Preserve exactly the same behavior.
- The output must be identical to the Python program.
- Use efficient C++.
- Use 64-bit integers where necessary.
- Return only valid C++ code.
- Do not include Markdown code fences or explanations.

Python code:

{python_code}
"""

    response = client.responses.create(
        model=deployment_name,
        input=prompt
    )

    cpp_code = response.output_text.strip()

    # Safety cleanup in case the model still returns fences.
    cpp_code = cpp_code.replace("```cpp", "")
    cpp_code = cpp_code.replace("```c++", "")
    cpp_code = cpp_code.replace("```", "")
    cpp_code = cpp_code.strip()

    # ---------------------------------------------------------
    # 2. Save both programs
    # ---------------------------------------------------------

    with open("original.py", "w") as file:
        file.write(python_code)

    with open("generated.cpp", "w") as file:
        file.write(cpp_code)

    print("\nGENERATED C++")
    print("=" * 60)
    print(cpp_code)

    # ---------------------------------------------------------
    # 3. Compile generated C++
    # ---------------------------------------------------------

    print("\nCOMPILING C++...")

    compile_result = subprocess.run(
        [
            "clang++",
            "-O3",
            "-std=c++17",
            "generated.cpp",
            "-o",
            "generated"
        ],
        capture_output=True,
        text=True
    )

    if compile_result.returncode != 0:
        print("\nCOMPILATION FAILED")
        print(compile_result.stderr)
        return

    print("Compilation successful.")

    # ---------------------------------------------------------
    # 4. Run Python program
    # ---------------------------------------------------------

    start = time.perf_counter()

    python_result = subprocess.run(
        [
            sys.executable,
            "original.py"
        ],
        capture_output=True,
        text=True
    )

    python_time = time.perf_counter() - start

    # ---------------------------------------------------------
    # 5. Run generated C++ program
    # ---------------------------------------------------------

    start = time.perf_counter()

    cpp_result = subprocess.run(
        ["./generated"],
        capture_output=True,
        text=True
    )

    cpp_time = time.perf_counter() - start

    # ---------------------------------------------------------
    # 6. Compare correctness
    # ---------------------------------------------------------

    python_output = python_result.stdout.strip()
    cpp_output = cpp_result.stdout.strip()

    print("\nRESULTS")
    print("=" * 60)

    print("Python output:", python_output)
    print("C++ output:   ", cpp_output)

    correct = python_output == cpp_output

    print("\nCorrect:", correct)

    # ---------------------------------------------------------
    # 7. Compare performance only if correct
    # ---------------------------------------------------------

    if correct:

        print(
            "\nPython execution time:",
            round(python_time, 4),
            "seconds"
        )

        print(
            "C++ execution time:",
            round(cpp_time, 4),
            "seconds"
        )

        if cpp_time > 0:

            speedup = python_time / cpp_time

            print(
                "Speedup:",
                round(speedup, 2),
                "x"
            )


if __name__ == "__main__":
    main()