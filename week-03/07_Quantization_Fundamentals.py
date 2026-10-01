# !pip install -q transformers accelerate bitsandbytes
# This script demonstrates the memory footprint of a model in FP16 vs 4-bit quantization. 
# It needs to be executed in Google Colab Envirobnment with GPU enabled. 
# The script uses the Qwen2.5-1.5B-Instruct model for demonstration.

# Colab Link : https://colab.research.google.com/drive/10tS4grOBVTREG3esPoPhTU20Gul-3DGF?usp=sharing

import gc

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


def main():

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # ---------------------------------------------------------
    # GPU information
    # ---------------------------------------------------------

    print("GPU available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    messages = [
        {
            "role": "user",
            "content": "Explain Kafka in one sentence."
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # =========================================================
    # PART 1 — FP16 MODEL
    # =========================================================

    print("\n" + "=" * 60)
    print("FP16 MODEL")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,

        # Each supported weight uses 16 bits.
        torch_dtype=torch.float16,

        # Accelerate decides where the model should live.
        device_map="auto"
    )

    print(
        "\nModel memory footprint:",
        round(
            model.get_memory_footprint() / 1024**3,
            2
        ),
        "GB"
    )

    devices = {
        str(parameter.device)
        for parameter in model.parameters()
    }

    print("Model devices:", devices)

    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            2
        ),
        "GB"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(next(model.parameters()).device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=60,

            # Greedy decoding:
            # always choose highest-scoring next token.
            do_sample=False
        )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("\nFP16 RESPONSE:")
    print(response)

    # ---------------------------------------------------------
    # Remove FP16 model before loading 4-bit model
    # ---------------------------------------------------------

    del model
    del inputs
    del output

    gc.collect()

    torch.cuda.empty_cache()

    print(
        "\nGPU memory after cleanup:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            2
        ),
        "GB"
    )

    # =========================================================
    # PART 2 — 4-BIT QUANTIZED MODEL
    # =========================================================

    print("\n" + "=" * 60)
    print("4-BIT QUANTIZED MODEL")
    print("=" * 60)

    quantization_config = BitsAndBytesConfig(

        # Store supported model weights using 4-bit representation.
        load_in_4bit=True,

        # During mathematical computation use FP16.
        bnb_4bit_compute_dtype=torch.float16,

        # NF4 is a 4-bit format designed for neural-network weights.
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print(
        "\nModel memory footprint:",
        round(
            model.get_memory_footprint() / 1024**3,
            2
        ),
        "GB"
    )

    devices = {
        str(parameter.device)
        for parameter in model.parameters()
    }

    print("Model devices:", devices)

    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            2
        ),
        "GB"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(next(model.parameters()).device)

    # Reset measurement so we can observe generation memory.
    torch.cuda.reset_peak_memory_stats()

    memory_before = torch.cuda.memory_allocated()

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False
        )

    peak_memory = torch.cuda.max_memory_allocated()

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("\n4-BIT RESPONSE:")
    print(response)

    print(
        "\nAdditional peak memory during generation:",
        round(
            (peak_memory - memory_before) / 1024**2,
            2
        ),
        "MB"
    )

    # ---------------------------------------------------------
    # Final conceptual summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        """
Same model:
    Qwen2.5-1.5B-Instruct

Same number of parameters:
    ~1.5 billion

What changed:

FP16
    16 bits per supported weight
    Higher memory usage
    Higher numerical precision

4-bit
    4 bits per supported weight
    Much lower memory usage
    Approximate representation

Important:
    Quantization DOES NOT reduce parameter count.

    1.5B model remains a 1.5B model.

    It only changes how compactly the weights
    are represented in memory.
"""
    )


if __name__ == "__main__":
    main()