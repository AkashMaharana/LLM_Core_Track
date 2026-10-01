import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def main():

    # ---------------------------------------------------------
    # 1. Check GPU
    # ---------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this QLoRA exercise."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    # ---------------------------------------------------------
    # 2. Model
    # ---------------------------------------------------------

    model_name = (
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    # ---------------------------------------------------------
    # 3. Decide computation precision
    # ---------------------------------------------------------

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print(
        "Compute dtype:",
        compute_dtype
    )

    # ---------------------------------------------------------
    # 4. Configure 4-bit quantization
    # ---------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # ---------------------------------------------------------
    # 5. Load tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # ---------------------------------------------------------
    # 6. Load Qwen in 4-bit
    # ---------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    print(
        "\nBase model loaded."
    )

    print(
        "Memory footprint:",
        round(
            model.get_memory_footprint()
            / 1024**3,
            2,
        ),
        "GB",
    )

    # ---------------------------------------------------------
    # 7. Prepare quantized model for training
    # ---------------------------------------------------------

    model = prepare_model_for_kbit_training(
        model
    )

    # ---------------------------------------------------------
    # 8. Configure LoRA
    # ---------------------------------------------------------

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    # ---------------------------------------------------------
    # 9. Attach LoRA adapters
    # ---------------------------------------------------------

    model = get_peft_model(
        model,
        lora_config,
    )

    # ---------------------------------------------------------
    # 10. Show trainable parameters
    # ---------------------------------------------------------

    print(
        "\nTRAINABLE PARAMETERS"
    )

    print(
        "=" * 70
    )

    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # 11. Show a few trainable parameter names
    # ---------------------------------------------------------

    print(
        "\nSAMPLE TRAINABLE PARAMETERS"
    )

    print(
        "=" * 70
    )

    count = 0

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            print(
                name,
                parameter.shape,
            )

            count += 1

            if count == 5:
                break


if __name__ == "__main__":
    main()