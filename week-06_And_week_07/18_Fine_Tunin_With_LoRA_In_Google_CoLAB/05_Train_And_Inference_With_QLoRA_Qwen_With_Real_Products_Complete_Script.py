import json

import torch

from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trl import (
    SFTConfig,
    SFTTrainer,
)

# !pip install -U transformers peft accelerate bitsandbytes>=0.46.1

# =============================================================
# CONFIGURATION
# =============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "amazon_price_train.jsonl"

VALIDATION_FILE = (
    "amazon_price_validation.jsonl"
)

OUTPUT_DIR = (
    "./qwen-price-training"
)

FINAL_ADAPTER_DIR = (
    "./qwen-price-lora-final"
)

METRICS_FILE = (
    "qwen-price-training-metrics.json"
)


# -------------------------------------------------------------
# Training settings
# -------------------------------------------------------------

NUM_EPOCHS = 3

# Use only one physical example at a time.
# This keeps GPU memory usage conservative on T4.
TRAIN_BATCH_SIZE = 1

EVAL_BATCH_SIZE = 1

# 1 physical example × 4 accumulation steps
# = effective batch size 4
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-4

MAX_LENGTH = 256


# -------------------------------------------------------------
# LoRA settings
# -------------------------------------------------------------

LORA_RANK = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05


# =============================================================
# BUILD CHAT TRAINING EXAMPLE
# =============================================================

def build_messages(example):

    price = example["price"]

    return {
        "messages": [
            {
                "role": "system",
                "content":
                    "You estimate product prices. "
                    "Reply only with the price "
                    "in US dollars.",
            },
            {
                "role": "user",
                "content":
                    example["description"],
            },
            {
                "role": "assistant",
                "content":
                    f"${price:.2f}",
            },
        ]
    }


# =============================================================
# LOAD DATASETS
# =============================================================

def load_training_datasets():

    train_dataset = load_dataset(
        "json",
        data_files=TRAIN_FILE,
        split="train",
    )

    validation_dataset = load_dataset(
        "json",
        data_files=VALIDATION_FILE,
        split="train",
    )

    print(
        "Raw training examples:",
        len(train_dataset)
    )

    print(
        "Raw validation examples:",
        len(validation_dataset)
    )

    # ---------------------------------------------------------
    # Convert:
    #
    # description + price
    #
    # into:
    #
    # system/user/assistant messages
    # ---------------------------------------------------------

    train_dataset = train_dataset.map(
        build_messages,
        remove_columns=(
            train_dataset.column_names
        ),
    )

    validation_dataset = (
        validation_dataset.map(
            build_messages,
            remove_columns=(
                validation_dataset.column_names
            ),
        )
    )

    return (
        train_dataset,
        validation_dataset,
    )


# =============================================================
# TOKENIZER
# =============================================================

def load_tokenizer():

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # ---------------------------------------------------------
    # A pad token is required when batching examples.
    # ---------------------------------------------------------

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    return tokenizer


# =============================================================
# LOAD 4-BIT MODEL
# =============================================================

def load_quantized_model():

    # ---------------------------------------------------------
    # IMPORTANT CHANGE
    #
    # Previous version:
    #
    # bnb_4bit_compute_dtype=torch.float16
    #
    # Trainer:
    #
    # fp16=True
    #
    # That activates AMP + GradScaler.
    #
    # Our T4/PyTorch environment then encounters an internal
    # BF16 gradient and GradScaler crashes.
    #
    # ---------------------------------------------------------
    #
    # New configuration:
    #
    # Weight storage:
    #
    #     4-bit NF4
    #
    # Compute:
    #
    #     FP32
    #
    # Mixed precision:
    #
    #     disabled
    #
    # ---------------------------------------------------------

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=(
                torch.float32
            ),

            bnb_4bit_use_double_quant=True,
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,

            quantization_config=(
                quantization_config
            ),

            # One GPU:
            # put the model explicitly on GPU 0.
            device_map={
                "": 0
            },

            # Non-quantized floating modules also use FP32.
            dtype=torch.float32,
        )
    )

    # ---------------------------------------------------------
    # KV cache is useful during generation but not training.
    # ---------------------------------------------------------

    model.config.use_cache = False

    # ---------------------------------------------------------
    # Prepare quantized model for PEFT training.
    # ---------------------------------------------------------

    model = (
        prepare_model_for_kbit_training(
            model
        )
    )

    return model


# =============================================================
# ATTACH LoRA
# =============================================================

def add_lora(model):

    lora_config = LoraConfig(

        r=LORA_RANK,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        bias="none",

        task_type="CAUSAL_LM",

        # QLoRA-style targeting:
        # add LoRA adapters to applicable linear layers.
        target_modules="all-linear",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    return model


# =============================================================
# FORCE TRAINABLE ADAPTERS TO FP32
# =============================================================

def cast_trainable_parameters_to_fp32(
    model
):

    parameter_count = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            parameter.data = (
                parameter.data.float()
            )

            parameter_count += (
                parameter.numel()
            )

    print(
        "Trainable parameters forced to FP32:",
        f"{parameter_count:,}",
    )


# =============================================================
# DTYPE AUDIT
# =============================================================

def print_trainable_dtypes(model):

    dtype_counts = {}

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        dtype_name = str(
            parameter.dtype
        )

        dtype_counts[
            dtype_name
        ] = (
            dtype_counts.get(
                dtype_name,
                0,
            )
            + parameter.numel()
        )

    print()

    print(
        "TRAINABLE PARAMETER DTYPES"
    )

    print(
        "=" * 70
    )

    for (
        dtype_name,
        count,
    ) in dtype_counts.items():

        print(
            dtype_name,
            ":",
            f"{count:,}",
        )


# =============================================================
# CREATE TRAINING CONFIGURATION
# =============================================================

def create_training_config():

    return SFTConfig(

        # -----------------------------------------------------
        # Output/checkpoints
        # -----------------------------------------------------

        output_dir=OUTPUT_DIR,

        # -----------------------------------------------------
        # Epochs
        # -----------------------------------------------------

        num_train_epochs=(
            NUM_EPOCHS
        ),

        # -----------------------------------------------------
        # Physical batch sizes
        # -----------------------------------------------------

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        # -----------------------------------------------------
        # Effective batch:
        #
        # 1 × 4 = 4
        # -----------------------------------------------------

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # -----------------------------------------------------
        # Learning rate
        # -----------------------------------------------------

        learning_rate=(
            LEARNING_RATE
        ),

        # -----------------------------------------------------
        # CRITICAL FIX
        #
        # Disable BOTH AMP modes.
        #
        # Therefore:
        #
        # no autocast
        # no FP16 GradScaler
        # no BF16 mixed precision
        # -----------------------------------------------------

        fp16=False,

        bf16=False,

        # -----------------------------------------------------
        # Sequence length
        # -----------------------------------------------------

        max_length=(
            MAX_LENGTH
        ),

        # -----------------------------------------------------
        # Save memory by recomputing some activations.
        # -----------------------------------------------------

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        use_cache=False,

        # -----------------------------------------------------
        # Train only against assistant output.
        # -----------------------------------------------------

        assistant_only_loss=True,

        # -----------------------------------------------------
        # Keep data flow simple for this exercise.
        # -----------------------------------------------------

        packing=False,

        # -----------------------------------------------------
        # Use standard AdamW.
        #
        # Avoid introducing another fused or 8-bit optimizer
        # while debugging dtype compatibility.
        # -----------------------------------------------------

        optim="adamw_torch",

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        logging_steps=10,

        logging_first_step=True,

        # -----------------------------------------------------
        # Validation every epoch
        # -----------------------------------------------------

        eval_strategy="epoch",

        # -----------------------------------------------------
        # Checkpoint every epoch
        # -----------------------------------------------------

        save_strategy="epoch",

        save_total_limit=2,

        # -----------------------------------------------------
        # No external logger
        # -----------------------------------------------------

        report_to="none",

        seed=42,
    )


# =============================================================
# SAVE METRICS
# =============================================================

def save_metrics(
    initial_eval,
    training_metrics,
    final_eval,
):

    metrics = {
        "initial_validation":
            initial_eval,

        "training":
            training_metrics,

        "final_validation":
            final_eval,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )


# =============================================================
# MAIN
# =============================================================

def main():

    # ---------------------------------------------------------
    # 1. GPU validation
    # ---------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    gpu_name = (
        torch.cuda.get_device_name(0)
    )

    capability = (
        torch.cuda.get_device_capability(0)
    )

    print(
        "GPU:",
        gpu_name
    )

    print(
        "CUDA capability:",
        capability
    )

    # ---------------------------------------------------------
    # 2. Experiment settings
    # ---------------------------------------------------------

    effective_batch_size = (
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "Effective batch size:",
        effective_batch_size
    )

    print(
        "Epochs:",
        NUM_EPOCHS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Mixed precision:",
        "DISABLED"
    )

    print(
        "4-bit compute dtype:",
        torch.float32
    )

    # ---------------------------------------------------------
    # 3. Load dataset
    # ---------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    (
        train_dataset,
        validation_dataset,
    ) = load_training_datasets()

    print()

    print(
        "FIRST TRAINING EXAMPLE"
    )

    print(
        "=" * 70
    )

    print(
        train_dataset[0]
    )

    # ---------------------------------------------------------
    # 4. Tokenizer
    # ---------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        load_tokenizer()
    )

    # ---------------------------------------------------------
    # 5. Quantized base model
    # ---------------------------------------------------------

    print(
        "\nLoading Qwen in 4-bit..."
    )

    model = (
        load_quantized_model()
    )

    print(
        "Base model memory footprint:",
        round(
            model.get_memory_footprint()
            / 1024**3,
            2,
        ),
        "GB",
    )

    # ---------------------------------------------------------
    # 6. LoRA
    # ---------------------------------------------------------

    print(
        "\nAttaching LoRA..."
    )

    model = add_lora(
        model
    )

    print()

    print(
        "TRAINABLE PARAMETERS"
    )

    print(
        "=" * 70
    )

    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # 7. Ensure adapters are FP32
    # ---------------------------------------------------------

    cast_trainable_parameters_to_fp32(
        model
    )

    print_trainable_dtypes(
        model
    )

    # ---------------------------------------------------------
    # 8. Training configuration
    # ---------------------------------------------------------

    training_config = (
        create_training_config()
    )

    # ---------------------------------------------------------
    # 9. Trainer
    # ---------------------------------------------------------

    trainer = SFTTrainer(

        model=model,

        args=training_config,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,
    )

    # ---------------------------------------------------------
    # 10. Initial validation
    # ---------------------------------------------------------

    print()

    print(
        "INITIAL VALIDATION"
    )

    print(
        "=" * 70
    )

    initial_eval = (
        trainer.evaluate()
    )

    print(
        "Initial eval loss:",
        initial_eval.get(
            "eval_loss"
        )
    )

    # ---------------------------------------------------------
    # 11. Training
    # ---------------------------------------------------------

    print()

    print(
        "STARTING QLORA TRAINING"
    )

    print(
        "=" * 70
    )

    train_result = (
        trainer.train()
    )

    # ---------------------------------------------------------
    # 12. Final validation
    # ---------------------------------------------------------

    print()

    print(
        "FINAL VALIDATION"
    )

    print(
        "=" * 70
    )

    final_eval = (
        trainer.evaluate()
    )

    print(
        "Final eval loss:",
        final_eval.get(
            "eval_loss"
        )
    )

    # ---------------------------------------------------------
    # 13. Save adapter
    # ---------------------------------------------------------

    print(
        "\nSaving LoRA adapter..."
    )

    model.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    tokenizer.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    # ---------------------------------------------------------
    # 14. Save metrics
    # ---------------------------------------------------------

    save_metrics(
        initial_eval,
        train_result.metrics,
        final_eval,
    )

    # ---------------------------------------------------------
    # 15. Summary
    # ---------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Initial eval loss:",
        initial_eval.get(
            "eval_loss"
        )
    )

    print(
        "Final eval loss:",
        final_eval.get(
            "eval_loss"
        )
    )

    print(
        "Training loss:",
        train_result.metrics.get(
            "train_loss"
        )
    )

    print(
        "Adapter saved to:",
        FINAL_ADAPTER_DIR
    )

    print(
        "Metrics saved to:",
        METRICS_FILE
    )


if __name__ == "__main__":
    main()