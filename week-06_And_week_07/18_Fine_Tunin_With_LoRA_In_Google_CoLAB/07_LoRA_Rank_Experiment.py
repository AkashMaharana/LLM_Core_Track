import csv
import gc
import json
import os

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


# =============================================================
# CONFIGURATION
# =============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "amazon_price_train.jsonl"

VALIDATION_FILE = (
    "amazon_price_validation.jsonl"
)


EXPERIMENT_ROOT = (
    "./lora_rank_experiments"
)

CSV_RESULT_FILE = (
    "lora_rank_experiment_results.csv"
)

JSON_RESULT_FILE = (
    "lora_rank_experiment_results.json"
)


# -------------------------------------------------------------
# Controlled rank experiments
#
# alpha / rank = 2 for every experiment
# -------------------------------------------------------------

LORA_EXPERIMENTS = [
    {
        "rank": 4,
        "alpha": 8,
    },
    {
        "rank": 16,
        "alpha": 32,
    },
    {
        "rank": 64,
        "alpha": 128,
    },
]


# -------------------------------------------------------------
# Everything below stays constant.
# -------------------------------------------------------------

LORA_DROPOUT = 0.05

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 1

EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-4

MAX_LENGTH = 256

RANDOM_SEED = 42


# =============================================================
# DATASET
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

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    return tokenizer


# =============================================================
# LOAD FRESH 4-BIT BASE MODEL
# =============================================================

def load_base_model():

    quantization_config = (
        BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            # Stable configuration from our T4 experiment
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

            device_map={
                "": 0
            },

            dtype=torch.float32,
        )
    )

    model.config.use_cache = False

    model = (
        prepare_model_for_kbit_training(
            model
        )
    )

    return model


# =============================================================
# ADD LoRA
# =============================================================

def add_lora(
    model,
    rank,
    alpha,
):

    lora_config = LoraConfig(

        r=rank,

        lora_alpha=alpha,

        lora_dropout=(
            LORA_DROPOUT
        ),

        bias="none",

        task_type="CAUSAL_LM",

        # QLoRA-style:
        # add adapters to all linear layers
        target_modules="all-linear",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    return model


# =============================================================
# FORCE TRAINABLE PARAMETERS TO FP32
# =============================================================

def cast_trainable_parameters_to_fp32(
    model
):

    for parameter in model.parameters():

        if parameter.requires_grad:

            parameter.data = (
                parameter.data.float()
            )


# =============================================================
# COUNT TRAINABLE PARAMETERS
# =============================================================

def count_trainable_parameters(
    model
):

    trainable = 0

    total = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:

            trainable += (
                parameter.numel()
            )

    percentage = (
        trainable
        / total
        * 100
    )

    return (
        trainable,
        total,
        percentage,
    )


# =============================================================
# TRAINING CONFIG
# =============================================================

def create_training_config(
    output_dir
):

    return SFTConfig(

        output_dir=output_dir,

        num_train_epochs=(
            NUM_EPOCHS
        ),

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=(
            LEARNING_RATE
        ),

        # -----------------------------------------------------
        # Stable configuration for our T4 runtime
        # -----------------------------------------------------

        fp16=False,

        bf16=False,

        optim="adamw_torch",

        max_length=(
            MAX_LENGTH
        ),

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        use_cache=False,

        assistant_only_loss=True,

        packing=False,

        logging_steps=10,

        logging_first_step=True,

        # -----------------------------------------------------
        # Validation-based model selection
        # -----------------------------------------------------

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model=(
            "eval_loss"
        ),

        greater_is_better=False,

        save_total_limit=3,

        report_to="none",

        seed=RANDOM_SEED,
    )


# =============================================================
# CALCULATE SAVED DIRECTORY SIZE
# =============================================================

def get_directory_size_mb(
    directory
):

    total_bytes = 0

    for (
        root,
        directories,
        files,
    ) in os.walk(
        directory
    ):

        for filename in files:

            file_path = os.path.join(
                root,
                filename,
            )

            total_bytes += (
                os.path.getsize(
                    file_path
                )
            )

    return (
        total_bytes
        / 1024
        / 1024
    )


# =============================================================
# RELEASE GPU MEMORY
# =============================================================

def cleanup_gpu():

    gc.collect()

    torch.cuda.empty_cache()


# =============================================================
# RUN ONE RANK EXPERIMENT
# =============================================================

def run_experiment(
    rank,
    alpha,
    tokenizer,
    train_dataset,
    validation_dataset,
):

    print()

    print(
        "=" * 80
    )

    print(
        f"LoRA EXPERIMENT: "
        f"rank={rank}, "
        f"alpha={alpha}"
    )

    print(
        "=" * 80
    )

    experiment_dir = os.path.join(
        EXPERIMENT_ROOT,
        f"rank_{rank}",
    )

    checkpoint_dir = os.path.join(
        experiment_dir,
        "checkpoints",
    )

    adapter_dir = os.path.join(
        experiment_dir,
        "best_adapter",
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Each experiment starts from a completely fresh Qwen
    # base model.
    #
    # Otherwise rank=16 would inherit training from rank=4,
    # which would make the comparison invalid.
    # ---------------------------------------------------------

    print(
        "\nLoading fresh Qwen base model..."
    )

    model = load_base_model()

    print(
        "Attaching LoRA..."
    )

    model = add_lora(
        model,
        rank,
        alpha,
    )

    cast_trainable_parameters_to_fp32(
        model
    )

    (
        trainable_parameters,
        total_parameters,
        trainable_percentage,
    ) = count_trainable_parameters(
        model
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "Trainable percentage:",
        round(
            trainable_percentage,
            4,
        ),
        "%",
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    training_config = (
        create_training_config(
            checkpoint_dir
        )
    )

    trainer = SFTTrainer(

        model=model,

        args=training_config,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,
    )

    # ---------------------------------------------------------
    # Validation before training
    # ---------------------------------------------------------

    print(
        "\nInitial validation..."
    )

    initial_eval = (
        trainer.evaluate()
    )

    initial_validation_loss = (
        initial_eval[
            "eval_loss"
        ]
    )

    print(
        "Initial validation loss:",
        initial_validation_loss,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print(
        "\nTraining..."
    )

    train_result = (
        trainer.train()
    )

    # ---------------------------------------------------------
    # Trainer has reloaded the best validation checkpoint.
    # ---------------------------------------------------------

    best_checkpoint = (
        trainer.state.best_model_checkpoint
    )

    best_validation_loss = (
        trainer.state.best_metric
    )

    print(
        "\nBest checkpoint:",
        best_checkpoint,
    )

    print(
        "Best validation loss:",
        best_validation_loss,
    )

    # ---------------------------------------------------------
    # Validate the reloaded best adapter
    # ---------------------------------------------------------

    best_eval = (
        trainer.evaluate()
    )

    loaded_validation_loss = (
        best_eval[
            "eval_loss"
        ]
    )

    # ---------------------------------------------------------
    # Save best adapter
    # ---------------------------------------------------------

    trainer.model.save_pretrained(
        adapter_dir
    )

    tokenizer.save_pretrained(
        adapter_dir
    )

    adapter_size_mb = (
        get_directory_size_mb(
            adapter_dir
        )
    )

    # ---------------------------------------------------------
    # Training metrics
    # ---------------------------------------------------------

    training_loss = (
        train_result.metrics.get(
            "train_loss"
        )
    )

    training_runtime = (
        train_result.metrics.get(
            "train_runtime"
        )
    )

    result = {

        "rank":
            rank,

        "alpha":
            alpha,

        "alpha_to_rank_ratio":
            alpha / rank,

        "trainable_parameters":
            trainable_parameters,

        "total_parameters":
            total_parameters,

        "trainable_percentage":
            trainable_percentage,

        "initial_validation_loss":
            initial_validation_loss,

        "best_validation_loss":
            best_validation_loss,

        "loaded_best_validation_loss":
            loaded_validation_loss,

        "training_loss":
            training_loss,

        "training_runtime_seconds":
            training_runtime,

        "adapter_size_mb":
            adapter_size_mb,

        "best_checkpoint":
            best_checkpoint,

        "adapter_path":
            adapter_dir,
    }

    print()

    print(
        "EXPERIMENT RESULT"
    )

    print(
        "-" * 80
    )

    print(
        "Rank:",
        rank
    )

    print(
        "Alpha:",
        alpha
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "Best validation loss:",
        round(
            best_validation_loss,
            6,
        )
    )

    print(
        "Adapter size:",
        round(
            adapter_size_mb,
            2,
        ),
        "MB",
    )

    # ---------------------------------------------------------
    # Delete model/trainer before starting next experiment.
    # ---------------------------------------------------------

    del trainer
    del model

    cleanup_gpu()

    return result


# =============================================================
# SAVE RESULTS
# =============================================================

def save_results(
    results
):

    with open(
        JSON_RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    fieldnames = [
        "rank",
        "alpha",
        "alpha_to_rank_ratio",
        "trainable_parameters",
        "total_parameters",
        "trainable_percentage",
        "initial_validation_loss",
        "best_validation_loss",
        "loaded_best_validation_loss",
        "training_loss",
        "training_runtime_seconds",
        "adapter_size_mb",
        "best_checkpoint",
        "adapter_path",
    ]

    with open(
        CSV_RESULT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# =============================================================
# PRINT FINAL COMPARISON
# =============================================================

def print_comparison(
    results
):

    print()

    print(
        "=" * 100
    )

    print(
        "FINAL LoRA RANK COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Rank':<8}"
        f"{'Alpha':<10}"
        f"{'Trainable Params':<22}"
        f"{'Best Val Loss':<20}"
        f"{'Adapter MB':<15}"
    )

    print(
        "-" * 100
    )

    for result in results:

        print(
            f"{result['rank']:<8}"
            f"{result['alpha']:<10}"
            f"{result['trainable_parameters']:<22,}"
            f"{result['best_validation_loss']:<20.6f}"
            f"{result['adapter_size_mb']:<15.2f}"
        )

    # ---------------------------------------------------------
    # Select by validation loss ONLY.
    #
    # We are deliberately not touching the held-out test set.
    # ---------------------------------------------------------

    best_result = min(
        results,
        key=lambda item:
            item[
                "best_validation_loss"
            ],
    )

    print()

    print(
        "Lowest validation loss:"
    )

    print(
        "Rank:",
        best_result[
            "rank"
        ]
    )

    print(
        "Alpha:",
        best_result[
            "alpha"
        ]
    )

    print(
        "Validation loss:",
        round(
            best_result[
                "best_validation_loss"
            ],
            6,
        )
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        CSV_RESULT_FILE
    )

    print(
        JSON_RESULT_FILE
    )


# =============================================================
# MAIN
# =============================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA capability:",
        torch.cuda.get_device_capability(
            0
        )
    )

    print(
        "\nLoading shared datasets..."
    )

    (
        train_dataset,
        validation_dataset,
    ) = load_training_datasets()

    print(
        "Training examples:",
        len(
            train_dataset
        )
    )

    print(
        "Validation examples:",
        len(
            validation_dataset
        )
    )

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = load_tokenizer()

    os.makedirs(
        EXPERIMENT_ROOT,
        exist_ok=True,
    )

    results = []

    # ---------------------------------------------------------
    # Run:
    #
    # rank 4
    # rank 16
    # rank 64
    #
    # sequentially.
    # ---------------------------------------------------------

    for experiment in (
        LORA_EXPERIMENTS
    ):

        result = run_experiment(

            rank=experiment[
                "rank"
            ],

            alpha=experiment[
                "alpha"
            ],

            tokenizer=tokenizer,

            train_dataset=(
                train_dataset
            ),

            validation_dataset=(
                validation_dataset
            ),
        )

        results.append(
            result
        )

    save_results(
        results
    )

    print_comparison(
        results
    )


if __name__ == "__main__":
    main()