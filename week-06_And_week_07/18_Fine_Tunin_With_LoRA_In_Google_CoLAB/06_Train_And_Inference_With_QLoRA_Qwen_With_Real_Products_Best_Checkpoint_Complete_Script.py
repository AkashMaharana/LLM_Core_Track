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


# =============================================================
# CONFIGURATION
# =============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "amazon_price_train.jsonl"
VALIDATION_FILE = "amazon_price_validation.jsonl"

OUTPUT_DIR = "./qwen-price-best-training"

FINAL_ADAPTER_DIR = "./qwen-price-lora-best"

METRICS_FILE = "qwen-price-best-training-metrics.json"


NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-4

MAX_LENGTH = 256


LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# =============================================================
# DATASET FORMAT
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

    train_dataset = train_dataset.map(
        build_messages,
        remove_columns=train_dataset.column_names,
    )

    validation_dataset = validation_dataset.map(
        build_messages,
        remove_columns=validation_dataset.column_names,
    )

    return (
        train_dataset,
        validation_dataset,
    )


# =============================================================
# TOKENIZER
# =============================================================

def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


# =============================================================
# LOAD 4-BIT BASE MODEL
# =============================================================

def load_quantized_model():

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",

        # Stable configuration for our T4 environment
        bnb_4bit_compute_dtype=torch.float32,

        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,

        quantization_config=quantization_config,

        device_map={
            "": 0
        },

        dtype=torch.float32,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model
    )

    return model


# =============================================================
# ADD LoRA
# =============================================================

def add_lora(model):

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    return model


# =============================================================
# KEEP TRAINABLE ADAPTERS IN FP32
# =============================================================

def cast_trainable_parameters_to_fp32(model):

    for parameter in model.parameters():

        if parameter.requires_grad:

            parameter.data = (
                parameter.data.float()
            )


# =============================================================
# TRAINING CONFIG
# =============================================================

def create_training_config():

    return SFTConfig(

        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        # Stable T4 configuration
        fp16=False,
        bf16=False,

        max_length=MAX_LENGTH,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        use_cache=False,

        assistant_only_loss=True,

        packing=False,

        optim="adamw_torch",

        logging_steps=10,
        logging_first_step=True,

        # -----------------------------------------------------
        # Evaluate after every epoch
        # -----------------------------------------------------

        eval_strategy="epoch",

        # -----------------------------------------------------
        # Save a checkpoint after every epoch.
        #
        # This matches eval_strategy, which is required when
        # load_best_model_at_end=True.
        # -----------------------------------------------------

        save_strategy="epoch",

        save_total_limit=3,

        # -----------------------------------------------------
        # NEW:
        #
        # At the end of training, automatically reload the
        # checkpoint with the best validation metric.
        # -----------------------------------------------------

        load_best_model_at_end=True,

        # Compare models using validation loss.
        metric_for_best_model="eval_loss",

        # Lower validation loss is better.
        greater_is_better=False,

        report_to="none",

        seed=42,
    )


# =============================================================
# SAVE METRICS
# =============================================================

def save_metrics(
    initial_eval,
    train_result,
    final_eval,
    best_checkpoint,
    best_metric,
):

    metrics = {

        "initial_validation":
            initial_eval,

        "training":
            train_result.metrics,

        "best_checkpoint":
            best_checkpoint,

        "best_validation_loss":
            best_metric,

        "final_loaded_model_validation":
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

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "\nLoading datasets..."
    )

    (
        train_dataset,
        validation_dataset,
    ) = load_training_datasets()

    print(
        "Training examples:",
        len(train_dataset)
    )

    print(
        "Validation examples:",
        len(validation_dataset)
    )

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = load_tokenizer()

    print(
        "\nLoading Qwen in 4-bit..."
    )

    model = load_quantized_model()

    print(
        "\nAttaching LoRA..."
    )

    model = add_lora(
        model
    )

    cast_trainable_parameters_to_fp32(
        model
    )

    print(
        "\nTRAINABLE PARAMETERS"
    )

    model.print_trainable_parameters()

    training_config = (
        create_training_config()
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
        "\nINITIAL VALIDATION"
    )

    initial_eval = trainer.evaluate()

    print(
        "Initial validation loss:",
        initial_eval["eval_loss"]
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print(
        "\nSTARTING TRAINING"
    )

    train_result = trainer.train()

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Because:
    #
    # load_best_model_at_end=True
    #
    # trainer.model is now the BEST checkpoint,
    # not necessarily the final epoch.
    # ---------------------------------------------------------

    best_checkpoint = (
        trainer.state.best_model_checkpoint
    )

    best_metric = (
        trainer.state.best_metric
    )

    print(
        "\nBEST CHECKPOINT"
    )

    print(
        "Checkpoint:",
        best_checkpoint
    )

    print(
        "Validation loss:",
        best_metric
    )

    # ---------------------------------------------------------
    # Evaluate the model that Trainer reloaded.
    # ---------------------------------------------------------

    final_eval = trainer.evaluate()

    print(
        "Loaded best model validation loss:",
        final_eval["eval_loss"]
    )

    # ---------------------------------------------------------
    # Save the BEST adapter.
    # ---------------------------------------------------------

    trainer.model.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    tokenizer.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    save_metrics(
        initial_eval,
        train_result,
        final_eval,
        best_checkpoint,
        best_metric,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Best checkpoint:",
        best_checkpoint
    )

    print(
        "Best validation loss:",
        best_metric
    )

    print(
        "Best adapter saved to:",
        FINAL_ADAPTER_DIR
    )


if __name__ == "__main__":
    main()