import csv
import json
import math
import os
import re

import torch

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# =============================================================
# CONFIGURATION
# =============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "./qwen-price-lora-final"

TEST_FILE = "amazon_price_test.jsonl"

CSV_OUTPUT_FILE = (
    "qwen_price_evaluation_results.csv"
)

JSON_OUTPUT_FILE = (
    "qwen_price_evaluation_results.json"
)

SUMMARY_FILE = (
    "qwen_price_evaluation_summary.json"
)


# -------------------------------------------------------------
# Maximum generated tokens.
#
# We only expect something like:
#
# $129.99
#
# so we do not need a long response.
# -------------------------------------------------------------

MAX_NEW_TOKENS = 16


# =============================================================
# LOAD TEST DATA
# =============================================================

def load_test_products():

    products = []

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            product = json.loads(
                line
            )

            products.append(
                product
            )

    return products


# =============================================================
# LOAD TOKENIZER + MODEL + ADAPTER
# =============================================================

def load_model():

    if not os.path.isdir(
        ADAPTER_PATH
    ):

        raise FileNotFoundError(
            f"LoRA adapter not found: "
            f"{ADAPTER_PATH}\n"
            f"Finish script 70 first."
        )

    # ---------------------------------------------------------
    # Keep inference configuration close to our final
    # training configuration.
    #
    # Base weights:
    #     4-bit NF4
    #
    # Compute:
    #     FP32
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

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # ---------------------------------------------------------
    # Load original Qwen.
    # ---------------------------------------------------------

    base_model = (
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

    # ---------------------------------------------------------
    # Attach the trained LoRA adapter.
    #
    # This creates:
    #
    # Base Qwen + trained LoRA
    # ---------------------------------------------------------

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()

    return (
        model,
        tokenizer,
    )


# =============================================================
# BUILD MODEL INPUT
# =============================================================

def prepare_inputs(
    tokenizer,
    model,
    description,
):

    messages = [
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
                description,
        },
    ]

    inputs = (
        tokenizer.apply_chat_template(
            messages,

            tokenize=True,

            add_generation_prompt=True,

            return_tensors="pt",

            return_dict=True,
        )
    )

    # ---------------------------------------------------------
    # Put input tensors on the same GPU as the model.
    # ---------------------------------------------------------

    input_device = (
        model.get_input_embeddings()
        .weight
        .device
    )

    inputs = inputs.to(
        input_device
    )

    return inputs


# =============================================================
# GENERATE RESPONSE
# =============================================================

def generate_response(
    model,
    tokenizer,
    description,
):

    inputs = prepare_inputs(
        tokenizer,
        model,
        description,
    )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,

            max_new_tokens=(
                MAX_NEW_TOKENS
            ),

            # Deterministic generation.
            do_sample=False,

            pad_token_id=(
                tokenizer.eos_token_id
            ),
        )

    # ---------------------------------------------------------
    # generate() returns:
    #
    # original prompt tokens
    # +
    # newly generated tokens
    #
    # We only want the new tokens.
    # ---------------------------------------------------------

    generated_tokens = (
        output[0][
            input_length:
        ]
    )

    response = tokenizer.decode(
        generated_tokens,

        skip_special_tokens=True,
    )

    return response.strip()


# =============================================================
# EXTRACT NUMERIC PRICE
# =============================================================

def extract_price(response):

    # ---------------------------------------------------------
    # Supports:
    #
    # $199
    # 199
    # $199.99
    # $1,299.99
    #
    # ---------------------------------------------------------

    match = re.search(
        r"\$?\s*(\d[\d,]*(?:\.\d+)?)",
        response,
    )

    if match is None:

        return None

    number_text = (
        match
        .group(1)
        .replace(
            ",",
            "",
        )
    )

    try:

        price = float(
            number_text
        )

    except ValueError:

        return None

    return max(
        0.0,
        price,
    )


# =============================================================
# BASE MODEL PREDICTION
# =============================================================

def predict_base(
    model,
    tokenizer,
    description,
):

    # ---------------------------------------------------------
    # Temporarily disable LoRA.
    #
    # The model now behaves as original Base Qwen.
    # ---------------------------------------------------------

    with model.disable_adapter():

        response = generate_response(
            model,
            tokenizer,
            description,
        )

    price = extract_price(
        response
    )

    return (
        price,
        response,
    )


# =============================================================
# FINE-TUNED MODEL PREDICTION
# =============================================================

def predict_finetuned(
    model,
    tokenizer,
    description,
):

    # ---------------------------------------------------------
    # LoRA is enabled normally.
    # ---------------------------------------------------------

    response = generate_response(
        model,
        tokenizer,
        description,
    )

    price = extract_price(
        response
    )

    return (
        price,
        response,
    )


# =============================================================
# MAE
# =============================================================

def calculate_mae(
    actual_values,
    predicted_values,
):

    total_error = 0.0

    for (
        actual,
        predicted,
    ) in zip(
        actual_values,
        predicted_values,
    ):

        total_error += abs(
            actual - predicted
        )

    return (
        total_error
        / len(actual_values)
    )


# =============================================================
# RMSLE
# =============================================================

def calculate_rmsle(
    actual_values,
    predicted_values,
):

    squared_log_errors = []

    for (
        actual,
        predicted,
    ) in zip(
        actual_values,
        predicted_values,
    ):

        log_actual = math.log1p(
            actual
        )

        log_predicted = math.log1p(
            max(
                0.0,
                predicted,
            )
        )

        error = (
            log_predicted
            - log_actual
        )

        squared_log_errors.append(
            error ** 2
        )

    mean_squared_log_error = (
        sum(
            squared_log_errors
        )
        / len(
            squared_log_errors
        )
    )

    return math.sqrt(
        mean_squared_log_error
    )


# =============================================================
# SAVE CSV
# =============================================================

def save_csv(results):

    fieldnames = [
        "product_id",

        "actual_price",

        "base_price",

        "fine_tuned_price",

        "base_absolute_error",

        "fine_tuned_absolute_error",

        "base_response",

        "fine_tuned_response",

        "description",
    ]

    with open(
        CSV_OUTPUT_FILE,
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
# SAVE JSON
# =============================================================

def save_json(results):

    with open(
        JSON_OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =============================================================
# MAIN
# =============================================================

def main():

    # ---------------------------------------------------------
    # 1. Check GPU
    # ---------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    # ---------------------------------------------------------
    # 2. Load untouched test dataset
    # ---------------------------------------------------------

    print(
        "\nLoading test products..."
    )

    test_products = (
        load_test_products()
    )

    print(
        "Test products:",
        len(
            test_products
        )
    )

    # ---------------------------------------------------------
    # 3. Load Qwen + LoRA
    # ---------------------------------------------------------

    print(
        "\nLoading Qwen + trained LoRA..."
    )

    (
        model,
        tokenizer,
    ) = load_model()

    print(
        "Model loaded."
    )

    # ---------------------------------------------------------
    # 4. Containers for metrics/results
    # ---------------------------------------------------------

    results = []

    actual_prices = []

    base_predictions = []

    fine_tuned_predictions = []

    base_parse_failures = 0

    fine_tuned_parse_failures = 0

    # ---------------------------------------------------------
    # 5. Evaluate every product
    # ---------------------------------------------------------

    for (
        index,
        product,
    ) in enumerate(
        test_products,
        start=1,
    ):

        description = (
            product[
                "description"
            ]
        )

        actual_price = float(
            product[
                "price"
            ]
        )

        product_id = (
            product.get(
                "product_id",
                "",
            )
        )

        # -----------------------------------------------------
        # Base Qwen
        # -----------------------------------------------------

        (
            base_price,
            base_response,
        ) = predict_base(
            model,
            tokenizer,
            description,
        )

        # -----------------------------------------------------
        # Fine-tuned Qwen
        # -----------------------------------------------------

        (
            fine_tuned_price,
            fine_tuned_response,
        ) = predict_finetuned(
            model,
            tokenizer,
            description,
        )

        # -----------------------------------------------------
        # Handle unparsable output.
        #
        # We count it separately.
        #
        # For metrics we use 0.0 so failures are penalized
        # rather than silently ignored.
        # -----------------------------------------------------

        if base_price is None:

            base_parse_failures += 1

            base_price = 0.0

        if fine_tuned_price is None:

            fine_tuned_parse_failures += 1

            fine_tuned_price = 0.0

        # -----------------------------------------------------
        # Errors
        # -----------------------------------------------------

        base_error = abs(
            actual_price
            - base_price
        )

        fine_tuned_error = abs(
            actual_price
            - fine_tuned_price
        )

        actual_prices.append(
            actual_price
        )

        base_predictions.append(
            base_price
        )

        fine_tuned_predictions.append(
            fine_tuned_price
        )

        results.append(
            {
                "product_id":
                    product_id,

                "actual_price":
                    actual_price,

                "base_price":
                    base_price,

                "fine_tuned_price":
                    fine_tuned_price,

                "base_absolute_error":
                    round(
                        base_error,
                        2,
                    ),

                "fine_tuned_absolute_error":
                    round(
                        fine_tuned_error,
                        2,
                    ),

                "base_response":
                    base_response,

                "fine_tuned_response":
                    fine_tuned_response,

                "description":
                    description,
            }
        )

        # -----------------------------------------------------
        # Progress display
        # -----------------------------------------------------

        print(
            f"[{index}/{len(test_products)}] "
            f"Actual=${actual_price:.2f} | "
            f"Base=${base_price:.2f} | "
            f"Fine-tuned=${fine_tuned_price:.2f}"
        )

    # ---------------------------------------------------------
    # 6. Calculate Base metrics
    # ---------------------------------------------------------

    base_mae = calculate_mae(
        actual_prices,
        base_predictions,
    )

    base_rmsle = calculate_rmsle(
        actual_prices,
        base_predictions,
    )

    # ---------------------------------------------------------
    # 7. Calculate Fine-tuned metrics
    # ---------------------------------------------------------

    fine_tuned_mae = calculate_mae(
        actual_prices,
        fine_tuned_predictions,
    )

    fine_tuned_rmsle = (
        calculate_rmsle(
            actual_prices,
            fine_tuned_predictions,
        )
    )

    # ---------------------------------------------------------
    # 8. Save per-product results
    # ---------------------------------------------------------

    save_csv(
        results
    )

    save_json(
        results
    )

    # ---------------------------------------------------------
    # 9. Build summary
    # ---------------------------------------------------------

    summary = {

        "test_products":
            len(
                test_products
            ),

        "base": {

            "mae":
                base_mae,

            "rmsle":
                base_rmsle,

            "parse_failures":
                base_parse_failures,
        },

        "fine_tuned": {

            "mae":
                fine_tuned_mae,

            "rmsle":
                fine_tuned_rmsle,

            "parse_failures":
                fine_tuned_parse_failures,
        },

        "mae_improvement": (
            base_mae
            - fine_tuned_mae
        ),

        "rmsle_improvement": (
            base_rmsle
            - fine_tuned_rmsle
        ),
    }

    # ---------------------------------------------------------
    # 10. Save summary
    # ---------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # 11. Print final comparison
    # ---------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL HELD-OUT TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "Products:",
        len(
            test_products
        )
    )

    print()

    print(
        "BASE QWEN"
    )

    print(
        "MAE:",
        round(
            base_mae,
            2,
        )
    )

    print(
        "RMSLE:",
        round(
            base_rmsle,
            4,
        )
    )

    print(
        "Parse failures:",
        base_parse_failures,
    )

    print()

    print(
        "FINE-TUNED QWEN"
    )

    print(
        "MAE:",
        round(
            fine_tuned_mae,
            2,
        )
    )

    print(
        "RMSLE:",
        round(
            fine_tuned_rmsle,
            4,
        )
    )

    print(
        "Parse failures:",
        fine_tuned_parse_failures,
    )

    print()

    print(
        "MAE improvement:",
        round(
            base_mae
            - fine_tuned_mae,
            2,
        )
    )

    print(
        "RMSLE improvement:",
        round(
            base_rmsle
            - fine_tuned_rmsle,
            4,
        )
    )

    print()

    print(
        "Detailed CSV:",
        CSV_OUTPUT_FILE
    )

    print(
        "Detailed JSON:",
        JSON_OUTPUT_FILE
    )

    print(
        "Summary:",
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()