import math
import re

import torch

from peft import PeftModel

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_log_error,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "./qwen-price-lora-final"


# -------------------------------------------------------------
# Load tokenizer + base model + trained LoRA adapter
# -------------------------------------------------------------

def load_model():

    # ---------------------------------------------------------
    # 1. Configure 4-bit QLoRA-style loading
    #
    # Tesla T4:
    # - base weights stored in 4-bit NF4
    # - computation uses FP16
    # ---------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ---------------------------------------------------------
    # 2. Load tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ---------------------------------------------------------
    # 3. Load original Qwen base model
    # ---------------------------------------------------------

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # ---------------------------------------------------------
    # 4. Attach our trained LoRA adapter
    # ---------------------------------------------------------

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    # Inference mode
    model.eval()

    return model, tokenizer


# -------------------------------------------------------------
# Convert product description into Qwen input tensors
# -------------------------------------------------------------

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
            "content": description,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move token tensors to the same GPU as the model
    inputs = inputs.to(
        model.device
    )

    return inputs


# -------------------------------------------------------------
# Generate assistant response
# -------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    inputs,
):

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    # ---------------------------------------------------------
    # model.generate() returns:
    #
    # prompt tokens + generated tokens
    #
    # We only want the newly generated part.
    # ---------------------------------------------------------

    prompt_length = inputs[
        "input_ids"
    ].shape[1]

    generated_ids = output_ids[0][
        prompt_length:
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return response.strip()


# -------------------------------------------------------------
# Extract numeric price from model response
#
# Examples:
#
# "$275"        -> 275.0
# "275"         -> 275.0
# "$275.99"     -> 275.99
# -------------------------------------------------------------

def extract_price(response):

    match = re.search(
        r"\$?\s*(\d+(?:\.\d+)?)",
        response,
    )

    if not match:
        return 0.0

    price = float(
        match.group(1)
    )

    # Product prices should not be negative.
    # This also keeps RMSLE input valid.
    return max(
        0.0,
        price,
    )


# -------------------------------------------------------------
# Base Qwen prediction
#
# Same PeftModel object is used, but LoRA is temporarily disabled.
# -------------------------------------------------------------

def predict_base(
    model,
    tokenizer,
    description,
):

    inputs = prepare_inputs(
        tokenizer,
        model,
        description,
    )

    with model.disable_adapter():

        response = generate_response(
            model,
            tokenizer,
            inputs,
        )

    price = extract_price(
        response
    )

    return price, response


# -------------------------------------------------------------
# Fine-tuned Qwen prediction
#
# LoRA adapter remains enabled.
# -------------------------------------------------------------

def predict_finetuned(
    model,
    tokenizer,
    description,
):

    inputs = prepare_inputs(
        tokenizer,
        model,
        description,
    )

    response = generate_response(
        model,
        tokenizer,
        inputs,
    )

    price = extract_price(
        response
    )

    return price, response


# -------------------------------------------------------------
# RMSLE
# -------------------------------------------------------------

def calculate_rmsle(
    actual_prices,
    predicted_prices,
):

    msle = mean_squared_log_error(
        actual_prices,
        predicted_prices,
    )

    return math.sqrt(
        msle
    )


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this demo."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "\nLoading Qwen + LoRA adapter..."
    )

    model, tokenizer = load_model()

    print(
        "Model loaded successfully."
    )

    # ---------------------------------------------------------
    # Held-out test dataset
    #
    # IMPORTANT:
    # These products should NOT be part of the training set.
    #
    # The prices below are only demo values.
    # Later, replace this with the real held-out product set.
    # ---------------------------------------------------------

    test_products = [
        {
            "description":
                "2 meter braided USB-C fast charging cable",
            "price": 18,
        },
        {
            "description":
                "Bluetooth over-ear headphones with "
                "active noise cancellation and long battery life",
            "price": 240,
        },
        {
            "description":
                "55 inch OLED 4K HDR smart television "
                "with Dolby Atmos",
            "price": 700,
        },
        {
            "description":
                "27 inch 4K gaming monitor "
                "with high refresh rate",
            "price": 425,
        },
        {
            "description":
                "Stainless steel programmable "
                "espresso coffee machine",
            "price": 150,
        },
    ]

    actual_prices = []

    base_predictions = []

    fine_tuned_predictions = []

    # ---------------------------------------------------------
    # Evaluate each product
    # ---------------------------------------------------------

    for index, product in enumerate(
        test_products,
        start=1,
    ):

        description = product[
            "description"
        ]

        actual_price = product[
            "price"
        ]

        # -----------------------------------------------------
        # Base model
        # -----------------------------------------------------

        base_price, base_response = predict_base(
            model,
            tokenizer,
            description,
        )

        # -----------------------------------------------------
        # Fine-tuned model
        # -----------------------------------------------------

        fine_tuned_price, fine_tuned_response = (
            predict_finetuned(
                model,
                tokenizer,
                description,
            )
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

        # -----------------------------------------------------
        # Individual result
        # -----------------------------------------------------

        print(
            "\n"
            + "=" * 80
        )

        print(
            "PRODUCT",
            index,
        )

        print(
            description
        )

        print(
            "\nActual price:",
            actual_price
        )

        print(
            "\nBase Qwen raw response:",
            base_response
        )

        print(
            "Base Qwen price:",
            base_price
        )

        print(
            "\nFine-tuned raw response:",
            fine_tuned_response
        )

        print(
            "Fine-tuned price:",
            fine_tuned_price
        )

        print(
            "\nBase absolute error:",
            round(
                abs(
                    actual_price
                    - base_price
                ),
                2,
            )
        )

        print(
            "Fine-tuned absolute error:",
            round(
                abs(
                    actual_price
                    - fine_tuned_price
                ),
                2,
            )
        )

    # ---------------------------------------------------------
    # Base model metrics
    # ---------------------------------------------------------

    base_mae = mean_absolute_error(
        actual_prices,
        base_predictions,
    )

    base_rmsle = calculate_rmsle(
        actual_prices,
        base_predictions,
    )

    # ---------------------------------------------------------
    # Fine-tuned model metrics
    # ---------------------------------------------------------

    fine_tuned_mae = mean_absolute_error(
        actual_prices,
        fine_tuned_predictions,
    )

    fine_tuned_rmsle = calculate_rmsle(
        actual_prices,
        fine_tuned_predictions,
    )

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        "\nBase Qwen"
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
        "\nFine-tuned Qwen"
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
        "\nImprovement in MAE:",
        round(
            base_mae
            - fine_tuned_mae,
            2,
        )
    )


if __name__ == "__main__":
    main()