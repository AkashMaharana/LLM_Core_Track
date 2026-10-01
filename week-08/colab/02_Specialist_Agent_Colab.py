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

MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

# If you just finished Week 7 in the same Colab runtime:
ADAPTER_PATH = (
    "/content/drive/MyDrive/"
    "llm-engineering/week-08/"
    "qwen-price-lora-final"
)

# If you later store the adapter in Google Drive,
# change the path, for example:
#
# ADAPTER_PATH = (
#     "/content/drive/MyDrive/"
#     "qwen-price-lora-final"
# )


MAX_NEW_TOKENS = 16


# =============================================================
# SPECIALIST AGENT
# =============================================================

class SpecialistAgent:

    def __init__(self):

        print(
            "Initializing SpecialistAgent..."
        )

        self.tokenizer = None
        self.model = None

        self.load_model()

    # ---------------------------------------------------------
    # Load Qwen + trained LoRA once
    # ---------------------------------------------------------

    def load_model(self):

        if not os.path.isdir(
            ADAPTER_PATH
        ):

            raise FileNotFoundError(
                f"LoRA adapter not found: "
                f"{ADAPTER_PATH}"
            )

        adapter_config = os.path.join(
            ADAPTER_PATH,
            "adapter_config.json",
        )

        if not os.path.isfile(
            adapter_config
        ):

            raise FileNotFoundError(
                "adapter_config.json not found "
                f"inside {ADAPTER_PATH}"
            )

        print(
            "Loading tokenizer..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        if (
            self.tokenizer.pad_token
            is None
        ):

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        # -----------------------------------------------------
        # Inference is different from training.
        #
        # Our T4 training had AMP/BF16 gradient problems.
        # During inference there is:
        #
        # no backward pass
        # no gradients
        # no GradScaler
        #
        # so FP16 compute is perfectly reasonable here.
        # -----------------------------------------------------

        quantization_config = (
            BitsAndBytesConfig(

                load_in_4bit=True,

                bnb_4bit_quant_type="nf4",

                bnb_4bit_compute_dtype=(
                    torch.float16
                ),

                bnb_4bit_use_double_quant=True,
            )
        )

        print(
            "Loading Base Qwen..."
        )

        base_model = (
            AutoModelForCausalLM
            .from_pretrained(

                MODEL_NAME,

                quantization_config=(
                    quantization_config
                ),

                device_map="auto",

                dtype=torch.float16,
            )
        )

        print(
            "Loading LoRA adapter..."
        )

        self.model = (
            PeftModel.from_pretrained(

                base_model,

                ADAPTER_PATH,

                is_trainable=False,
            )
        )

        self.model.eval()

        print(
            "SpecialistAgent ready."
        )

    # ---------------------------------------------------------
    # Convert model text into numeric price
    # ---------------------------------------------------------

    def extract_price(
        self,
        response,
    ):

        match = re.search(
            r"\$?\s*(\d[\d,]*(?:\.\d+)?)",
            response,
        )

        if match is None:

            return None

        number = (
            match.group(1)
            .replace(
                ",",
                "",
            )
        )

        try:

            return float(
                number
            )

        except ValueError:

            return None

    # ---------------------------------------------------------
    # Price one product
    # ---------------------------------------------------------

    def price(
        self,
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
            self.tokenizer
            .apply_chat_template(

                messages,

                tokenize=True,

                add_generation_prompt=True,

                return_tensors="pt",

                return_dict=True,
            )
        )

        input_device = (
            self.model
            .get_input_embeddings()
            .weight
            .device
        )

        inputs = inputs.to(
            input_device
        )

        input_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )

        with torch.no_grad():

            output = (
                self.model.generate(

                    **inputs,

                    max_new_tokens=(
                        MAX_NEW_TOKENS
                    ),

                    do_sample=False,

                    pad_token_id=(
                        self.tokenizer
                        .eos_token_id
                    ),
                )
            )

        generated_tokens = (
            output[0][
                input_length:
            ]
        )

        raw_response = (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True,
            )
            .strip()
        )

        estimated_price = (
            self.extract_price(
                raw_response
            )
        )

        return {
            "price":
                estimated_price,

            "raw_response":
                raw_response,
        }


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

    # ---------------------------------------------------------
    # Model loads once here.
    # ---------------------------------------------------------

    specialist = (
        SpecialistAgent()
    )

    # ---------------------------------------------------------
    # Multiple requests reuse the same loaded model.
    # ---------------------------------------------------------

    products = [

        (
            "55 inch OLED 4K HDR Smart TV "
            "with Dolby Vision, Dolby Atmos "
            "and four HDMI ports"
        ),

        (
            "Premium stainless steel "
            "espresso machine with "
            "integrated milk frother "
            "and programmable settings"
        ),

        (
            "Wireless noise cancelling "
            "over-ear headphones with "
            "Bluetooth and 30 hour "
            "battery life"
        ),
    ]

    print()

    print(
        "=" * 70
    )

    print(
        "SPECIALIST AGENT RESULTS"
    )

    print(
        "=" * 70
    )

    for product in products:

        result = (
            specialist.price(
                product
            )
        )

        print()

        print(
            "Product:",
            product
        )

        print(
            "Estimated price:",
            result[
                "price"
            ]
        )

        print(
            "Raw response:",
            result[
                "raw_response"
            ]
        )


if __name__ == "__main__":
    main()