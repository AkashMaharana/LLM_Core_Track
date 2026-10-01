import modal

# modal run 02_Specialist_Agent_Modal.py

# =============================================================
# CONFIGURATION
# =============================================================

MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

LOCAL_ADAPTER_PATH = (
    "./qwen-price-lora-final"
)

REMOTE_ADAPTER_PATH = (
    "/models/qwen-price-lora-final"
)

HF_CACHE_PATH = (
    "/root/.cache/huggingface"
)

APP_NAME = (
    "week8-specialist-agent"
)


# =============================================================
# MODAL APP
# =============================================================

app = modal.App(
    APP_NAME
)


# =============================================================
# PERSISTENT HUGGING FACE CACHE
#
# First container:
#
# Hugging Face
#     ↓
# download Qwen
#     ↓
# Volume
#
# Later containers:
#
# Volume
#     ↓
# reuse cached Qwen
# =============================================================

hf_cache_volume = (
    modal.Volume.from_name(

        "week8-huggingface-cache",

        create_if_missing=True,
    )
)


# =============================================================
# REMOTE CONTAINER IMAGE
# =============================================================

image = (

    modal.Image
    .debian_slim(
        python_version="3.11"
    )

    .uv_pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
    )

    # ---------------------------------------------------------
    # Point Hugging Face cache into persistent Modal Volume.
    # ---------------------------------------------------------

    .env(
        {
            "HF_HOME":
                HF_CACHE_PATH,
        }
    )

    # ---------------------------------------------------------
    # Copy/mount our trained LoRA adapter into the remote
    # container.
    #
    # Current Modal uses add_local_dir() for local directories.
    # ---------------------------------------------------------

    .add_local_dir(

        LOCAL_ADAPTER_PATH,

        REMOTE_ADAPTER_PATH,
    )
)


# =============================================================
# REMOTE SPECIALIST MODEL
# =============================================================

@app.cls(

    image=image,

    gpu="T4",

    volumes={
        HF_CACHE_PATH:
            hf_cache_volume
    },

    timeout=900,
)
class SpecialistModel:

    # ---------------------------------------------------------
    # Runs ONCE when one Modal container starts.
    #
    # This is analogous to:
    #
    # @PostConstruct
    #
    # or application startup initialization.
    # ---------------------------------------------------------

    @modal.enter()
    def load_model(self):

        import re

        import torch

        from peft import (
            PeftModel
        )

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        print(
            "Loading Specialist model..."
        )

        self.re = re

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
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

        self.model = (
            PeftModel
            .from_pretrained(

                base_model,

                REMOTE_ADAPTER_PATH,

                is_trainable=False,
            )
        )

        self.model.eval()

        # -----------------------------------------------------
        # Persist any newly downloaded Hugging Face cache files.
        # Modal Volumes require commit() for changes to become
        # durable/visible outside the current container.
        # -----------------------------------------------------

        hf_cache_volume.commit()

        print(
            "Specialist model ready."
        )

    # ---------------------------------------------------------
    # Helper: parse price
    # ---------------------------------------------------------

    def extract_price(
        self,
        response,
    ):

        match = self.re.search(
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
    # This method can be invoked remotely.
    # ---------------------------------------------------------

    @modal.method()
    def price(
        self,
        description,
    ):

        import torch

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

                    max_new_tokens=16,

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
# LOCAL TEST CLIENT
# =============================================================

@app.local_entrypoint()
def main():

    description = (
        "55 inch OLED 4K HDR Smart TV "
        "with Dolby Vision, Dolby Atmos "
        "and four HDMI ports"
    )

    print(
        "Calling remote SpecialistAgent..."
    )

    # ---------------------------------------------------------
    # Instantiate the remote class.
    # ---------------------------------------------------------

    specialist = (
        SpecialistModel()
    )

    # ---------------------------------------------------------
    # Network / RPC boundary happens here.
    # ---------------------------------------------------------

    result = (
        specialist
        .price
        .remote(
            description
        )
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
    pass