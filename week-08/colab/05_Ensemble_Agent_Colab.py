import json
import os
import re

import faiss
import numpy as np
import torch

from openai import OpenAI

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

ADAPTER_PATH = (
    "./qwen-price-lora-final"
)

TRAIN_FILE = (
    "amazon_price_train.jsonl"
)

INDEX_FILE = (
    "product_price.faiss"
)

METADATA_FILE = (
    "product_price_metadata.json"
)

EMBEDDING_MODEL = (
    "text-embedding-3-small"
)

FRONTIER_MODEL = (
    "gpt-5.1"
)

TOP_K = 5

EMBEDDING_BATCH_SIZE = 100


# -------------------------------------------------------------
# Initial ensemble weights
#
# These are NOT claimed to be optimal.
# Later they should be selected using validation data.
# -------------------------------------------------------------

SPECIALIST_WEIGHT = 0.50

FRONTIER_WEIGHT = 0.50


# =============================================================
# COMMON PRICE PARSER
# =============================================================

def extract_price(
    response,
):

    match = re.search(
        r"\$?\s*(\d[\d,]*(?:\.\d+)?)",
        response,
    )

    if match is None:
        return None

    value = (
        match.group(1)
        .replace(
            ",",
            "",
        )
    )

    try:

        return float(
            value
        )

    except ValueError:

        return None


# =============================================================
# SPECIALIST AGENT
# =============================================================

class SpecialistAgent:

    def __init__(self):

        print(
            "Initializing SpecialistAgent..."
        )

        if not os.path.isdir(
            ADAPTER_PATH
        ):

            raise FileNotFoundError(
                f"Adapter not found: "
                f"{ADAPTER_PATH}"
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

        response = (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True,
            )
            .strip()
        )

        return {
            "price":
                extract_price(
                    response
                ),

            "raw_response":
                response,
        }


# =============================================================
# PRODUCT KNOWLEDGE BASE
# =============================================================

class ProductKnowledgeBase:

    def __init__(self):

        self.client = OpenAI()

        self.index = None

        self.products = []

    def load_products(self):

        products = []

        with open(
            TRAIN_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = (
                    line.strip()
                )

                if line:

                    products.append(
                        json.loads(
                            line
                        )
                    )

        return products

    def create_embeddings(
        self,
        texts,
    ):

        response = (
            self.client.embeddings.create(

                model=EMBEDDING_MODEL,

                input=texts,
            )
        )

        return [

            item.embedding

            for item in response.data
        ]

    def build_index(self):

        print(
            "Building RAG index..."
        )

        self.products = (
            self.load_products()
        )

        embeddings = []

        for start in range(
            0,
            len(self.products),
            EMBEDDING_BATCH_SIZE,
        ):

            batch = self.products[
                start:
                start + EMBEDDING_BATCH_SIZE
            ]

            descriptions = [

                product[
                    "description"
                ]

                for product in batch
            ]

            embeddings.extend(
                self.create_embeddings(
                    descriptions
                )
            )

        matrix = np.array(
            embeddings,
            dtype="float32",
        )

        faiss.normalize_L2(
            matrix
        )

        self.index = (
            faiss.IndexFlatIP(
                matrix.shape[1]
            )
        )

        self.index.add(
            matrix
        )

        faiss.write_index(
            self.index,
            INDEX_FILE,
        )

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.products,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load_index(self):

        self.index = (
            faiss.read_index(
                INDEX_FILE
            )
        )

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            self.products = (
                json.load(
                    file
                )
            )

    def initialize(self):

        if (
            os.path.exists(
                INDEX_FILE
            )
            and
            os.path.exists(
                METADATA_FILE
            )
        ):

            print(
                "Loading existing RAG index..."
            )

            self.load_index()

        else:

            self.build_index()

    def search(
        self,
        query,
        top_k=TOP_K,
    ):

        embedding = (
            self.create_embeddings(
                [query]
            )[0]
        )

        vector = np.array(
            [embedding],
            dtype="float32",
        )

        faiss.normalize_L2(
            vector
        )

        (
            similarities,
            indexes,
        ) = self.index.search(
            vector,
            top_k,
        )

        results = []

        for (
            similarity,
            index,
        ) in zip(
            similarities[0],
            indexes[0],
        ):

            product = (
                self.products[index]
            )

            results.append(
                {
                    "description":
                        product[
                            "description"
                        ],

                    "price":
                        float(
                            product[
                                "price"
                            ]
                        ),

                    "similarity":
                        float(
                            similarity
                        ),
                }
            )

        return results


# =============================================================
# FRONTIER AGENT
# =============================================================

class FrontierAgent:

    def __init__(
        self,
        knowledge_base,
    ):

        print(
            "Initializing FrontierAgent..."
        )

        self.client = OpenAI()

        self.knowledge_base = (
            knowledge_base
        )

        print(
            "FrontierAgent ready."
        )

    def build_context(
        self,
        products,
    ):

        lines = []

        for (
            position,
            product,
        ) in enumerate(
            products,
            start=1,
        ):

            lines.append(
                (
                    f"Comparable {position}\n"
                    f"Price: "
                    f"${product['price']:.2f}\n"
                    f"Description: "
                    f"{product['description']}"
                )
            )

        return "\n\n".join(
            lines
        )

    def price(
        self,
        description,
    ):

        retrieved = (
            self.knowledge_base.search(
                description,
                TOP_K,
            )
        )

        context = (
            self.build_context(
                retrieved
            )
        )

        prompt = f"""
You are an expert product price estimator.

Estimate the fair market price of the TARGET PRODUCT.

Use the comparable products below as evidence.

TARGET PRODUCT:

{description}


COMPARABLE PRODUCTS:

{context}


Return ONLY the estimated price in US dollars.

Example:

$129.99
"""

        response = (
            self.client.responses.create(

                model=FRONTIER_MODEL,

                input=prompt,
            )
        )

        raw_response = (
            response.output_text.strip()
        )

        return {
            "price":
                extract_price(
                    raw_response
                ),

            "raw_response":
                raw_response,

            "retrieved_products":
                retrieved,
        }


# =============================================================
# ENSEMBLE AGENT
# =============================================================

class EnsembleAgent:

    def __init__(
        self,
        specialist,
        frontier,
    ):

        self.specialist = (
            specialist
        )

        self.frontier = (
            frontier
        )

    def price(
        self,
        description,
    ):

        # -----------------------------------------------------
        # Independent estimate 1
        # -----------------------------------------------------

        specialist_result = (
            self.specialist.price(
                description
            )
        )

        # -----------------------------------------------------
        # Independent estimate 2
        # -----------------------------------------------------

        frontier_result = (
            self.frontier.price(
                description
            )
        )

        specialist_price = (
            specialist_result[
                "price"
            ]
        )

        frontier_price = (
            frontier_result[
                "price"
            ]
        )

        # -----------------------------------------------------
        # Fallback behavior
        # -----------------------------------------------------

        if (
            specialist_price is None
            and
            frontier_price is None
        ):

            final_price = None

        elif specialist_price is None:

            final_price = (
                frontier_price
            )

        elif frontier_price is None:

            final_price = (
                specialist_price
            )

        else:

            final_price = (

                SPECIALIST_WEIGHT
                * specialist_price

                +

                FRONTIER_WEIGHT
                * frontier_price
            )

        return {

            "ensemble_price":
                final_price,

            "specialist_price":
                specialist_price,

            "frontier_price":
                frontier_price,

            "specialist_response":
                specialist_result[
                    "raw_response"
                ],

            "frontier_response":
                frontier_result[
                    "raw_response"
                ],

            "retrieved_products":
                frontier_result[
                    "retrieved_products"
                ],
        }


# =============================================================
# MAIN
# =============================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    if not os.environ.get(
        "OPENAI_API_KEY"
    ):

        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    # ---------------------------------------------------------
    # Initialize components once
    # ---------------------------------------------------------

    knowledge_base = (
        ProductKnowledgeBase()
    )

    knowledge_base.initialize()

    specialist = (
        SpecialistAgent()
    )

    frontier = (
        FrontierAgent(
            knowledge_base
        )
    )

    ensemble = (
        EnsembleAgent(
            specialist,
            frontier,
        )
    )

    # ---------------------------------------------------------
    # Test product
    # ---------------------------------------------------------

    description = (
        "Premium stainless steel espresso "
        "machine with integrated milk frother, "
        "programmable drink settings and "
        "adjustable temperature control"
    )

    result = (
        ensemble.price(
            description
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "ENSEMBLE RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Product:"
    )

    print(
        description
    )

    print()

    print(
        "Specialist price:",
        result[
            "specialist_price"
        ]
    )

    print(
        "Frontier price:",
        result[
            "frontier_price"
        ]
    )

    print(
        "Ensemble price:",
        result[
            "ensemble_price"
        ]
    )


if __name__ == "__main__":
    main()