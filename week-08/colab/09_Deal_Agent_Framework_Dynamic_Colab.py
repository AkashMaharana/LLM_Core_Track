import json
import os
import re
from pathlib import Path

import feedparser
import faiss
import numpy as np
import requests
import torch

from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================
# CONFIG
# ============================================================

PLANNER_MODEL = "gpt-5.1"
FRONTIER_MODEL = "gpt-5.1"
SCANNER_MODEL = "gpt-5.1"
MESSENGER_MODEL = "gpt-5.1"

EMBEDDING_MODEL = "text-embedding-3-small"

QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = (
    "/content/drive/MyDrive/"
    "llm-engineering/week-08/"
    "qwen-price-lora-final"
)

INDEX_FILE = "product_price.faiss"
METADATA_FILE = "product_price_metadata.json"

TOP_K = 5

SPECIALIST_WEIGHT = 0.50
FRONTIER_WEIGHT = 0.50

MAX_AGENT_STEPS = 12
MAX_SELECTED_DEALS = 5

# Keep notification safe while learning.
ENABLE_PUSH = False

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]


# ============================================================
# MEMORY LOCATION
#
# If Google Drive is mounted, memory survives Colab timeout.
# Otherwise it survives only in the current runtime.
# ============================================================

DRIVE_MEMORY = Path(
    "/content/drive/MyDrive/"
    "llm-engineering/week-08/deal_agent_memory.json"
)

LOCAL_MEMORY = Path(
    "./deal_agent_memory.json"
)

MEMORY_FILE = (
    DRIVE_MEMORY
    if DRIVE_MEMORY.parent.exists()
    else LOCAL_MEMORY
)


# ============================================================
# DATA MODELS
# ============================================================

class Deal(BaseModel):
    product_description: str
    price: float
    url: str


class DealSelection(BaseModel):
    deals: list[Deal]


# ============================================================
# PERSISTENT MEMORY
# ============================================================

class DealMemory:

    def __init__(self, path):
        self.path = Path(path)

        self.data = {
            "seen_urls": [],
            "notified_urls": [],
        }

        self.load()

    def load(self):
        if self.path.exists():
            with open(
                self.path,
                "r",
                encoding="utf-8",
            ) as file:
                self.data = json.load(file)

        self.data.setdefault(
            "seen_urls",
            [],
        )

        self.data.setdefault(
            "notified_urls",
            [],
        )

    def save(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.data,
                file,
                indent=2,
            )

    def seen_urls(self):
        return set(
            self.data["seen_urls"]
        )

    def notified_urls(self):
        return set(
            self.data["notified_urls"]
        )

    def mark_seen(self, url):
        if url not in self.data["seen_urls"]:
            self.data["seen_urls"].append(url)
            self.save()

    def mark_notified(self, url):
        if url not in self.data["notified_urls"]:
            self.data["notified_urls"].append(url)

        self.mark_seen(url)
        self.save()

    def already_notified(self, url):
        return (
            url
            in self.notified_urls()
        )


# ============================================================
# COMMON PRICE PARSER
# ============================================================

def extract_price(text):

    match = re.search(
        r"\$?\s*(\d[\d,]*(?:\.\d+)?)",
        text,
    )

    if not match:
        return None

    return float(
        match.group(1)
        .replace(",", "")
    )


# ============================================================
# SPECIALIST AGENT
# ============================================================

class SpecialistAgent:

    def __init__(self):

        print(
            "Loading SpecialistAgent..."
        )

        if not os.path.isdir(
            ADAPTER_PATH
        ):
            raise FileNotFoundError(
                f"Adapter missing: {ADAPTER_PATH}"
            )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                QWEN_MODEL
            )
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base = (
            AutoModelForCausalLM
            .from_pretrained(
                QWEN_MODEL,
                quantization_config=quantization,
                device_map="auto",
                dtype=torch.float16,
            )
        )

        self.model = (
            PeftModel.from_pretrained(
                base,
                ADAPTER_PATH,
                is_trainable=False,
            )
        )

        self.model.eval()

        print(
            "SpecialistAgent ready."
        )

    def price(self, description):

        messages = [
            {
                "role": "system",
                "content":
                    "You estimate product prices. "
                    "Reply only with the price in US dollars.",
            },
            {
                "role": "user",
                "content": description,
            },
        ]

        inputs = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        )

        device = (
            self.model
            .get_input_embeddings()
            .weight
            .device
        )

        inputs = inputs.to(device)

        input_length = (
            inputs["input_ids"]
            .shape[1]
        )

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer
                    .eos_token_id
                ),
            )

        generated = output[0][
            input_length:
        ]

        raw = (
            self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            .strip()
        )

        return extract_price(raw)


# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================

class ProductKnowledgeBase:

    def __init__(self):

        self.client = OpenAI()

        self.index = None
        self.products = []

        if (
            os.path.exists(INDEX_FILE)
            and os.path.exists(METADATA_FILE)
        ):
            self.load_index()

        else:
            print(
                "RAG index not found. Building it..."
            )

            self.build_index()

    def load_products(self):

        train_file = (
            "amazon_price_train.jsonl"
        )

        if not os.path.exists(
            train_file
        ):
            raise FileNotFoundError(
                f"Missing {train_file}. "
                "Run dataset preparation first."
            )

        products = []

        with open(
            train_file,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                products.append(
                    json.loads(line)
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

        self.products = (
            self.load_products()
        )

        print(
            "Products:",
            len(self.products)
        )

        all_embeddings = []

        batch_size = 100

        for start in range(
            0,
            len(self.products),
            batch_size,
        ):

            batch = self.products[
                start:
                start + batch_size
            ]

            descriptions = [
                product["description"]
                for product in batch
            ]

            print(
                "Embedding:",
                start + 1,
                "to",
                min(
                    start + batch_size,
                    len(self.products),
                ),
            )

            embeddings = (
                self.create_embeddings(
                    descriptions
                )
            )

            all_embeddings.extend(
                embeddings
            )

        matrix = np.array(
            all_embeddings,
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

        print(
            "RAG index created."
        )

    def load_index(self):

        print(
            "Loading existing RAG index..."
        )

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
                json.load(file)
            )

        print(
            "RAG products:",
            len(self.products)
        )

    def search(
        self,
        query,
        top_k=TOP_K,
    ):

        response = (
            self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query],
            )
        )

        embedding = (
            response.data[0]
            .embedding
        )

        vector = np.array(
            [embedding],
            dtype="float32",
        )

        faiss.normalize_L2(
            vector
        )

        similarities, indexes = (
            self.index.search(
                vector,
                top_k,
            )
        )

        results = []

        for similarity, index in zip(
            similarities[0],
            indexes[0],
        ):

            product = (
                self.products[
                    int(index)
                ]
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


# ============================================================
# FRONTIER AGENT
# ============================================================

class FrontierAgent:

    def __init__(
        self,
        knowledge_base,
    ):
        self.client = OpenAI()
        self.knowledge_base = (
            knowledge_base
        )

    def price(
        self,
        description,
    ):

        retrieved = (
            self.knowledge_base.search(
                description
            )
        )

        context_parts = []

        for i, product in enumerate(
            retrieved,
            start=1,
        ):
            context_parts.append(
                f"""
Comparable {i}
Price: ${product['price']:.2f}
Description: {product['description']}
"""
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
Estimate the fair market price of the target product.

TARGET PRODUCT:
{description}

COMPARABLE PRODUCTS:
{context}

Use the comparable products as evidence.
Return ONLY the estimated price in US dollars.

Example:
$149.99
"""

        response = (
            self.client.responses.create(
                model=FRONTIER_MODEL,
                input=prompt,
            )
        )

        return extract_price(
            response.output_text.strip()
        )


# ============================================================
# ENSEMBLE AGENT
# ============================================================

class EnsembleAgent:

    def __init__(
        self,
        specialist,
        frontier,
    ):
        self.specialist = specialist
        self.frontier = frontier

    def price(
        self,
        description,
    ):

        specialist_price = (
            self.specialist.price(
                description
            )
        )

        frontier_price = (
            self.frontier.price(
                description
            )
        )

        if specialist_price is None:
            return frontier_price

        if frontier_price is None:
            return specialist_price

        result = (
            SPECIALIST_WEIGHT
            * specialist_price
            +
            FRONTIER_WEIGHT
            * frontier_price
        )

        print(
            "\nVALUATION"
        )

        print(
            "Specialist:",
            specialist_price
        )

        print(
            "Frontier:",
            frontier_price
        )

        print(
            "Ensemble:",
            round(result, 2)
        )

        return round(
            result,
            2,
        )


# ============================================================
# SCANNER AGENT
# ============================================================

class ScannerAgent:

    def __init__(
        self,
        memory,
    ):
        self.client = OpenAI()
        self.memory = memory

    @staticmethod
    def clean_html(text):

        return (
            BeautifulSoup(
                text or "",
                "html.parser",
            )
            .get_text(
                " ",
                strip=True,
            )
        )

    def fetch_deals(self):

        deals = []

        for feed_url in FEEDS:

            print(
                "Reading:",
                feed_url
            )

            feed = (
                feedparser.parse(
                    feed_url
                )
            )

            for entry in feed.entries[:10]:

                url = entry.get(
                    "link",
                    "",
                )

                if not url:
                    continue

                if (
                    url
                    in self.memory.seen_urls()
                ):
                    continue

                deals.append(
                    {
                        "title":
                            self.clean_html(
                                entry.get(
                                    "title",
                                    "",
                                )
                            ),

                        "summary":
                            self.clean_html(
                                entry.get(
                                    "summary",
                                    "",
                                )
                            ),

                        "url":
                            url,
                    }
                )

        return deals

    def scan(self):

        raw_deals = (
            self.fetch_deals()
        )

        if not raw_deals:
            return []

        blocks = []

        for deal in raw_deals:
            blocks.append(
                f"""
Title:
{deal['title']}

Summary:
{deal['summary']}

URL:
{deal['url']}
"""
            )

        prompt = f"""
Select up to {MAX_SELECTED_DEALS} product deals.

Requirements:

- Actual selling price must be clear.
- "$100 off" is NOT a selling price.
- Product description must contain useful product details.
- Keep the original supplied URL.
- Rewrite descriptions around the product, not promotion.

Candidates:

{"---".join(blocks)}
"""

        response = (
            self.client.responses.parse(
                model=SCANNER_MODEL,
                input=prompt,
                text_format=DealSelection,
            )
        )

        result = (
            response.output_parsed
        )

        if result is None:
            return []

        deals = result.deals[
            :MAX_SELECTED_DEALS
        ]

        # Persist that we've processed these candidates.
        for deal in deals:
            self.memory.mark_seen(
                deal.url
            )

        return [
            deal.model_dump()
            for deal in deals
        ]


# ============================================================
# MESSENGER AGENT
# ============================================================

class MessengerAgent:

    def __init__(
        self,
        memory,
    ):

        self.client = OpenAI()
        self.memory = memory

        self.pushover_user = (
            os.environ.get(
                "PUSHOVER_USER"
            )
        )

        self.pushover_token = (
            os.environ.get(
                "PUSHOVER_TOKEN"
            )
        )

    def notify(
        self,
        description,
        deal_price,
        estimated_true_value,
        url,
    ):

        # ----------------------------------------------------
        # Application-level idempotency.
        # Do NOT rely on the LLM prompt alone.
        # ----------------------------------------------------

        if self.memory.already_notified(
            url
        ):

            return {
                "status":
                    "duplicate_ignored",

                "url":
                    url,
            }

        saving = (
            estimated_true_value
            - deal_price
        )

        prompt = f"""
Write a concise 2-sentence deal notification.

Product:
{description}

Selling price:
${deal_price:.2f}

Estimated fair value:
${estimated_true_value:.2f}

Estimated saving:
${saving:.2f}

Be factual.
Return only the message.
"""

        response = (
            self.client.responses.create(
                model=MESSENGER_MODEL,
                input=prompt,
            )
        )

        message = (
            response.output_text
            .strip()
        )

        # ----------------------------------------------------
        # Safe learning mode
        # ----------------------------------------------------

        if not ENABLE_PUSH:

            print()
            print("=" * 70)
            print("NOTIFICATION PREVIEW")
            print("=" * 70)

            print(message)
            print(url)

            delivery = {
                "status":
                    "preview_only"
            }

        else:

            if not self.pushover_user:
                raise RuntimeError(
                    "PUSHOVER_USER missing."
                )

            if not self.pushover_token:
                raise RuntimeError(
                    "PUSHOVER_TOKEN missing."
                )

            response = requests.post(
                PUSHOVER_URL,
                data={
                    "token":
                        self.pushover_token,

                    "user":
                        self.pushover_user,

                    "title":
                        "Deal Alert",

                    "message":
                        message,

                    "url":
                        url,

                    "url_title":
                        "View Deal",
                },
                timeout=15,
            )

            response.raise_for_status()

            delivery = (
                response.json()
            )

        # ----------------------------------------------------
        # Mark notification only after successful delivery /
        # preview.
        # ----------------------------------------------------

        self.memory.mark_notified(
            url
        )

        return {
            "status":
                "sent",

            "message":
                message,

            "saving":
                saving,

            "url":
                url,

            "delivery":
                delivery,
        }


# ============================================================
# AUTONOMOUS PLANNER
# ============================================================

class AutonomousPlannerAgent:

    def __init__(
        self,
        scanner,
        ensemble,
        messenger,
    ):

        self.client = OpenAI()

        self.scanner = scanner
        self.ensemble = ensemble
        self.messenger = messenger

    def tools(self):

        return [
            {
                "type": "function",
                "name":
                    "scan_for_deals",

                "description":
                    (
                        "Find new product deals that "
                        "have not already been processed."
                    ),

                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties":
                        False,
                },

                "strict": True,
            },

            {
                "type": "function",
                "name":
                    "estimate_true_value",

                "description":
                    (
                        "Estimate the fair market value "
                        "of a product."
                    ),

                "parameters": {
                    "type": "object",

                    "properties": {
                        "description": {
                            "type": "string"
                        }
                    },

                    "required": [
                        "description"
                    ],

                    "additionalProperties":
                        False,
                },

                "strict": True,
            },

            {
                "type": "function",
                "name":
                    "notify_user_of_deal",

                "description":
                    (
                        "Notify the user about the single "
                        "most compelling bargain. "
                        "Call this at most once."
                    ),

                "parameters": {
                    "type": "object",

                    "properties": {
                        "description": {
                            "type": "string"
                        },

                        "deal_price": {
                            "type": "number"
                        },

                        "estimated_true_value": {
                            "type": "number"
                        },

                        "url": {
                            "type": "string"
                        },
                    },

                    "required": [
                        "description",
                        "deal_price",
                        "estimated_true_value",
                        "url",
                    ],

                    "additionalProperties":
                        False,
                },

                "strict": True,
            },
        ]

    def execute_tool(
        self,
        name,
        arguments,
    ):

        if name == "scan_for_deals":
            return {
                "deals":
                    self.scanner.scan()
            }

        if name == "estimate_true_value":

            value = (
                self.ensemble.price(
                    arguments[
                        "description"
                    ]
                )
            )

            return {
                "description":
                    arguments[
                        "description"
                    ],

                "estimated_true_value":
                    value,
            }

        if name == "notify_user_of_deal":

            return (
                self.messenger.notify(
                    **arguments
                )
            )

        raise ValueError(
            f"Unknown tool: {name}"
        )

    def run(self):

        conversation = [
            {
                "role": "system",
                "content":
                    (
                        "You are an autonomous deal-finding "
                        "planner. "
                        "First scan for new deals. "
                        "Estimate the fair value of the useful "
                        "candidate products. "
                        "Compare each selling price with its "
                        "estimated fair value. "
                        "Select the single most compelling "
                        "bargain. "
                        "Notify the user once about that deal. "
                        "If there are no suitable new deals, "
                        "finish without notification."
                    ),
            },

            {
                "role": "user",
                "content":
                    (
                        "Find the best new bargain "
                        "and notify me."
                    ),
            },
        ]

        notification_sent = False

        for step in range(
            1,
            MAX_AGENT_STEPS + 1,
        ):

            print()
            print(
                f"===== PLANNER STEP {step} ====="
            )

            response = (
                self.client.responses.create(
                    model=PLANNER_MODEL,
                    input=conversation,
                    tools=self.tools(),
                )
            )

            conversation += (
                response.output
            )

            function_calls = [
                item
                for item in response.output
                if item.type
                == "function_call"
            ]

            if not function_calls:

                print()
                print(
                    "Planner finished:"
                )

                print(
                    response.output_text
                )

                return

            for tool_call in function_calls:

                name = tool_call.name

                arguments = (
                    json.loads(
                        tool_call.arguments
                    )
                )

                print(
                    "\nPlanner requested:",
                    name,
                )

                # ------------------------------------------------
                # Extra deterministic guardrail.
                #
                # Even if GPT asks twice, only allow one notify
                # call in this planner execution.
                # ------------------------------------------------

                if (
                    name
                    == "notify_user_of_deal"
                    and notification_sent
                ):

                    result = {
                        "status":
                            "second_notification_blocked"
                    }

                else:

                    result = (
                        self.execute_tool(
                            name,
                            arguments,
                        )
                    )

                    if (
                        name
                        == "notify_user_of_deal"
                        and result.get(
                            "status"
                        )
                        in {
                            "sent",
                            "duplicate_ignored",
                        }
                    ):
                        notification_sent = True

                print(
                    "Tool result:",
                    result,
                )

                conversation.append(
                    {
                        "type":
                            "function_call_output",

                        "call_id":
                            tool_call.call_id,

                        "output":
                            json.dumps(
                                result
                            ),
                    }
                )

        raise RuntimeError(
            "Planner exceeded maximum steps."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.environ.get(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Colab GPU is required "
            "for SpecialistAgent."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Memory file:",
        MEMORY_FILE
    )

    memory = DealMemory(
        MEMORY_FILE
    )

    knowledge_base = (
        ProductKnowledgeBase()
    )

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

    scanner = (
        ScannerAgent(
            memory
        )
    )

    messenger = (
        MessengerAgent(
            memory
        )
    )

    planner = (
        AutonomousPlannerAgent(
            scanner=scanner,
            ensemble=ensemble,
            messenger=messenger,
        )
    )

    planner.run()

    print()
    print(
        "=" * 70
    )

    print(
        "PERSISTED MEMORY"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            memory.data,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()