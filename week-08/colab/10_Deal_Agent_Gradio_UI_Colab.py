import json
import os
import re
from pathlib import Path

import feedparser
import faiss
import gradio as gr
import numpy as np
import requests
import torch

from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel
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

TOP_K = 5

SPECIALIST_WEIGHT = 0.50
FRONTIER_WEIGHT = 0.50

MAX_AGENT_STEPS = 12
MAX_SELECTED_DEALS = 5

ENABLE_PUSH = False

PUSHOVER_URL = (
    "https://api.pushover.net/1/messages.json"
)


# ============================================================
# ARTIFACT LOCATION
#
# Prefer Google Drive if mounted.
# Otherwise use current Colab directory.
# ============================================================

DRIVE_BASE = Path(
    "/content/drive/MyDrive/"
    "llm-engineering/week-08"
)

LOCAL_BASE = Path(".")


def choose_artifact(
    filename,
):

    drive_path = (
        DRIVE_BASE
        / filename
    )

    local_path = (
        LOCAL_BASE
        / filename
    )

    if drive_path.exists():
        return str(
            drive_path
        )

    return str(
        local_path
    )


ADAPTER_PATH = choose_artifact(
    "qwen-price-lora-final"
)

INDEX_FILE = choose_artifact(
    "product_price.faiss"
)

METADATA_FILE = choose_artifact(
    "product_price_metadata.json"
)

MEMORY_FILE = (
    str(
        DRIVE_BASE
        / "deal_agent_memory.json"
    )
    if DRIVE_BASE.exists()
    else "deal_agent_memory.json"
)


# ============================================================
# DEAL DATA
# ============================================================

class Deal(BaseModel):

    product_description: str

    price: float

    url: str


class DealSelection(BaseModel):

    deals: list[Deal]


# ============================================================
# MEMORY
# ============================================================

class DealMemory:

    def __init__(
        self,
        path,
    ):

        self.path = Path(
            path
        )

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

                self.data = (
                    json.load(
                        file
                    )
                )

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
            self.data[
                "seen_urls"
            ]
        )

    def notified_urls(self):

        return set(
            self.data[
                "notified_urls"
            ]
        )

    def mark_seen(
        self,
        url,
    ):

        if (
            url
            not in self.data[
                "seen_urls"
            ]
        ):

            self.data[
                "seen_urls"
            ].append(
                url
            )

            self.save()

    def mark_notified(
        self,
        url,
    ):

        if (
            url
            not in self.data[
                "notified_urls"
            ]
        ):

            self.data[
                "notified_urls"
            ].append(
                url
            )

        self.mark_seen(
            url
        )

        self.save()

    def already_notified(
        self,
        url,
    ):

        return (
            url
            in self.notified_urls()
        )


# ============================================================
# COMMON PRICE PARSER
# ============================================================

def extract_price(
    text,
):

    match = re.search(
        r"\$?\s*(\d[\d,]*(?:\.\d+)?)",
        text,
    )

    if not match:

        return None

    return float(
        match.group(1)
        .replace(
            ",",
            "",
        )
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
                f"Adapter missing: "
                f"{ADAPTER_PATH}"
            )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                QWEN_MODEL
            )
        )

        if (
            self.tokenizer.pad_token
            is None
        ):

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        quantization = (
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
                QWEN_MODEL,
                quantization_config=(
                    quantization
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
                "role":
                    "system",

                "content":
                    (
                        "You estimate product prices. "
                        "Reply only with the price "
                        "in US dollars."
                    ),
            },
            {
                "role":
                    "user",

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

        device = (
            self.model
            .get_input_embeddings()
            .weight
            .device
        )

        inputs = (
            inputs.to(
                device
            )
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

        generated = (
            output[0][
                input_length:
            ]
        )

        response = (
            self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            .strip()
        )

        return (
            extract_price(
                response
            )
        )


# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================

class ProductKnowledgeBase:

    def __init__(self):

        self.client = (
            OpenAI()
        )

        if not os.path.exists(
            INDEX_FILE
        ):

            raise FileNotFoundError(
                f"Missing RAG index: "
                f"{INDEX_FILE}"
            )

        if not os.path.exists(
            METADATA_FILE
        ):

            raise FileNotFoundError(
                f"Missing metadata: "
                f"{METADATA_FILE}"
            )

        print(
            "Loading RAG index..."
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
                json.load(
                    file
                )
            )

        print(
            "RAG products:",
            len(
                self.products
            )
        )

    def search(
        self,
        query,
        top_k=TOP_K,
    ):

        response = (
            self.client
            .embeddings
            .create(
                model=(
                    EMBEDDING_MODEL
                ),
                input=[
                    query
                ],
            )
        )

        vector = np.array(
            [
                response
                .data[0]
                .embedding
            ],
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

        self.client = (
            OpenAI()
        )

        self.knowledge_base = (
            knowledge_base
        )

    def price(
        self,
        description,
    ):

        retrieved = (
            self.knowledge_base
            .search(
                description
            )
        )

        blocks = []

        for (
            position,
            product,
        ) in enumerate(
            retrieved,
            start=1,
        ):

            blocks.append(
                (
                    f"Comparable {position}\n"
                    f"Price: "
                    f"${product['price']:.2f}\n"
                    f"Description: "
                    f"{product['description']}"
                )
            )

        context = (
            "\n\n".join(
                blocks
            )
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
            self.client
            .responses
            .create(
                model=(
                    FRONTIER_MODEL
                ),
                input=prompt,
            )
        )

        return (
            extract_price(
                response
                .output_text
                .strip()
            )
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

        specialist_price = (
            self.specialist
            .price(
                description
            )
        )

        frontier_price = (
            self.frontier
            .price(
                description
            )
        )

        if specialist_price is None:

            ensemble_price = (
                frontier_price
            )

        elif frontier_price is None:

            ensemble_price = (
                specialist_price
            )

        else:

            ensemble_price = (
                SPECIALIST_WEIGHT
                * specialist_price

                +

                FRONTIER_WEIGHT
                * frontier_price
            )

        return {
            "specialist_price":
                specialist_price,

            "frontier_price":
                frontier_price,

            "ensemble_price":
                (
                    round(
                        ensemble_price,
                        2,
                    )
                    if ensemble_price
                    is not None
                    else None
                ),
        }


# ============================================================
# SCANNER AGENT
# ============================================================

class ScannerAgent:

    def __init__(
        self,
        memory,
    ):

        self.client = (
            OpenAI()
        )

        self.memory = (
            memory
        )

    @staticmethod
    def clean_html(
        text,
    ):

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

    def scan(self):

        raw_deals = []

        for feed_url in FEEDS:

            feed = (
                feedparser.parse(
                    feed_url
                )
            )

            for entry in (
                feed.entries[:10]
            ):

                url = (
                    entry.get(
                        "link",
                        "",
                    )
                )

                if not url:

                    continue

                if (
                    url
                    in self.memory
                    .seen_urls()
                ):

                    continue

                raw_deals.append(
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
Select up to {MAX_SELECTED_DEALS} useful product deals.

Rules:

- Actual selling price must be clear.
- "$100 off" is NOT the selling price.
- Product description must contain enough details
  to estimate fair market value.
- Keep the original URL.
- Describe the product rather than the promotion.

Candidates:

{"---".join(blocks)}
"""

        response = (
            self.client
            .responses
            .parse(
                model=(
                    SCANNER_MODEL
                ),
                input=prompt,
                text_format=(
                    DealSelection
                ),
            )
        )

        selection = (
            response.output_parsed
        )

        if selection is None:

            return []

        deals = (
            selection.deals[
                :MAX_SELECTED_DEALS
            ]
        )

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

        self.client = (
            OpenAI()
        )

        self.memory = (
            memory
        )

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

        if (
            self.memory
            .already_notified(
                url
            )
        ):

            return {
                "status":
                    "duplicate_ignored",

                "message":
                    (
                        "This deal was "
                        "already notified."
                    ),
            }

        saving = (
            estimated_true_value
            - deal_price
        )

        prompt = f"""
Write a concise two-sentence deal notification.

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
            self.client
            .responses
            .create(
                model=(
                    MESSENGER_MODEL
                ),
                input=prompt,
            )
        )

        message = (
            response
            .output_text
            .strip()
        )

        if not ENABLE_PUSH:

            delivery = {
                "status":
                    "preview_only"
            }

        else:

            if not self.pushover_user:

                raise RuntimeError(
                    "PUSHOVER_USER "
                    "is missing."
                )

            if not self.pushover_token:

                raise RuntimeError(
                    "PUSHOVER_TOKEN "
                    "is missing."
                )

            response = (
                requests.post(
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
            )

            response.raise_for_status()

            delivery = (
                response.json()
            )

        self.memory.mark_notified(
            url
        )

        return {
            "status":
                "sent",

            "message":
                message,

            "saving":
                round(
                    saving,
                    2,
                ),

            "delivery":
                delivery,
        }


# ============================================================
# PLANNER TOOL DEFINITIONS
# ============================================================

PLANNER_TOOLS = [
    {
        "type":
            "function",

        "name":
            "scan_for_deals",

        "description":
            (
                "Find new product deals "
                "not previously processed."
            ),

        "parameters": {
            "type":
                "object",

            "properties": {},

            "required": [],

            "additionalProperties":
                False,
        },

        "strict":
            True,
    },

    {
        "type":
            "function",

        "name":
            "estimate_true_value",

        "description":
            (
                "Estimate the fair market "
                "value of a product."
            ),

        "parameters": {
            "type":
                "object",

            "properties": {
                "description": {
                    "type":
                        "string"
                },

                "deal_price": {
                    "type":
                        "number"
                },

                "url": {
                    "type":
                        "string"
                },
            },

            "required": [
                "description",
                "deal_price",
                "url",
            ],

            "additionalProperties":
                False,
        },

        "strict":
            True,
    },

    {
        "type":
            "function",

        "name":
            "notify_user_of_deal",

        "description":
            (
                "Notify the user about "
                "the single best bargain."
            ),

        "parameters": {
            "type":
                "object",

            "properties": {
                "description": {
                    "type":
                        "string"
                },

                "deal_price": {
                    "type":
                        "number"
                },

                "estimated_true_value": {
                    "type":
                        "number"
                },

                "url": {
                    "type":
                        "string"
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

        "strict":
            True,
    },
]


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

        self.client = (
            OpenAI()
        )

        self.scanner = scanner
        self.ensemble = ensemble
        self.messenger = messenger

        self.evaluations = []
        self.notification = None

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

            description = (
                arguments[
                    "description"
                ]
            )

            deal_price = float(
                arguments[
                    "deal_price"
                ]
            )

            url = (
                arguments[
                    "url"
                ]
            )

            valuation = (
                self.ensemble.price(
                    description
                )
            )

            estimated_value = (
                valuation[
                    "ensemble_price"
                ]
            )

            saving = (
                round(
                    estimated_value
                    - deal_price,
                    2,
                )
                if estimated_value
                is not None
                else None
            )

            record = {
                "description":
                    description,

                "deal_price":
                    deal_price,

                "specialist_price":
                    valuation[
                        "specialist_price"
                    ],

                "frontier_price":
                    valuation[
                        "frontier_price"
                    ],

                "estimated_value":
                    estimated_value,

                "saving":
                    saving,

                "url":
                    url,
            }

            self.evaluations.append(
                record
            )

            return record

        if name == "notify_user_of_deal":

            result = (
                self.messenger.notify(
                    **arguments
                )
            )

            self.notification = {
                **arguments,
                **result,
            }

            return result

        raise ValueError(
            f"Unknown tool: {name}"
        )

    def run(self):

        self.evaluations = []
        self.notification = None

        conversation = [
            {
                "role":
                    "system",

                "content":
                    (
                        "You are an autonomous "
                        "deal-finding planner. "
                        "First scan for new deals. "
                        "Evaluate useful candidates. "
                        "Compare deal price with fair value. "
                        "Choose the single strongest bargain. "
                        "Notify once only. "
                        "If no suitable deal exists, finish."
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    (
                        "Find the best new bargain "
                        "and notify me."
                    ),
            },
        ]

        notification_sent = False

        final_text = ""

        for step in range(
            1,
            MAX_AGENT_STEPS + 1,
        ):

            response = (
                self.client
                .responses
                .create(
                    model=(
                        PLANNER_MODEL
                    ),
                    input=conversation,
                    tools=(
                        PLANNER_TOOLS
                    ),
                )
            )

            conversation += (
                response.output
            )

            function_calls = [
                item

                for item
                in response.output

                if item.type
                == "function_call"
            ]

            if not function_calls:

                final_text = (
                    response.output_text
                    or "Planner finished."
                )

                return {
                    "final_text":
                        final_text,

                    "steps":
                        step,

                    "evaluations":
                        self.evaluations,

                    "notification":
                        self.notification,
                }

            for call in function_calls:

                arguments = (
                    json.loads(
                        call.arguments
                    )
                )

                if (
                    call.name
                    == "notify_user_of_deal"
                    and notification_sent
                ):

                    result = {
                        "status":
                            (
                                "second_"
                                "notification_"
                                "blocked"
                            )
                    }

                else:

                    result = (
                        self.execute_tool(
                            call.name,
                            arguments,
                        )
                    )

                    if (
                        call.name
                        == "notify_user_of_deal"
                    ):

                        notification_sent = (
                            True
                        )

                conversation.append(
                    {
                        "type":
                            "function_call_output",

                        "call_id":
                            call.call_id,

                        "output":
                            json.dumps(
                                result
                            ),
                    }
                )

        return {
            "final_text":
                (
                    "Planner stopped because "
                    "maximum step limit was reached."
                ),

            "steps":
                MAX_AGENT_STEPS,

            "evaluations":
                self.evaluations,

            "notification":
                self.notification,
        }


# ============================================================
# APPLICATION INITIALIZATION
#
# Expensive models load ONCE here.
# Button clicks reuse them.
# ============================================================

def initialize_application():

    if not os.environ.get(
        "OPENAI_API_KEY"
    ):

        raise RuntimeError(
            "OPENAI_API_KEY "
            "is not set."
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "Select a GPU runtime "
            "in Colab."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "Adapter:",
        ADAPTER_PATH
    )

    print(
        "FAISS index:",
        INDEX_FILE
    )

    print(
        "Memory:",
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
            scanner,
            ensemble,
            messenger,
        )
    )

    return (
        planner,
        memory,
    )


PLANNER, MEMORY = (
    initialize_application()
)


# ============================================================
# GRADIO CALLBACK
# ============================================================

def run_agent():

    try:

        result = (
            PLANNER.run()
        )

        rows = []

        for evaluation in (
            result[
                "evaluations"
            ]
        ):

            rows.append(
                [
                    evaluation[
                        "description"
                    ],

                    evaluation[
                        "deal_price"
                    ],

                    evaluation[
                        "specialist_price"
                    ],

                    evaluation[
                        "frontier_price"
                    ],

                    evaluation[
                        "estimated_value"
                    ],

                    evaluation[
                        "saving"
                    ],

                    evaluation[
                        "url"
                    ],
                ]
            )

        notification = (
            result[
                "notification"
            ]
        )

        if notification:

            notification_text = (
                f"""
### Notification

**Product:**  
{notification.get('description', '')}

**Deal price:**  
${notification.get('deal_price', 0):.2f}

**Estimated fair value:**  
${notification.get('estimated_true_value', 0):.2f}

**Estimated saving:**  
${notification.get('saving', 0):.2f}

**Status:**  
{notification.get('status', '')}

**Message:**  
{notification.get('message', '')}

**URL:**  
{notification.get('url', '')}
"""
            )

        else:

            notification_text = (
                "### Notification\n"
                "No notification was sent."
            )

        status = (
            f"""
### Planner

**Steps:** {result['steps']}

**Final response:**  
{result['final_text']}

**Seen deals:**  
{len(MEMORY.data['seen_urls'])}

**Previously notified:**  
{len(MEMORY.data['notified_urls'])}
"""
        )

        return (
            rows,
            notification_text,
            status,
        )

    except Exception as error:

        return (
            [],
            "### Notification\nNo result.",
            (
                "### Error\n"
                f"`{type(error).__name__}: "
                f"{error}`"
            ),
        )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Autonomous Deal Finder"
) as demo:

    gr.Markdown(
        """
# Autonomous Deal Finder

The planner scans current deals, estimates fair value using
**Qwen LoRA + RAG/GPT-5.1**, selects the strongest opportunity
and prepares a notification.
"""
    )

    run_button = (
        gr.Button(
            "Find Best Deal",
            variant="primary",
        )
    )

    results_table = (
        gr.Dataframe(
            headers=[
                "Product",
                "Deal Price",
                "Specialist",
                "Frontier",
                "Fair Value",
                "Saving",
                "URL",
            ],
            datatype=[
                "str",
                "number",
                "number",
                "number",
                "number",
                "number",
                "str",
            ],
            interactive=False,
            label="Evaluated Deals",
        )
    )

    with gr.Row():

        notification_output = (
            gr.Markdown(
                "### Notification\n"
                "No run yet."
            )
        )

        status_output = (
            gr.Markdown(
                "### Planner\n"
                "Ready."
            )
        )

    run_button.click(
        fn=run_agent,
        inputs=[],
        outputs=[
            results_table,
            notification_output,
            status_output,
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    demo.launch(
        debug=True
    )


if __name__ == "__main__":
    main()