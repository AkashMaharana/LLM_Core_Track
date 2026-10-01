import json
import os
import re

import faiss
import numpy as np

from openai import OpenAI

# !pip install -U openai pydantic feedparser beautifulsoup4 requests

# =============================================================
# CONFIGURATION
# =============================================================

TRAIN_FILE = "amazon_price_train.jsonl"

INDEX_FILE = "product_price.faiss"

METADATA_FILE = "product_price_metadata.json"

EMBEDDING_MODEL = "text-embedding-3-small"

FRONTIER_MODEL = "gpt-5.1"

TOP_K = 5

EMBEDDING_BATCH_SIZE = 100


# =============================================================
# PRODUCT KNOWLEDGE BASE
# =============================================================

class ProductKnowledgeBase:

    def __init__(self):

        self.client = OpenAI()

        self.index = None

        self.products = []

    # ---------------------------------------------------------
    # Load source products
    # ---------------------------------------------------------

    def load_products(self):

        products = []

        with open(
            TRAIN_FILE,
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

    # ---------------------------------------------------------
    # OpenAI embeddings
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Build FAISS knowledge base
    # ---------------------------------------------------------

    def build_index(self):

        print(
            "Building product knowledge base..."
        )

        self.products = (
            self.load_products()
        )

        all_embeddings = []

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
                product["description"]
                for product in batch
            ]

            print(
                "Embedding products",
                start + 1,
                "to",
                min(
                    start + EMBEDDING_BATCH_SIZE,
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

        dimensions = (
            matrix.shape[1]
        )

        self.index = (
            faiss.IndexFlatIP(
                dimensions
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
            "Knowledge base saved."
        )

    # ---------------------------------------------------------
    # Load existing knowledge base
    # ---------------------------------------------------------

    def load_index(self):

        print(
            "Loading existing FAISS index..."
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

    # ---------------------------------------------------------
    # Initialize
    # ---------------------------------------------------------

    def initialize(self):

        if (
            os.path.exists(INDEX_FILE)
            and
            os.path.exists(METADATA_FILE)
        ):

            self.load_index()

        else:

            self.build_index()

    # ---------------------------------------------------------
    # Retrieve similar products
    # ---------------------------------------------------------

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

        query_vector = np.array(
            [embedding],
            dtype="float32",
        )

        faiss.normalize_L2(
            query_vector
        )

        (
            similarities,
            indexes,
        ) = self.index.search(
            query_vector,
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

        self.client = OpenAI()

        self.knowledge_base = (
            knowledge_base
        )

    # ---------------------------------------------------------
    # Convert GPT response into number
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

    # ---------------------------------------------------------
    # Convert retrieved products into prompt context
    # ---------------------------------------------------------

    def build_context(
        self,
        retrieved_products,
    ):

        lines = []

        for (
            position,
            product,
        ) in enumerate(
            retrieved_products,
            start=1,
        ):

            lines.append(
                (
                    f"Comparable {position}\n"
                    f"Price: ${product['price']:.2f}\n"
                    f"Similarity: "
                    f"{product['similarity']:.4f}\n"
                    f"Description: "
                    f"{product['description']}"
                )
            )

        return "\n\n".join(
            lines
        )

    # ---------------------------------------------------------
    # Estimate product price
    # ---------------------------------------------------------

    def price(
        self,
        description,
    ):

        # -----------------------------------------------------
        # Step 1:
        # retrieve evidence
        # -----------------------------------------------------

        retrieved_products = (
            self.knowledge_base.search(
                description,
                TOP_K,
            )
        )

        # -----------------------------------------------------
        # Simple retrieval baseline
        # -----------------------------------------------------

        average_price = (
            sum(
                product["price"]
                for product
                in retrieved_products
            )
            /
            len(
                retrieved_products
            )
        )

        # -----------------------------------------------------
        # Step 2:
        # build RAG context
        # -----------------------------------------------------

        context = (
            self.build_context(
                retrieved_products
            )
        )

        # -----------------------------------------------------
        # Step 3:
        # ask GPT-5.1 to reason over retrieved evidence
        # -----------------------------------------------------

        prompt = f"""
You are an expert product price estimator.

Estimate the fair market price of the TARGET PRODUCT.

Use the comparable products below as evidence.
Consider product quality, features, category, and differences
between the target and the retrieved products.

TARGET PRODUCT:

{description}


COMPARABLE PRODUCTS:

{context}


Return ONLY the estimated price in US dollars.

Example output:

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

            "retrieval_average_price":
                average_price,

            "retrieved_products":
                retrieved_products,
        }


# =============================================================
# MAIN
# =============================================================

def main():

    if not os.environ.get(
        "OPENAI_API_KEY"
    ):

        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    # ---------------------------------------------------------
    # Initialize RAG
    # ---------------------------------------------------------

    knowledge_base = (
        ProductKnowledgeBase()
    )

    knowledge_base.initialize()

    # ---------------------------------------------------------
    # Initialize FrontierAgent
    # ---------------------------------------------------------

    frontier_agent = (
        FrontierAgent(
            knowledge_base
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
        frontier_agent.price(
            description
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "FRONTIER AGENT RESULT"
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
        "Retrieval average price:",
        round(
            result[
                "retrieval_average_price"
            ],
            2,
        )
    )

    print(
        "GPT-5.1 estimated price:",
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

    print()

    print(
        "RETRIEVED EVIDENCE"
    )

    print(
        "-" * 70
    )

    for (
        position,
        product,
    ) in enumerate(
        result[
            "retrieved_products"
        ],
        start=1,
    ):

        print()

        print(
            "Comparable:",
            position
        )

        print(
            "Price:",
            product[
                "price"
            ]
        )

        print(
            "Similarity:",
            round(
                product[
                    "similarity"
                ],
                4,
            )
        )

        print(
            "Description:",
            product[
                "description"
            ][:250]
        )


if __name__ == "__main__":
    main()