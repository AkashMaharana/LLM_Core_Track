import modal


# =============================================================
# CONFIGURATION
# =============================================================

APP_NAME = (
    "week8-frontier-agent"
)

LOCAL_INDEX_FILE = (
    "product_price.faiss"
)

LOCAL_METADATA_FILE = (
    "product_price_metadata.json"
)

REMOTE_INDEX_FILE = (
    "/data/product_price.faiss"
)

REMOTE_METADATA_FILE = (
    "/data/product_price_metadata.json"
)

EMBEDDING_MODEL = (
    "text-embedding-3-small"
)

FRONTIER_MODEL = (
    "gpt-5.1"
)

TOP_K = 5


# modal secret create openai-secret OPENAI_API_KEY=YOUR_KEY

# =============================================================
# MODAL APP
# =============================================================

app = modal.App(
    APP_NAME
)


# =============================================================
# OPENAI SECRET
# =============================================================

openai_secret = (
    modal.Secret.from_name(
        "openai-secret",
        required_keys=[
            "OPENAI_API_KEY"
        ],
    )
)


# =============================================================
# CONTAINER IMAGE
# =============================================================

image = (

    modal.Image
    .debian_slim(
        python_version="3.11"
    )

    .uv_pip_install(
        "openai",
        "faiss-cpu",
        "numpy",
    )

    # ---------------------------------------------------------
    # Include the already-created FAISS artifacts.
    # ---------------------------------------------------------

    .add_local_file(
        LOCAL_INDEX_FILE,
        REMOTE_INDEX_FILE,
    )

    .add_local_file(
        LOCAL_METADATA_FILE,
        REMOTE_METADATA_FILE,
    )
)


# =============================================================
# FRONTIER AGENT
# =============================================================

@app.cls(

    image=image,

    secrets=[
        openai_secret
    ],

    timeout=300,
)
class FrontierAgent:

    # ---------------------------------------------------------
    # Load FAISS index once per container
    # ---------------------------------------------------------

    @modal.enter()
    def initialize(self):

        import json
        import re

        import faiss

        from openai import OpenAI

        self.faiss = faiss

        self.re = re

        self.client = OpenAI()

        self.index = (
            faiss.read_index(
                REMOTE_INDEX_FILE
            )
        )

        with open(
            REMOTE_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            self.products = (
                json.load(file)
            )

        print(
            "FrontierAgent ready."
        )

    # ---------------------------------------------------------
    # Parse numeric price
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
    # Retrieve similar products
    # ---------------------------------------------------------

    def retrieve(
        self,
        query,
        top_k,
    ):

        import numpy as np

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

        query_vector = (
            np.array(
                [embedding],
                dtype="float32",
            )
        )

        self.faiss.normalize_L2(
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

    # ---------------------------------------------------------
    # Build RAG prompt evidence
    # ---------------------------------------------------------

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
    # Remote agent method
    # ---------------------------------------------------------

    @modal.method()
    def price(
        self,
        description,
    ):

        retrieved_products = (
            self.retrieve(
                description,
                TOP_K,
            )
        )

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

        context = (
            self.build_context(
                retrieved_products
            )
        )

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
# LOCAL TEST CLIENT
# =============================================================

@app.local_entrypoint()
def main():

    description = (
        "Premium stainless steel espresso "
        "machine with integrated milk frother, "
        "programmable settings and adjustable "
        "temperature control"
    )

    frontier = (
        FrontierAgent()
    )

    result = (
        frontier
        .price
        .remote(
            description
        )
    )

    print(
        "Frontier price:",
        result[
            "price"
        ]
    )

    print(
        "Retrieval average:",
        result[
            "retrieval_average_price"
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