import json
import os

import faiss
import numpy as np

from openai import OpenAI


# =============================================================
# CONFIGURATION
# =============================================================

TRAIN_FILE = "amazon_price_train.jsonl"

EMBEDDING_MODEL = "text-embedding-3-small"

INDEX_FILE = "product_price.faiss"

METADATA_FILE = "product_price_metadata.json"

TOP_K = 5

EMBEDDING_BATCH_SIZE = 100


# =============================================================
# RAG KNOWLEDGE BASE
# =============================================================

class ProductKnowledgeBase:

    def __init__(self):

        self.client = OpenAI()

        self.index = None

        self.products = []

    # ---------------------------------------------------------
    # Load raw product data
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
    # Generate embeddings for a list of texts
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

        embeddings = [

            item.embedding

            for item in response.data
        ]

        return embeddings

    # ---------------------------------------------------------
    # Build FAISS index
    # ---------------------------------------------------------

    def build_index(self):

        print(
            "Loading training products..."
        )

        self.products = (
            self.load_products()
        )

        print(
            "Products:",
            len(self.products)
        )

        all_embeddings = []

        # -----------------------------------------------------
        # Create embeddings in batches rather than one request
        # per product.
        # -----------------------------------------------------

        for start in range(
            0,
            len(self.products),
            EMBEDDING_BATCH_SIZE,
        ):

            end = (
                start
                + EMBEDDING_BATCH_SIZE
            )

            batch = self.products[
                start:end
            ]

            descriptions = [

                product["description"]

                for product in batch
            ]

            print(
                f"Embedding "
                f"{start + 1}"
                f"-"
                f"{min(end, len(self.products))}"
            )

            embeddings = (
                self.create_embeddings(
                    descriptions
                )
            )

            all_embeddings.extend(
                embeddings
            )

        # -----------------------------------------------------
        # Convert Python lists into NumPy matrix.
        #
        # FAISS expects float32 vectors.
        # -----------------------------------------------------

        matrix = np.array(
            all_embeddings,
            dtype="float32",
        )

        # -----------------------------------------------------
        # Normalize vectors.
        #
        # After normalization:
        #
        # inner product ≈ cosine similarity
        # -----------------------------------------------------

        faiss.normalize_L2(
            matrix
        )

        dimensions = (
            matrix.shape[1]
        )

        # -----------------------------------------------------
        # Exact vector search using inner product.
        # -----------------------------------------------------

        self.index = (
            faiss.IndexFlatIP(
                dimensions
            )
        )

        self.index.add(
            matrix
        )

        # -----------------------------------------------------
        # Persist index
        # -----------------------------------------------------

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
            "Index saved."
        )

    # ---------------------------------------------------------
    # Load an existing index
    # ---------------------------------------------------------

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
                json.load(file)
            )

        print(
            "Existing index loaded."
        )

        print(
            "Products:",
            len(self.products)
        )

    # ---------------------------------------------------------
    # Build only when index does not already exist
    # ---------------------------------------------------------

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

            self.load_index()

        else:

            self.build_index()

    # ---------------------------------------------------------
    # Search for similar products
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

        similarities, indexes = (
            self.index.search(
                query_vector,
                top_k,
            )
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
                        product[
                            "price"
                        ],

                    "product_id":
                        product.get(
                            "product_id",
                            "",
                        ),

                    "similarity":
                        float(
                            similarity
                        ),
                }
            )

        return results


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

    knowledge_base = (
        ProductKnowledgeBase()
    )

    knowledge_base.initialize()

    query = (
        "Premium stainless steel espresso "
        "machine with milk frother and "
        "programmable drink settings"
    )

    print()

    print(
        "=" * 70
    )

    print(
        "QUERY"
    )

    print(
        "=" * 70
    )

    print(
        query
    )

    results = (
        knowledge_base.search(
            query
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "RETRIEVED PRODUCTS"
    )

    print(
        "=" * 70
    )

    for position, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(
            "Result:",
            position
        )

        print(
            "Known price:",
            result[
                "price"
            ]
        )

        print(
            "Similarity:",
            round(
                result[
                    "similarity"
                ],
                4,
            )
        )

        print(
            "Description:",
            result[
                "description"
            ][:300]
        )


if __name__ == "__main__":
    main()