import modal


# =============================================================
# CONFIGURATION
# =============================================================

APP_NAME = (
    "week8-product-rag"
)

REMOTE_DATA_PATH = (
    "/data"
)

INDEX_FILE = (
    "/data/product_price.faiss"
)

METADATA_FILE = (
    "/data/product_price_metadata.json"
)

EMBEDDING_MODEL = (
    "text-embedding-3-small"
)


# =============================================================
# MODAL APP
# =============================================================

app = modal.App(
    APP_NAME
)


# =============================================================
# PERSISTENT VECTOR DATA
# =============================================================

data_volume = (
    modal.Volume.from_name(

        "week8-product-rag-data",

        create_if_missing=True,
    )
)


# =============================================================
# IMAGE
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
)


# =============================================================
# REMOTE RAG SERVICE
# =============================================================

@app.cls(

    image=image,

    volumes={
        REMOTE_DATA_PATH:
            data_volume
    },

    secrets=[
        modal.Secret.from_name(
            "openai-secret"
        )
    ],

    timeout=300,
)
class ProductKnowledgeBase:

    # ---------------------------------------------------------
    # Load index ONCE per container.
    # ---------------------------------------------------------

    @modal.enter()
    def load(self):

        import json

        import faiss

        from openai import OpenAI

        self.faiss = faiss

        self.client = OpenAI()

        print(
            "Loading FAISS index..."
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
            "RAG knowledge base ready."
        )

    # ---------------------------------------------------------
    # Remote search
    # ---------------------------------------------------------

    @modal.method()
    def search(
        self,
        query,
        top_k=5,
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

        query_vector = np.array(
            [embedding],
            dtype="float32",
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
                        product[
                            "price"
                        ],

                    "similarity":
                        float(
                            similarity
                        ),
                }
            )

        return results


# =============================================================
# LOCAL TEST CLIENT
# =============================================================

@app.local_entrypoint()
def main():

    query = (
        "Premium stainless steel "
        "espresso machine with milk frother"
    )

    rag = (
        ProductKnowledgeBase()
    )

    results = (
        rag.search.remote(
            query,
            5,
        )
    )

    for result in results:

        print(
            result
        )


if __name__ == "__main__":
    pass