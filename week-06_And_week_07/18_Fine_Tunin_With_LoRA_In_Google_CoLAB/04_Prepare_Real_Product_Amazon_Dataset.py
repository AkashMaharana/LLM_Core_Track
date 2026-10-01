import json
import os
import random
import re

import requests


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

DATA_URL = (
    "https://huggingface.co/datasets/"
    "McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/"
    "meta_Appliances.jsonl"
)


TRAIN_SIZE = 500
VALIDATION_SIZE = 50
TEST_SIZE = 250


TOTAL_REQUIRED = (
    TRAIN_SIZE
    + VALIDATION_SIZE
    + TEST_SIZE
)


MIN_PRICE = 0.50
MAX_PRICE = 999.49


MAX_DESCRIPTION_CHARS = 2500


RANDOM_SEED = 42


TRAIN_FILE = "amazon_price_train.jsonl"

VALIDATION_FILE = (
    "amazon_price_validation.jsonl"
)

TEST_FILE = "amazon_price_test.jsonl"


# -------------------------------------------------------------
# Parse price
# -------------------------------------------------------------

def parse_price(value):

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    text = text.replace(
        "$",
        "",
    )

    text = text.replace(
        ",",
        "",
    )

    match = re.search(
        r"\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    try:

        return float(
            match.group()
        )

    except ValueError:

        return None


# -------------------------------------------------------------
# Clean text
# -------------------------------------------------------------

def clean_text(value):

    if value is None:
        return ""

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# -------------------------------------------------------------
# Convert list/string into normal text
# -------------------------------------------------------------

def join_text(value):

    if value is None:
        return ""

    if isinstance(
        value,
        list,
    ):

        cleaned_items = []

        for item in value:

            cleaned = clean_text(
                item
            )

            if cleaned:

                cleaned_items.append(
                    cleaned
                )

        return " ".join(
            cleaned_items
        )

    return clean_text(
        value
    )


# -------------------------------------------------------------
# Build product text
#
# We use:
#
# title
# +
# features
# +
# description
# -------------------------------------------------------------

def build_product_text(product):

    title = clean_text(
        product.get(
            "title"
        )
    )

    features = join_text(
        product.get(
            "features"
        )
    )

    description = join_text(
        product.get(
            "description"
        )
    )

    parts = []

    if title:

        parts.append(
            title
        )

    if features:

        parts.append(
            features
        )

    if description:

        parts.append(
            description
        )

    combined = " ".join(
        parts
    )

    combined = clean_text(
        combined
    )

    # ---------------------------------------------------------
    # Character-level truncation.
    #
    # Token-level truncation happens later during SFT.
    # ---------------------------------------------------------

    return combined[
        :MAX_DESCRIPTION_CHARS
    ]


# -------------------------------------------------------------
# Curate one Amazon product
# -------------------------------------------------------------

def curate_product(product):

    price = parse_price(
        product.get(
            "price"
        )
    )

    if price is None:

        return None

    if not (
        MIN_PRICE
        <= price
        <= MAX_PRICE
    ):

        return None

    description = build_product_text(
        product
    )

    # Very short descriptions contain little useful signal.
    if len(description) < 40:

        return None

    product_id = product.get(
        "parent_asin"
    )

    if product_id is None:

        product_id = ""

    return {
        "description": description,

        "price": round(
            price,
            2,
        ),

        "product_id": str(
            product_id
        ),
    }


# -------------------------------------------------------------
# Write JSONL
# -------------------------------------------------------------

def write_jsonl(
    filename,
    products,
):

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:

        for product in products:

            json_line = json.dumps(
                product,
                ensure_ascii=False,
            )

            file.write(
                json_line
                + "\n"
            )


# -------------------------------------------------------------
# Reservoir sampling
#
# WHY?
#
# We do NOT want to:
#
# 1. Load the complete 285 MB dataset into memory.
# 2. Simply take the first 800 products.
#
# Reservoir sampling lets us read the dataset as a stream while
# keeping only 800 randomly selected valid products in memory.
# -------------------------------------------------------------

def collect_products():

    rng = random.Random(
        RANDOM_SEED
    )

    sampled_products = []

    valid_count = 0
    raw_count = 0

    # Metadata should normally contain one row per parent ASIN,
    # but we still guard against duplicates.
    seen_product_ids = set()

    print(
        "Connecting to Amazon Reviews dataset..."
    )

    print(
        "Source:",
        DATA_URL
    )

    # ---------------------------------------------------------
    # stream=True:
    #
    # requests does not download the entire response into
    # memory before we start processing it.
    # ---------------------------------------------------------

    with requests.get(
        DATA_URL,
        stream=True,
        timeout=300,
    ) as response:

        response.raise_for_status()

        print(
            "Connected successfully."
        )

        print(
            "Streaming and curating products..."
        )

        # -----------------------------------------------------
        # Each physical line is one JSON product.
        # -----------------------------------------------------

        for line in response.iter_lines(
            decode_unicode=True
        ):

            if not line:
                continue

            raw_count += 1

            try:

                raw_product = json.loads(
                    line
                )

            except json.JSONDecodeError:

                continue

            curated = curate_product(
                raw_product
            )

            if curated is None:

                continue

            product_id = curated[
                "product_id"
            ]

            if product_id:

                if (
                    product_id
                    in seen_product_ids
                ):

                    continue

                seen_product_ids.add(
                    product_id
                )

            # -------------------------------------------------
            # Number of valid products encountered so far.
            # -------------------------------------------------

            valid_count += 1

            # -------------------------------------------------
            # First fill the reservoir with 800 products.
            # -------------------------------------------------

            if (
                len(sampled_products)
                < TOTAL_REQUIRED
            ):

                sampled_products.append(
                    curated
                )

            else:

                # ---------------------------------------------
                # Reservoir sampling:
                #
                # Choose a random position from all valid
                # examples seen so far.
                #
                # If it falls inside our reservoir size,
                # replace one existing sample.
                # ---------------------------------------------

                position = rng.randint(
                    1,
                    valid_count,
                )

                if (
                    position
                    <= TOTAL_REQUIRED
                ):

                    sampled_products[
                        position - 1
                    ] = curated

            if (
                raw_count % 50_000
                == 0
            ):

                print(
                    "Rows inspected:",
                    raw_count,
                    "| Valid products:",
                    valid_count,
                )

    print(
        "\nFinished reading dataset."
    )

    print(
        "Raw rows inspected:",
        raw_count,
    )

    print(
        "Valid priced products:",
        valid_count,
    )

    if (
        len(sampled_products)
        < TOTAL_REQUIRED
    ):

        raise RuntimeError(
            "Not enough valid products found. "
            f"Required {TOTAL_REQUIRED}, "
            f"found {len(sampled_products)}."
        )

    # ---------------------------------------------------------
    # Shuffle the final 800 selected products before splitting.
    # ---------------------------------------------------------

    rng.shuffle(
        sampled_products
    )

    return sampled_products


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():

    products = collect_products()

    # ---------------------------------------------------------
    # Create splits
    # ---------------------------------------------------------

    train_end = TRAIN_SIZE

    validation_end = (
        TRAIN_SIZE
        + VALIDATION_SIZE
    )

    train_products = products[
        :train_end
    ]

    validation_products = products[
        train_end:
        validation_end
    ]

    test_products = products[
        validation_end:
    ]

    # ---------------------------------------------------------
    # Save splits
    # ---------------------------------------------------------

    write_jsonl(
        TRAIN_FILE,
        train_products,
    )

    write_jsonl(
        VALIDATION_FILE,
        validation_products,
    )

    write_jsonl(
        TEST_FILE,
        test_products,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATASET PREPARATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Training products:",
        len(
            train_products
        ),
    )

    print(
        "Validation products:",
        len(
            validation_products
        ),
    )

    print(
        "Test products:",
        len(
            test_products
        ),
    )

    print(
        "\nFILES"
    )

    print(
        "Train:",
        os.path.abspath(
            TRAIN_FILE
        )
    )

    print(
        "Validation:",
        os.path.abspath(
            VALIDATION_FILE
        )
    )

    print(
        "Test:",
        os.path.abspath(
            TEST_FILE
        )
    )

    # ---------------------------------------------------------
    # Print one example from each split.
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAIN EXAMPLE"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            train_products[0],
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION EXAMPLE"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            validation_products[0],
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST EXAMPLE"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            test_products[0],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()