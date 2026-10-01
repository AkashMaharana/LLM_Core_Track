import random
from collections import defaultdict


def main():

    random.seed(42)

    products = [
        {"name": "USB Cable", "price": 9.99},
        {"name": "Desk Fan", "price": 24.99},
        {"name": "Coffee Maker", "price": 129.99},
        {"name": "Headphones", "price": 249.00},
        {"name": "Monitor", "price": 399.00},
        {"name": "TV", "price": 699.00},
        {"name": "Laptop", "price": 899.00},
        {"name": "Invalid Cheap", "price": 0.10},
        {"name": "Invalid Expensive", "price": 2500.00},
    ]

    # ---------------------------------------------------------
    # 1. Price-range filtering
    # ---------------------------------------------------------

    filtered = [
        product
        for product in products
        if 0.50 <= product["price"] <= 999.49
    ]

    # ---------------------------------------------------------
    # 2. Put products into price buckets
    # ---------------------------------------------------------

    buckets = defaultdict(list)

    for product in filtered:

        bucket = int(product["price"])

        buckets[bucket].append(product)

    print("\nPRICE BUCKETS")
    print("=" * 60)

    for bucket, items in buckets.items():

        print(
            bucket,
            "→",
            [item["name"] for item in items]
        )

    # ---------------------------------------------------------
    # 3. Shuffle reproducibly
    # ---------------------------------------------------------

    random.shuffle(filtered)

    # ---------------------------------------------------------
    # 4. Train / test split
    # ---------------------------------------------------------

    split_index = int(
        len(filtered) * 0.8
    )

    train = filtered[:split_index]
    test = filtered[split_index:]

    print("\nTRAIN")
    print("=" * 60)

    for product in train:
        print(product)

    print("\nTEST")
    print("=" * 60)

    for product in test:
        print(product)


if __name__ == "__main__":
    main()