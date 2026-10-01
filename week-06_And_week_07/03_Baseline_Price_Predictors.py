import random


def main():

    # ---------------------------------------------------------
    # Our tiny training dataset
    # ---------------------------------------------------------

    train_products = [
        {
            "name": "USB Cable",
            "price": 9.99
        },
        {
            "name": "Desk Fan",
            "price": 24.99
        },
        {
            "name": "Coffee Maker",
            "price": 129.99
        },
        {
            "name": "Headphones",
            "price": 249.00
        },
        {
            "name": "Monitor",
            "price": 399.00
        },
    ]

    # ---------------------------------------------------------
    # Test products
    # ---------------------------------------------------------

    test_products = [
        {
            "name": "Keyboard",
            "price": 79.00
        },
        {
            "name": "Laptop",
            "price": 799.00
        },
        {
            "name": "Portable Speaker",
            "price": 129.00
        },
    ]

    # ---------------------------------------------------------
    # RANDOM PRICER
    # ---------------------------------------------------------

    print("\nRANDOM PRICER")
    print("=" * 60)

    for product in test_products:

        prediction = random.uniform(
            1,
            1000
        )

        error = abs(
            product["price"] - prediction
        )

        print(
            product["name"],
            "| Actual:",
            round(product["price"], 2),
            "| Predicted:",
            round(prediction, 2),
            "| Error:",
            round(error, 2)
        )

    # ---------------------------------------------------------
    # CONSTANT PRICER
    # ---------------------------------------------------------

    train_prices = [
        product["price"]
        for product in train_products
    ]

    average_price = (
        sum(train_prices)
        /
        len(train_prices)
    )

    print("\nCONSTANT PRICER")
    print("=" * 60)

    print(
        "Average training price:",
        round(average_price, 2)
    )

    for product in test_products:

        prediction = average_price

        error = abs(
            product["price"] - prediction
        )

        print(
            product["name"],
            "| Actual:",
            round(product["price"], 2),
            "| Predicted:",
            round(prediction, 2),
            "| Error:",
            round(error, 2)
        )


if __name__ == "__main__":
    main()