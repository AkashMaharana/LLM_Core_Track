import modal


# -------------------------------------------------------------
# Create Modal application
# -------------------------------------------------------------

app = modal.App(
    "week8-modal-basics"
)


# -------------------------------------------------------------
# This function will execute in Modal infrastructure.
# -------------------------------------------------------------

@app.function()
def calculate_price_with_tax(
    price: float,
    tax_rate: float,
):

    print(
        "This calculation is running remotely."
    )

    final_price = (
        price
        + price * tax_rate
    )

    return final_price


# -------------------------------------------------------------
# This entrypoint itself runs locally.
# -------------------------------------------------------------

@app.local_entrypoint()
def main():

    print(
        "Calling Modal remote function..."
    )

    result = (
        calculate_price_with_tax.remote(
            100.0,
            0.18,
        )
    )

    print(
        "Remote result:",
        result,
    )


if __name__ == "__main__":

    # When using `modal run`, Modal invokes the
    # registered local_entrypoint automatically.
    #
    # We therefore don't directly call main() here.
    pass