import modal


# =============================================================
# CONFIGURATION
# =============================================================

APP_NAME = (
    "week8-ensemble-agent"
)

SPECIALIST_APP = (
    "week8-specialist-agent"
)

SPECIALIST_CLASS = (
    "SpecialistModel"
)

FRONTIER_APP = (
    "week8-frontier-agent"
)

FRONTIER_CLASS = (
    "FrontierAgent"
)


SPECIALIST_WEIGHT = 0.50

FRONTIER_WEIGHT = 0.50


# =============================================================
# MODAL APP
# =============================================================

app = modal.App(
    APP_NAME
)


# =============================================================
# ENSEMBLE AGENT
# =============================================================

@app.cls(
    timeout=600,
)
class EnsembleAgent:

    @modal.enter()
    def initialize(self):

        # -----------------------------------------------------
        # Obtain references to already deployed Modal classes.
        #
        # No model is loaded in this container.
        #
        # Specialist owns GPU resources.
        # Frontier owns RAG/OpenAI resources.
        # -----------------------------------------------------

        SpecialistModel = (
            modal.Cls.from_name(

                SPECIALIST_APP,

                SPECIALIST_CLASS,
            )
        )

        FrontierModel = (
            modal.Cls.from_name(

                FRONTIER_APP,

                FRONTIER_CLASS,
            )
        )

        self.specialist = (
            SpecialistModel()
        )

        self.frontier = (
            FrontierModel()
        )

        print(
            "EnsembleAgent ready."
        )

    # ---------------------------------------------------------
    # Remote ensemble prediction
    # ---------------------------------------------------------

    @modal.method()
    def price(
        self,
        description,
    ):

        specialist_result = (
            self.specialist
            .price
            .remote(
                description
            )
        )

        frontier_result = (
            self.frontier
            .price
            .remote(
                description
            )
        )

        specialist_price = (
            specialist_result[
                "price"
            ]
        )

        frontier_price = (
            frontier_result[
                "price"
            ]
        )

        if (
            specialist_price is None
            and
            frontier_price is None
        ):

            final_price = None

        elif specialist_price is None:

            final_price = (
                frontier_price
            )

        elif frontier_price is None:

            final_price = (
                specialist_price
            )

        else:

            final_price = (

                SPECIALIST_WEIGHT
                * specialist_price

                +

                FRONTIER_WEIGHT
                * frontier_price
            )

        return {

            "ensemble_price":
                final_price,

            "specialist_price":
                specialist_price,

            "frontier_price":
                frontier_price,
        }


# =============================================================
# LOCAL TEST CLIENT
# =============================================================

@app.local_entrypoint()
def main():

    description = (
        "Premium stainless steel espresso "
        "machine with integrated milk frother, "
        "programmable drink settings and "
        "temperature control"
    )

    ensemble = (
        EnsembleAgent()
    )

    result = (
        ensemble
        .price
        .remote(
            description
        )
    )

    print(
        result
    )


if __name__ == "__main__":
    pass