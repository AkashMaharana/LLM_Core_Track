import json

import modal


# =============================================================
# CONFIGURATION
# =============================================================

APP_NAME = (
    "week8-autonomous-planner"
)

MODEL = (
    "gpt-5.1"
)

MAX_AGENT_STEPS = 10


# =============================================================
# APP
# =============================================================

app = modal.App(
    APP_NAME
)


openai_secret = (
    modal.Secret.from_name(
        "openai-secret",
        required_keys=[
            "OPENAI_API_KEY"
        ],
    )
)


image = (

    modal.Image
    .debian_slim(
        python_version="3.11"
    )

    .uv_pip_install(
        "openai"
    )
)


# =============================================================
# AUTONOMOUS PLANNER
# =============================================================

@app.cls(

    image=image,

    secrets=[
        openai_secret
    ],

    timeout=600,
)
class AutonomousPlannerAgent:

    @modal.enter()
    def initialize(self):

        from openai import OpenAI

        self.client = (
            OpenAI()
        )

        # -----------------------------------------------------
        # In Part 44 these become references to deployed:
        #
        # ScannerAgent
        # EnsembleAgent
        # MessengerAgent
        # -----------------------------------------------------

        print(
            "AutonomousPlannerAgent ready."
        )


    def get_tools(self):

        return [

            {
                "type":
                    "function",

                "name":
                    "scan_for_deals",

                "description":
                    (
                        "Scan for candidate "
                        "product deals."
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
                        "Estimate the fair value "
                        "of a product."
                    ),

                "parameters": {
                    "type":
                        "object",

                    "properties": {

                        "description": {
                            "type":
                                "string"
                        }
                    },

                    "required": [
                        "description"
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
                        "the single best deal."
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


    # ---------------------------------------------------------
    # Temporary tool implementations for Planner mechanics.
    # ---------------------------------------------------------

    def execute_tool(
        self,
        name,
        arguments,
    ):

        if name == "scan_for_deals":

            return {
                "deals": [
                    {
                        "description":
                            "55 inch OLED TV",

                        "deal_price":
                            599.0,

                        "url":
                            "https://example.com/tv",
                    },

                    {
                        "description":
                            "Premium espresso machine",

                        "deal_price":
                            179.0,

                        "url":
                            "https://example.com/espresso",
                    },
                ]
            }


        if name == "estimate_true_value":

            description = (
                arguments[
                    "description"
                ]
            )

            if (
                "OLED"
                in description
            ):

                value = 950.0

            else:

                value = 420.0

            return {
                "estimated_true_value":
                    value
            }


        if name == "notify_user_of_deal":

            return {
                "status":
                    "notification_preview_created",

                **arguments,
            }


        raise ValueError(
            f"Unknown tool: {name}"
        )


    @modal.method()
    def run(self):

        conversation = [

            {
                "role":
                    "system",

                "content":
                    (
                        "Find the single best "
                        "bargain using your tools. "
                        "Scan first, estimate candidate "
                        "values, notify the user about "
                        "one best opportunity only, "
                        "then finish."
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    (
                        "Find the best bargain "
                        "and notify me."
                    ),
            },
        ]


        for _ in range(
            MAX_AGENT_STEPS
        ):

            response = (
                self.client.responses.create(

                    model=MODEL,

                    input=conversation,

                    tools=(
                        self.get_tools()
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

                return {
                    "final_response":
                        response.output_text
                }


            for tool_call in function_calls:

                arguments = (
                    json.loads(
                        tool_call.arguments
                    )
                )

                result = (
                    self.execute_tool(

                        tool_call.name,

                        arguments,
                    )
                )


                conversation.append(
                    {
                        "type":
                            "function_call_output",

                        "call_id":
                            tool_call.call_id,

                        "output":
                            json.dumps(
                                result
                            ),
                    }
                )


        raise RuntimeError(
            "Maximum planner steps exceeded."
        )


# =============================================================
# LOCAL ENTRYPOINT
# =============================================================

@app.local_entrypoint()
def main():

    planner = (
        AutonomousPlannerAgent()
    )

    result = (
        planner
        .run
        .remote()
    )

    print(
        result
    )


if __name__ == "__main__":
    pass