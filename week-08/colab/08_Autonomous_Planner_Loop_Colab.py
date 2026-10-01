import json
import os

from openai import OpenAI


# =============================================================
# CONFIGURATION
# =============================================================

MODEL = "gpt-5.1"

MAX_AGENT_STEPS = 10


# =============================================================
# SAMPLE DATA
#
# For Part 43 we deliberately isolate the Planner mechanics.
#
# Part 44 will replace these with our real:
#
# ScannerAgent
# EnsembleAgent
# MessengerAgent
# =============================================================

SAMPLE_DEALS = [
    {
        "description":
            "55 inch OLED 4K HDR Smart TV "
            "with Dolby Vision and Dolby Atmos",

        "deal_price":
            599.00,

        "url":
            "https://example.com/tv",
    },

    {
        "description":
            "Premium espresso machine with "
            "integrated milk frother and "
            "temperature control",

        "deal_price":
            179.00,

        "url":
            "https://example.com/espresso",
    },

    {
        "description":
            "Wireless noise cancelling headphones "
            "with 30 hour battery life",

        "deal_price":
            149.00,

        "url":
            "https://example.com/headphones",
    },
]


# =============================================================
# TOOL IMPLEMENTATIONS
# =============================================================

def scan_for_deals():

    print()
    print("TOOL: scan_for_deals")

    return {
        "deals":
            SAMPLE_DEALS
    }


def estimate_true_value(
    description,
):

    print()
    print(
        "TOOL: estimate_true_value"
    )

    print(
        "Product:",
        description
    )

    # ---------------------------------------------------------
    # Small deterministic lookup only for this planner lesson.
    #
    # Part 44 replaces this with EnsembleAgent.
    # ---------------------------------------------------------

    description_lower = (
        description.lower()
    )

    if "oled" in description_lower:

        estimated_value = 950.00

    elif "espresso" in description_lower:

        estimated_value = 420.00

    elif "headphones" in description_lower:

        estimated_value = 280.00

    else:

        estimated_value = 100.00

    return {
        "description":
            description,

        "estimated_true_value":
            estimated_value,
    }


def notify_user_of_deal(
    description,
    deal_price,
    estimated_true_value,
    url,
):

    print()
    print(
        "TOOL: notify_user_of_deal"
    )

    discount = (
        estimated_true_value
        - deal_price
    )

    print(
        "Description:",
        description
    )

    print(
        "Deal price:",
        deal_price
    )

    print(
        "Estimated value:",
        estimated_true_value
    )

    print(
        "Saving:",
        discount
    )

    print(
        "URL:",
        url
    )

    # ---------------------------------------------------------
    # Preview only.
    #
    # No real push notification yet.
    # ---------------------------------------------------------

    return {
        "status":
            "notification_preview_created",

        "description":
            description,

        "deal_price":
            deal_price,

        "estimated_true_value":
            estimated_true_value,

        "saving":
            discount,

        "url":
            url,
    }


# =============================================================
# TOOL DEFINITIONS
#
# These are contracts shown to GPT-5.1.
#
# Similar idea to Java interfaces / API schemas.
# =============================================================

TOOLS = [

    {
        "type":
            "function",

        "name":
            "scan_for_deals",

        "description":
            (
                "Scan for current candidate product deals. "
                "Returns product descriptions, selling prices "
                "and URLs."
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
                "Estimate the true fair value of one product. "
                "Use this for candidate deals before deciding "
                "which deal is best."
            ),

        "parameters": {
            "type":
                "object",

            "properties": {

                "description": {
                    "type":
                        "string",

                    "description":
                        (
                            "Full product description "
                            "from the scanner."
                        ),
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
                "Notify the user about the single best bargain. "
                "Call this at most once, after evaluating the "
                "candidate products."
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


# =============================================================
# TOOL ROUTER
#
# This maps LLM tool names to actual Python functions.
# =============================================================

def execute_tool(
    name,
    arguments,
):

    if name == "scan_for_deals":

        return scan_for_deals()


    if name == "estimate_true_value":

        return estimate_true_value(
            **arguments
        )


    if name == "notify_user_of_deal":

        return notify_user_of_deal(
            **arguments
        )


    raise ValueError(
        f"Unknown tool: {name}"
    )


# =============================================================
# AUTONOMOUS PLANNER AGENT
# =============================================================

class AutonomousPlannerAgent:

    def __init__(self):

        self.client = OpenAI()


    def run(self):

        # -----------------------------------------------------
        # This list becomes the agent's working conversation.
        # -----------------------------------------------------

        conversation = [

            {
                "role":
                    "system",

                "content":
                    (
                        "You are a deal-finding planner agent. "
                        "Your goal is to find the single best "
                        "bargain and notify the user once. "
                        "\n\n"
                        "First scan for deals. "
                        "Estimate the true value of candidate "
                        "products. Compare estimated value with "
                        "selling price. "
                        "Choose the deal with the largest "
                        "meaningful saving. "
                        "Then notify the user about that one "
                        "deal only. "
                        "After notification, finish."
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    (
                        "Find the best bargain currently "
                        "available and notify me."
                    ),
            },
        ]


        # -----------------------------------------------------
        # AGENT LOOP
        # -----------------------------------------------------

        for step in range(
            1,
            MAX_AGENT_STEPS + 1,
        ):

            print()
            print(
                "=" * 70
            )

            print(
                "AGENT STEP",
                step
            )

            print(
                "=" * 70
            )

            response = (
                self.client.responses.create(

                    model=MODEL,

                    input=conversation,

                    tools=TOOLS,
                )
            )


            # -------------------------------------------------
            # Important:
            #
            # preserve everything GPT produced.
            #
            # Reasoning models may return items that need to
            # remain in the continuing interaction.
            # -------------------------------------------------

            conversation += (
                response.output
            )


            # -------------------------------------------------
            # Find requested function calls.
            # -------------------------------------------------

            function_calls = [

                item

                for item in response.output

                if item.type
                == "function_call"
            ]


            # -------------------------------------------------
            # No tool call means the agent has finished.
            # -------------------------------------------------

            if not function_calls:

                print()
                print(
                    "AGENT FINISHED"
                )

                print(
                    response.output_text
                )

                return


            # -------------------------------------------------
            # Execute every requested tool.
            #
            # The API may return more than one function call.
            # -------------------------------------------------

            for tool_call in function_calls:

                tool_name = (
                    tool_call.name
                )

                arguments = (
                    json.loads(
                        tool_call.arguments
                    )
                )

                print()

                print(
                    "Planner requested:",
                    tool_name
                )

                print(
                    "Arguments:",
                    arguments
                )


                # ---------------------------------------------
                # OUR APPLICATION executes the function.
                # GPT does not execute Python directly.
                # ---------------------------------------------

                result = (
                    execute_tool(
                        tool_name,
                        arguments,
                    )
                )


                print(
                    "Tool result:",
                    result
                )


                # ---------------------------------------------
                # Feed observation back to the model.
                #
                # call_id correlates:
                #
                # function request
                #       ↕
                # function result
                # ---------------------------------------------

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
            "Planner exceeded maximum "
            f"{MAX_AGENT_STEPS} steps."
        )


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

    planner = (
        AutonomousPlannerAgent()
    )

    planner.run()


if __name__ == "__main__":
    main()