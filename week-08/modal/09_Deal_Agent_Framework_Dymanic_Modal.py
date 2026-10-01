import json

import modal


# ============================================================
# CONFIG
# ============================================================

APP_NAME = (
    "week8-deal-agent-framework"
)

MODEL = "gpt-5.1"

MAX_AGENT_STEPS = 12


# ============================================================
# APP
# ============================================================

app = modal.App(
    APP_NAME
)


# ============================================================
# SECRETS
# ============================================================

openai_secret = (
    modal.Secret.from_name(
        "openai-secret",
        required_keys=[
            "OPENAI_API_KEY"
        ],
    )
)


# ============================================================
# PERSISTENT MEMORY
# ============================================================

memory = (
    modal.Dict.from_name(
        "week8-deal-memory",
        create_if_missing=True,
    )
)


# ============================================================
# IMAGE
# ============================================================

image = (
    modal.Image
    .debian_slim(
        python_version="3.11"
    )
    .uv_pip_install(
        "openai"
    )
)


# ============================================================
# PLANNER
# ============================================================

@app.cls(
    image=image,
    secrets=[
        openai_secret
    ],
    timeout=900,
)
class AutonomousPlannerAgent:

    @modal.enter()
    def initialize(self):

        from openai import OpenAI

        self.client = OpenAI()

        # ----------------------------------------------------
        # References to separately deployed agents.
        # ----------------------------------------------------

        Scanner = (
            modal.Cls.from_name(
                "week8-scanner-agent",
                "ScannerAgent",
            )
        )

        Ensemble = (
            modal.Cls.from_name(
                "week8-ensemble-agent",
                "EnsembleAgent",
            )
        )

        Messenger = (
            modal.Cls.from_name(
                "week8-messenger-agent",
                "MessengerAgent",
            )
        )

        self.scanner = Scanner()
        self.ensemble = Ensemble()
        self.messenger = Messenger()

    def get_seen_urls(self):

        return memory.get(
            "seen_urls",
            [],
        )

    def get_notified_urls(self):

        return memory.get(
            "notified_urls",
            [],
        )

    def mark_seen(
        self,
        urls,
    ):

        current = set(
            self.get_seen_urls()
        )

        current.update(
            urls
        )

        memory["seen_urls"] = (
            list(current)
        )

    def mark_notified(
        self,
        url,
    ):

        current = set(
            self.get_notified_urls()
        )

        current.add(url)

        memory[
            "notified_urls"
        ] = list(current)

        self.mark_seen(
            [url]
        )

    def tools(self):

        return [
            {
                "type": "function",
                "name":
                    "scan_for_deals",

                "description":
                    "Find new candidate product deals.",

                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties":
                        False,
                },

                "strict": True,
            },

            {
                "type": "function",
                "name":
                    "estimate_true_value",

                "description":
                    (
                        "Estimate the fair market "
                        "value of a product."
                    ),

                "parameters": {
                    "type": "object",

                    "properties": {
                        "description": {
                            "type": "string"
                        }
                    },

                    "required": [
                        "description"
                    ],

                    "additionalProperties":
                        False,
                },

                "strict": True,
            },

            {
                "type": "function",
                "name":
                    "notify_user_of_deal",

                "description":
                    (
                        "Notify the user about the "
                        "single best bargain."
                    ),

                "parameters": {
                    "type": "object",

                    "properties": {
                        "description": {
                            "type": "string"
                        },

                        "deal_price": {
                            "type": "number"
                        },

                        "estimated_true_value": {
                            "type": "number"
                        },

                        "url": {
                            "type": "string"
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

                "strict": True,
            },
        ]

    def execute_tool(
        self,
        name,
        arguments,
    ):

        if name == "scan_for_deals":

            result = (
                self.scanner
                .scan
                .remote(
                    self.get_seen_urls()
                )
            )

            deals = result.get(
                "deals",
                []
            )

            self.mark_seen(
                [
                    deal["url"]
                    for deal in deals
                ]
            )

            return result

        if name == "estimate_true_value":

            result = (
                self.ensemble
                .price
                .remote(
                    arguments[
                        "description"
                    ]
                )
            )

            return {
                "estimated_true_value":
                    result[
                        "ensemble_price"
                    ]
            }

        if name == "notify_user_of_deal":

            url = arguments[
                "url"
            ]

            if (
                url
                in self.get_notified_urls()
            ):
                return {
                    "status":
                        "duplicate_ignored"
                }

            result = (
                self.messenger
                .notify
                .remote(
                    arguments[
                        "description"
                    ],

                    arguments[
                        "deal_price"
                    ],

                    arguments[
                        "estimated_true_value"
                    ],

                    url,
                )
            )

            self.mark_notified(
                url
            )

            return result

        raise ValueError(
            f"Unknown tool: {name}"
        )

    @modal.method()
    def run(self):

        conversation = [
            {
                "role": "system",

                "content":
                    (
                        "You are an autonomous deal "
                        "planner. Scan for new deals, "
                        "estimate candidate values, "
                        "select the single most "
                        "compelling bargain and notify "
                        "the user once. If no suitable "
                        "deal exists, finish."
                    ),
            },

            {
                "role": "user",

                "content":
                    (
                        "Find the best new bargain "
                        "and notify me."
                    ),
            },
        ]

        notification_sent = False

        for step in range(
            MAX_AGENT_STEPS
        ):

            response = (
                self.client.responses.create(
                    model=MODEL,
                    input=conversation,
                    tools=self.tools(),
                )
            )

            conversation += (
                response.output
            )

            tool_calls = [
                item
                for item in response.output
                if item.type
                == "function_call"
            ]

            if not tool_calls:

                return {
                    "result":
                        response.output_text,

                    "steps":
                        step + 1,
                }

            for call in tool_calls:

                arguments = (
                    json.loads(
                        call.arguments
                    )
                )

                if (
                    call.name
                    == "notify_user_of_deal"
                    and notification_sent
                ):

                    result = {
                        "status":
                            "second_notification_blocked"
                    }

                else:

                    result = (
                        self.execute_tool(
                            call.name,
                            arguments,
                        )
                    )

                    if (
                        call.name
                        == "notify_user_of_deal"
                    ):
                        notification_sent = True

                conversation.append(
                    {
                        "type":
                            "function_call_output",

                        "call_id":
                            call.call_id,

                        "output":
                            json.dumps(
                                result
                            ),
                    }
                )

        raise RuntimeError(
            "Planner step limit exceeded."
        )


# ============================================================
# TEST CLIENT
# ============================================================

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