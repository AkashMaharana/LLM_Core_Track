import modal


APP_NAME = (
    "week8-deal-agent-ui"
)


app = modal.App(
    APP_NAME
)


web_image = (
    modal.Image
    .debian_slim(
        python_version="3.11"
    )
    .uv_pip_install(
        "fastapi[standard]",
        "gradio",
    )
)


@app.function(
    image=web_image,
    max_containers=1,
)
@modal.concurrent(
    max_inputs=20
)
@modal.asgi_app()
def ui():

    import gradio as gr

    from fastapi import FastAPI

    # --------------------------------------------------------
    # Reference the independently deployed planner.
    # --------------------------------------------------------

    Planner = (
        modal.Cls.from_name(
            "week8-deal-agent-framework",
            "AutonomousPlannerAgent",
        )
    )

    planner = Planner()


    def run_agent():

        try:

            result = (
                planner
                .run
                .remote()
            )

            return (
                str(
                    result
                )
            )

        except Exception as error:

            return (
                f"{type(error).__name__}: "
                f"{error}"
            )


    with gr.Blocks(
        title="Autonomous Deal Finder"
    ) as demo:

        gr.Markdown(
            """
# Autonomous Deal Finder

Run the deployed autonomous deal-finding framework.
"""
        )

        run_button = (
            gr.Button(
                "Find Best Deal",
                variant="primary",
            )
        )

        output = (
            gr.Textbox(
                label="Agent Result",
                lines=15,
            )
        )

        run_button.click(
            fn=run_agent,
            outputs=output,
        )


    fastapi_app = (
        FastAPI()
    )

    return (
        gr.mount_gradio_app(
            fastapi_app,
            demo,
            path="/",
        )
    )