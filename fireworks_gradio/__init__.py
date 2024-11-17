import os
from openai import OpenAI
import gradio as gr
from typing import Callable

__version__ = "0.0.3"


def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str):
    def fn(message, history):
        inputs = preprocess(message, history)
        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key
        )
        completion = client.completions.create(
            model="accounts/fireworks/models/" + model_name,
            prompt=inputs["prompt"],
            stream=True,
        )
        response_text = ""
        for chunk in completion:
            delta = chunk.choices[0].text or ""
            response_text += delta
            yield postprocess(response_text)

    return fn


def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):
            prompt = ""
            for user_msg, assistant_msg in history:
                prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
            prompt += f"User: {message}\nAssistant: "
            return {"prompt": prompt}

        postprocess = lambda x: x  # No post-processing needed
    else:
        # Add other pipeline types when they will be needed
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    # Determine the pipeline type based on the model name
    # For simplicity, assuming all models are chat models at the moment
    return "chat"


def registry(name: str, token: str | None = None, **kwargs):
    """
    Create a Gradio Interface for a model on Fireworks.

    Parameters:
        - name (str): The name of the OpenAI model.
        - token (str, optional): The API key for OpenAI.
    """
    api_key = token or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable is not set.")

    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(name, preprocess, postprocess, api_key)

    if pipeline == "chat":
        interface = gr.ChatInterface(fn=fn, **kwargs)
    else:
        # For other pipelines, create a standard Interface (not implemented yet)
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface
