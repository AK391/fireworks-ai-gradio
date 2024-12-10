import os
from openai import OpenAI
import gradio as gr
from typing import Callable
from fireworks.client.audio import AudioInference

__version__ = "0.0.3"


def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str):
    if "whisper" in model_name:
        def fn(message, history, audio_input=None):
            # Handle audio input if provided
            if audio_input:
                if not audio_input.endswith('.wav'):
                    new_path = audio_input + '.wav'
                    os.rename(audio_input, new_path)
                    audio_input = new_path
                
                client = AudioInference(
                    model=model_name,
                    base_url="https://audio-prod.us-virginia-1.direct.fireworks.ai",
                    api_key=api_key
                )
                
                with open(audio_input, "rb") as f:
                    audio_data = f.read()
                response = client.transcribe(audio=audio_data)
                return {"role": "assistant", "content": response.text}
            
            # Handle text message
            if isinstance(message, dict):  # Multimodal input
                audio_path = message.get("files", [None])[0] or message.get("audio")
                text = message.get("text", "")
                if audio_path:
                    # Process audio file
                    return fn(None, history, audio_path)
                return {"role": "assistant", "content": "No audio input provided."}
            else:  # String input
                return {"role": "assistant", "content": "Please upload an audio file or use the microphone to record audio."}
            
    else:
        def fn(message, history, audio_input=None):
            # Ignore audio_input for non-whisper models
            inputs = preprocess(message, history)
            client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=api_key
            )
            
            model_path = (
                "accounts/fireworks/agents/f1-preview" if model_name == "f1-preview"
                else "accounts/fireworks/agents/f1-mini-preview" if model_name == "f1-mini"
                else f"accounts/fireworks/models/{model_name}"
            )
            
            completion = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": inputs["prompt"]}],
                stream=True,
                max_tokens=1024,
                temperature=0.7,
                top_p=1,
            )
            
            response_text = ""
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                response_text += delta
                yield {"role": "assistant", "content": response_text}

    return fn


def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):
            prompt = ""
            # Format history in OpenAI style
            for h in history:
                user_msg = h["role"] == "user" and h["content"] or ""
                assistant_msg = h["role"] == "assistant" and h["content"] or ""
                prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
            
            # Add current message
            prompt += f"User: {message}\nAssistant: "
            return {"prompt": prompt}

        def postprocess(response):
            # Return in OpenAI message format
            return {"role": "assistant", "content": response}

    elif pipeline == "audio":
        inputs = [
            gr.Audio(sources=["microphone"], type="filepath"),
            gr.Radio(["transcribe", "translate", "align"], label="Task", value="transcribe"),
            gr.Textbox(label="Text for Alignment", visible=False)
        ]
        outputs = "text"

        def preprocess(audio_path, task, text, history):
            if task == "align" and not text:
                raise ValueError("Text is required for alignment task")
            if audio_path and not audio_path.endswith('.wav'):
                new_path = audio_path + '.wav'
                os.rename(audio_path, new_path)
                audio_path = new_path
            return {"role": "user", "content": {"audio_path": audio_path, "task": task, "text": text}}

        def postprocess(text):
            return {"role": "assistant", "content": text}

    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    if "whisper" in model_name:
        return "audio"
    return "chat"


def registry(name: str, token: str | None = None, **kwargs):
    """
    Create a Gradio Interface for a model on Fireworks.
    Can be used directly or with gr.load:

    Example:
        # Direct usage
        interface = fireworks_gradio.registry("whisper-v3", token="your-api-key")
        interface.launch()

        # With gr.load
        gr.load(
            name='whisper-v3',
            src=fireworks_gradio.registry,
        ).launch()

    Parameters:
        name (str): The name of the OpenAI model.
        token (str, optional): The API key for OpenAI.
    """
    # Make the function compatible with gr.load by accepting name as a positional argument
    if not isinstance(name, str):
        raise ValueError("Model name must be a string")

    api_key = token or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable is not set.")

    pipeline = get_pipeline(name)
    _, _, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(name, preprocess, postprocess, api_key)

    description = kwargs.pop("description", None)
    if "whisper" in name:
        description = (description or "") + """
        \n\nSupported commands:
        - Record audio using the microphone
        - Use "/translate" to translate to English
        - Use "/align your text here" to align text
        """

        with gr.Blocks() as interface:
            chatbot = gr.Chatbot(type="messages", placeholder="<strong>Your Personal Audio Assistant</strong><br>Ask Me Anything")
            mic = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio")
            
            def process_audio(audio_path):
                if audio_path:
                    # Ensure WAV format
                    if not audio_path.endswith('.wav'):
                        new_path = audio_path + '.wav'
                        os.rename(audio_path, new_path)
                        audio_path = new_path
                    
                    # Create message format expected by fn
                    message = {"files": [audio_path], "text": ""}
                    response = fn(message, [])
                    
                    # Format messages in OpenAI style (Flat list)
                    return [
                        {"role": "user", "content": {"path": audio_path}},
                        {"role": "assistant", "content": response["content"]}
                    ]
                return []

            mic.change(
                fn=process_audio,
                inputs=[mic],
                outputs=[chatbot]
            )
    else:
        # For non-whisper models, use regular ChatInterface
        interface = gr.ChatInterface(
            fn=fn,
            type="messages",
            description=description,
            additional_inputs=gr.Audio(sources=["microphone"], type="filepath", label="Audio Input"),
            **kwargs
        )

    return interface


# Add these to make the module more discoverable
MODELS = [
    "whisper-v3",
    "whisper-v3-turbo",
    "f1-preview",
    "f1-mini",
    # Add other supported models here
]

def get_all_models():
    """Returns a list of all supported models."""
    return MODELS
