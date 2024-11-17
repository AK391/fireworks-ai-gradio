import gradio as gr
import fireworks_gradio

gr.load(
    name='llama-v3p1-405b-instruct',
    src=fireworks_gradio.registry,
    title='Fireworks-Gradio Integration',
    description="Chat with llama-v3p1-405b-instruct model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()