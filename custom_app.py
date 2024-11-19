import gradio as gr
import fireworks_gradio

gr.load(
    name='f1-preview',
    src=fireworks_gradio.registry,
    title='Fireworks-Gradio Integration',
    description="Chat with f1-preview model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()