import gradio as gr
import fireworks_gradio

gr.load(
    name='llama-v3p1-405b-instruct',
    src=fireworks_gradio.registry,
).launch()