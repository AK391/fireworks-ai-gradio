import gradio as gr
import fireworks_gradio

with gr.Blocks() as demo:
    with gr.Tab("llama-v3p2-11b-vision-instruct"):
        gr.load('llama-v3p2-11b-vision-instruct', src=fireworks_gradio.registry)
    with gr.Tab("llama-v3p1-405b-instruct"):
        gr.load('llama-v3p1-405b-instruct', src=fireworks_gradio.registry)

demo.launch()