import gradio as gr
import fireworks_gradio
 
gr.load(
    name='f1-preview',
    src=fireworks_gradio.registry,
).launch()