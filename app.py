# %%
import gradio as gr
from auxiliaries import embed_upsert, retrieve

# %%
with gr.Blocks() as iface:
    namespace = gr.Textbox()

    gr.ChatInterface(
        fn=retrieve, 
        additional_inputs=[namespace]
    ).launch()

    file = gr.File()
    feedback = gr.Markdown()

    file.change(fn=embed_upsert, inputs=file, outputs=[feedback, namespace])

# %%
iface.launch(share=True)


