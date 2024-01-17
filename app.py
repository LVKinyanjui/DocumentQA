# %%
import gradio as gr
from auxiliaries import embed_upsert, retrieve

# %%
with gr.Blocks() as iface:
    namespace = gr.Textbox(
        label="Namespace", 
        placeholder="Display or Enter Pinecone Document Namespace",
        visible=False)

    gr.ChatInterface(
        fn=retrieve, 
        additional_inputs=[namespace]
    )

    file = gr.File()
    feedback = gr.Markdown()

    file.change(fn=embed_upsert, inputs=file, outputs=[feedback, namespace])

# %%
iface.launch()

# %%



