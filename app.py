# %%
import gradio as gr
from auxiliaries import embed_upsert, retrieve

# %%
with gr.Blocks() as iface:
    namespace = gr.Textbox(
        label="Document Namespace", 
        placeholder= "Enter the document's pinecone namespace")

    gr.ChatInterface(
        fn=retrieve, 
        additional_inputs=[namespace]
    )

    file = gr.File()
    feedback = gr.Markdown()

    file.change(fn=embed_upsert, inputs=file, outputs=[feedback, namespace])

# %%
iface.launch(share=True)


