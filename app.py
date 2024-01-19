# %%
import gradio as gr
from auxiliaries import embed_upsert, retrieve, summarize

# %%
with gr.Blocks() as iface:
    namespace = gr.Textbox(
    label="Namespace (Pinecone)", 
    placeholder="(Optional) Display or Enter Pinecone Document Namespace",
    visible=False
    )

    gr.ChatInterface(
        fn=retrieve, 
        additional_inputs=[namespace]
    )

    with gr.Accordion("Document Summary"):
        summary = gr.Markdown()

    file = gr.File()
    feedback = gr.Markdown()

    file.change(fn=embed_upsert, inputs=file, outputs=[feedback, namespace])
    file.change(fn=summarize, inputs=file, outputs=summary)

# %%
iface.launch(share=True)

# %%



