# %%
import gradio as gr
from auxiliaries_beta import embed_upsert, retrieve, summarize

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
        button = gr.Button("Summarize")

    file = gr.File()
    feedback = gr.Markdown()

    file.change(fn=embed_upsert, inputs=file, outputs=[feedback, namespace])
    button.click(fn=summarize, inputs=file, outputs=summary)
    # file.change(fn=summarize, inputs=file, outputs=summary)

# %%
iface.launch()

# %%



