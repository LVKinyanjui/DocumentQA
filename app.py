# %%
import gradio as gr
from auxiliaries import embed_upsert, retrieve, summarize

# %%
# Adding a welcome messahge
chatbot = gr.Chatbot(
    value=[[None, "Hi. I am here to help you search through your documents through question and answer"]]
    )

with gr.Blocks() as iface:
    namespace = gr.Textbox(
    label="Namespace (Pinecone)", 
    placeholder="(Optional) Display or Enter Pinecone Document Namespace",
    visible=False
    )


    gr.ChatInterface(
        fn=retrieve, 
        additional_inputs=[namespace],
        chatbot=chatbot
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



