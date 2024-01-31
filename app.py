# %%
import gradio as gr
from auxiliaries_beta import (
    embed_upsert, summarize, ask_question, answer_question
)

# %%
def save_chats(history):
    with open("chat_history.txt", "w", encoding='utf-8') as f:
        for line in history:
            f.writelines(line)
            f.write('\n\n')

# %%
chatbot_ask = gr.Chatbot(
    value=[[None, "Hi. Ask me a question about the document you uploaded."]],
    )
chatbot_answer = gr.Chatbot(
    value=[[None, "Hi. Let me ask you a question about the document you uploaded."]],
)

namespace = gr.Textbox(
    label="Namespace (Pinecone)", 
    placeholder="(Optional) Display or Enter Pinecone Document Namespace",
    visible=False
)

with gr.Blocks() as demo:
    with gr.Tab("Upload Document"):
        file = gr.File()
        feedback = gr.Markdown()

        file.change(fn=embed_upsert, inputs=file, outputs=[feedback, namespace])

        with gr.Accordion("Document Summary"):
            summary = gr.Markdown()
            button = gr.Button("Summarize")


        button.click(fn=summarize, inputs=file, outputs=summary)

    with gr.Tab("Ask Questions"):

        gr.ChatInterface(
            fn=ask_question, 
            additional_inputs=[namespace],
            chatbot=chatbot_ask
        )


    with gr.Tab("Answer Questions"):
        gr.ChatInterface(
            fn=answer_question, 
            additional_inputs=[namespace],
            chatbot=chatbot_answer
        )


# %%
demo.launch()

# %%



