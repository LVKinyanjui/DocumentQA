from langchain.chat_models import ChatGooglePalm
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

import os
genai_key = os.getenv("PALM_API_KEY")

llm = ChatGooglePalm(
    google_api_key=genai_key
)

def route_user_responses(user_input, contexts):
    retriever_template = f"""
    You are a teacher \
    You start off by creating A SINGLE qustion from a textbook context, 
    enclosed in triple backticks ``` \
    You are provided with a retrieved context on a topic \
    You are to create a single question for the user \
    You are provided with a user input enclosed in triple hyphens ---

    ---
    {user_input}
    ---

    and you are also provided with the context from which you ask a question related to the user input
    enclosed in triple backticks

    ```
    {contexts}
    ```

    """

    # %% [markdown]
    # ### Responder Chain
    # This takes the user response as a variable and uses its memory to rate this response. It will NOT make use of retrieval text in this step as that will have been extracted during question construction.

    # %%
    responder_template = f"""
    You are providedw with an answer by the user \
    this is a response to a question you may have asked earlier \
    You are to rate this answer according to the information you already know \
    If it is correct, tell them 'correct', plus any additional feedback \
    If wrong, tell user in a kindly what the right answer is \
    You are provided with the user answer enclosed in triple hyphens ---

    ---
    {user_input}
    ---
    """

    # %%
    prompt_infos = [
        {
            "name": "retriever",
            "description": "Taking queried text from a vectorstore and using it to create a question for the user",
            "prompt_template": retriever_template
        },
        {
            "name": "responder",
            "description": "Takes the user response as a variable and uses its memory to rate this response",
            "prompt_template": responder_template
        }
    ]

    # %%

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain  
        
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    # %%
    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    # %%
    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below OR it can be "DEFAULT" if the input is not\
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""

    # %%
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str
    )
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    # %%
    chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains, 
    default_chain=default_chain, verbose=True
    )

    res = chain.run(user_input)
    return res

