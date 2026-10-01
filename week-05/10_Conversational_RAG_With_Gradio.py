import gradio as gr

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"
deployment_name_gpt = "gpt-5.1"


def main():

    # ---------------------------------------------------------
    # 1. Connect to existing vector store
    # ---------------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model=deployment_name_embedding,
        base_url=endpoint
    )

    vector_store = Chroma(
        collection_name="employee_policies_langchain",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    # ---------------------------------------------------------
    # 2. LLM
    # ---------------------------------------------------------

    model = ChatOpenAI(
        model=deployment_name_gpt,
        base_url=endpoint
    )

    # =========================================================
    # 3. Query rewriting prompt
    # =========================================================

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Given the chat history and latest user question,
rewrite the question as a standalone search query.

Do not answer the question.

If the latest question is already standalone,
return it unchanged.
"""
            ),

            MessagesPlaceholder(
                "chat_history"
            ),

            (
                "human",
                "{input}"
            ),
        ]
    )

    history_aware_retriever = (
        create_history_aware_retriever(
            model,
            retriever,
            rewrite_prompt,
        )
    )

    # =========================================================
    # 4. Answer prompt
    # =========================================================

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an assistant answering questions from
an internal company knowledge base.

Use only the retrieved context below.

If the answer cannot be determined from the
context, say:

"I don't know based on the provided information."

Do not invent facts.

CONTEXT:

{context}
"""
            ),

            MessagesPlaceholder(
                "chat_history"
            ),

            (
                "human",
                "{input}"
            ),
        ]
    )

    answer_chain = create_stuff_documents_chain(
        model,
        answer_prompt,
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        answer_chain,
    )

    # ---------------------------------------------------------
    # 5. Convert Gradio content → plain text
    # ---------------------------------------------------------

    def extract_text(content):

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            parts = []

            for block in content:

                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                ):
                    parts.append(
                        block.get("text", "")
                    )

            return "\n".join(parts)

        return str(content)

    # ---------------------------------------------------------
    # 6. Gradio callback
    # ---------------------------------------------------------

    def respond(message, history):

        chat_history = []

        # Convert Gradio history into LangChain messages
        for item in history:

            role = item.get("role")

            content = extract_text(
                item.get("content", "")
            )

            if role == "user":

                chat_history.append(
                    HumanMessage(
                        content=content
                    )
                )

            elif role == "assistant":

                chat_history.append(
                    AIMessage(
                        content=content
                    )
                )

        # Run conversational RAG
        response = rag_chain.invoke(
            {
                "input": message,
                "chat_history": chat_history,
            }
        )

        answer = response["answer"]

        return answer

    # ---------------------------------------------------------
    # 7. UI
    # ---------------------------------------------------------

    demo = gr.ChatInterface(
        fn=respond,

        title="Employee Policy Assistant",

        description=(
            "Ask questions about the employee "
            "policy knowledge base."
        ),

        examples=[
            "What is the annual leave policy?",
            "How many days can I work remotely?",
            "When do expenses require manager approval?",
        ],
    )

    demo.launch()


if __name__ == "__main__":
    main()