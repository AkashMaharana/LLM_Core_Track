from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"
deployment_name_gpt = "gpt-5.1"

def main():

    # ---------------------------------------------------------
    # 1. Embedding model
    # ---------------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model=deployment_name_embedding,
        base_url=endpoint
    )

    # ---------------------------------------------------------
    # 2. Connect to existing Chroma database
    # ---------------------------------------------------------

    vector_store = Chroma(
        collection_name="employee_policies_langchain",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    # ---------------------------------------------------------
    # 3. Vector Store → Retriever
    # ---------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    # ---------------------------------------------------------
    # 4. Format retrieved Document objects
    # ---------------------------------------------------------

    def format_documents(documents):

        return "\n\n".join(
            document.page_content
            for document in documents
        )

    # ---------------------------------------------------------
    # 5. Prompt template
    # ---------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are answering questions using an internal knowledge base.

Use only the context provided below.

If the answer cannot be determined from the context,
say:

"I don't know based on the provided information."

Do not invent information.

CONTEXT:

{context}

QUESTION:

{question}
"""
    )

    # ---------------------------------------------------------
    # 6. LLM
    # ---------------------------------------------------------

    model = ChatOpenAI(
        model=deployment_name_gpt,
        base_url=endpoint
    )

    # ---------------------------------------------------------
    # 7. Build RAG chain using LCEL
    # ---------------------------------------------------------

    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    # ---------------------------------------------------------
    # 8. Ask question
    # ---------------------------------------------------------

    question = (
        "How much vacation do employees receive "
        "and how much can they carry forward?"
    )

    answer = rag_chain.invoke(
        question
    )

    print("\nQUESTION")
    print("=" * 70)
    print(question)

    print("\nANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()