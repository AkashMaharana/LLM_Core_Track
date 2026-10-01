from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"
deployment_name_gpt = "gpt-5.1"


def main():

    # ---------------------------------------------------------
    # 1. Models
    # ---------------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model=deployment_name_embedding,
        base_url=endpoint
    )

    model = ChatOpenAI(
        model=deployment_name_gpt,
        base_url=endpoint
    )

    # ---------------------------------------------------------
    # 2. Existing Chroma database
    # ---------------------------------------------------------

    vector_store = Chroma(
        collection_name="employee_policies_langchain",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    # ---------------------------------------------------------
    # 3. Conversation
    # ---------------------------------------------------------

    previous_user_question = (
        "What is the annual leave policy?"
    )

    previous_answer = (
        "Employees receive 25 days of "
        "annual leave each year."
    )

    current_question = (
        "What about after five years?"
    )

    # =========================================================
    # STAGE 1 — QUERY REWRITING
    # =========================================================

    rewrite_prompt = ChatPromptTemplate.from_template(
        """
Rewrite the latest question as a standalone
search query using the conversation.

Do not answer the question.

Previous user question:
{previous_question}

Previous assistant answer:
{previous_answer}

Latest question:
{current_question}

Standalone search query:
"""
    )

    rewrite_chain = (
        rewrite_prompt
        | model
        | StrOutputParser()
    )

    rewritten_query = rewrite_chain.invoke(
        {
            "previous_question":
                previous_user_question,

            "previous_answer":
                previous_answer,

            "current_question":
                current_question,
        }
    )

    print("\n1. ORIGINAL QUESTION")
    print("=" * 70)
    print(current_question)

    print("\n2. REWRITTEN QUERY")
    print("=" * 70)
    print(rewritten_query)

    # =========================================================
    # STAGE 2 — RETRIEVAL
    # =========================================================

    results = vector_store.similarity_search_with_score(
        rewritten_query,
        k=3,
    )

    print("\n3. RETRIEVED CHUNKS")
    print("=" * 70)

    retrieved_documents = []

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):

        retrieved_documents.append(document)

        print(f"\nChunk {index}")
        print("-" * 70)

        print("Score:")
        print(score)

        print("\nContent:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)

    # =========================================================
    # STAGE 3 — BUILD CONTEXT
    # =========================================================

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    print("\n4. FINAL CONTEXT")
    print("=" * 70)
    print(context)

    # =========================================================
    # STAGE 4 — GENERATION
    # =========================================================

    answer_prompt = ChatPromptTemplate.from_template(
        """
Answer the question using only the context.

If the answer is not present in the context,
say:

"I don't know based on the provided information."

Do not add unsupported information.

CONTEXT:

{context}

QUESTION:

{question}
"""
    )

    answer_chain = (
        answer_prompt
        | model
        | StrOutputParser()
    )

    answer = answer_chain.invoke(
        {
            "context": context,
            "question": current_question,
        }
    )

    print("\n5. FINAL ANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()