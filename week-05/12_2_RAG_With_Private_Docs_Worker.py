from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI,
)

from langchain_chroma import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate,
)

from langchain_core.output_parsers import (
    StrOutputParser,
)
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
    # 2. Existing knowledge base
    # ---------------------------------------------------------

    vector_store = Chroma(
        collection_name="personal_knowledge",
        embedding_function=embeddings,
        persist_directory="./personal_knowledge_db",
    )

    # ---------------------------------------------------------
    # 3. User question
    # ---------------------------------------------------------

    question = input(
        "\nAsk your knowledge base: "
    )

    # ---------------------------------------------------------
    # 4. Retrieve relevant chunks
    # ---------------------------------------------------------

    results = vector_store.similarity_search_with_score(
        question,
        k=5,
    )

    documents = [
        document
        for document, score in results
    ]

    # ---------------------------------------------------------
    # 5. Display retrieved evidence
    # ---------------------------------------------------------

    print("\nRETRIEVED CONTEXT")
    print("=" * 70)

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nResult {index}"
        )

        print(
            "Source:",
            document.metadata.get(
                "filename",
                "unknown",
            )
        )

        print(
            "Score:",
            round(score, 4)
        )

        print(
            document.page_content[:500]
        )

    # ---------------------------------------------------------
    # 6. Build context
    # ---------------------------------------------------------

    context = "\n\n".join(
        f"""
SOURCE: {document.metadata.get("filename", "unknown")}

{document.page_content}
"""
        for document in documents
    )

    # ---------------------------------------------------------
    # 7. Prompt
    # ---------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are a personal knowledge assistant.

Answer the question using only the supplied
knowledge-base context.

If the context does not contain enough
information, say that you do not know based
on the available documents.

Do not invent facts.

Where possible, mention which source document
supports the answer.

CONTEXT:

{context}

QUESTION:

{question}
"""
    )

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    # ---------------------------------------------------------
    # 8. Generate answer
    # ---------------------------------------------------------

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print("\nANSWER")
    print("=" * 70)

    print(answer)


if __name__ == "__main__":
    main()