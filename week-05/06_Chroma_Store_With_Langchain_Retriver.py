from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"


def main():

    # ---------------------------------------------------------
    # 1. Load source document
    # ---------------------------------------------------------

    loader = TextLoader(
        "/Users/in46032455/Documents/My Learnings/LLM_0_To_1/Core_Track/week-05/knowledge_base/employee_policy.txt"
    )

    documents = loader.load()

    documents[0].metadata["doctype"] = "HR_POLICY"

    # ---------------------------------------------------------
    # 2. Split into chunks
    # ---------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        "Total chunks:",
        len(chunks)
    )

    # ---------------------------------------------------------
    # 3. Configure embedding model
    # ---------------------------------------------------------

    embeddings = OpenAIEmbeddings(
        base_url=endpoint,
        model=deployment_name_embedding
    )

    # ---------------------------------------------------------
    # 4. Create LangChain Chroma vector store
    # ---------------------------------------------------------

    vector_store = Chroma(
        collection_name="employee_policies_langchain",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    # ---------------------------------------------------------
    # 5. Add documents
    # ---------------------------------------------------------

    ids = [
        f"chunk-{index}"
        for index in range(len(chunks))
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    # ---------------------------------------------------------
    # 6. Convert vector store → Retriever
    # ---------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    # ---------------------------------------------------------
    # 7. Query
    # ---------------------------------------------------------

    question = (
        "How much vacation do employees receive "
        "and how much can they carry forward?"
    )

    retrieved_documents = retriever.invoke(
        question
    )

    # ---------------------------------------------------------
    # 8. Inspect results
    # ---------------------------------------------------------

    print("\nQUESTION")
    print(question)

    print("\nRETRIEVED DOCUMENTS")

    for index, document in enumerate(
        retrieved_documents
    ):

        print("\n" + "=" * 60)

        print(
            f"Result {index + 1}"
        )

        print("\nContent:")
        print(
            document.page_content
        )

        print("\nMetadata:")
        print(
            document.metadata
        )


if __name__ == "__main__":
    main()