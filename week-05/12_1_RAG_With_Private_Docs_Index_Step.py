from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"
deployment_name_gpt = "gpt-5.1"


def main():

    document_folder = Path("/Users/in46032455/Documents/My Learnings/LLM_0_To_1/Core_Track/week-05/private_docs")

    documents = []

    # ---------------------------------------------------------
    # 1. Load files
    # ---------------------------------------------------------

    for file_path in document_folder.rglob("*"):

        if file_path.suffix.lower() in [".txt", ".md"]:

            loader = TextLoader(
                str(file_path),
                encoding="utf-8",
            )

            documents.extend(
                loader.load()
            )

        elif file_path.suffix.lower() == ".pdf":

            loader = PyPDFLoader(
                str(file_path)
            )

            documents.extend(
                loader.load()
            )

    print(
        "Documents/pages loaded:",
        len(documents)
    )

    # ---------------------------------------------------------
    # 2. Add useful metadata
    # ---------------------------------------------------------

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown"
        )

        document.metadata["filename"] = (
            Path(source).name
        )

    # ---------------------------------------------------------
    # 3. Split into chunks
    # ---------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        "Chunks created:",
        len(chunks)
    )

    # ---------------------------------------------------------
    # 4. Embedding model
    # ---------------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model=deployment_name_embedding,
        base_url=endpoint,
    )

    # ---------------------------------------------------------
    # 5. Create persistent vector store
    # ---------------------------------------------------------

    vector_store = Chroma(
        collection_name="personal_knowledge",
        embedding_function=embeddings,
        persist_directory="./personal_knowledge_db",
    )

    # ---------------------------------------------------------
    # 6. Store chunks
    # ---------------------------------------------------------

    ids = [
        f"chunk-{index}"
        for index in range(len(chunks))
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(
        "Knowledge base indexed successfully."
    )


if __name__ == "__main__":
    main()