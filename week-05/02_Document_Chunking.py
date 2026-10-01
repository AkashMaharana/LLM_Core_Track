from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():

    # ---------------------------------------------------------
    # 1. Load document
    # ---------------------------------------------------------

    loader = TextLoader(
        "/Users/in46032455/Documents/My Learnings/LLM_0_To_1/Core_Track/week-05/knowledge_base/employee_policy.txt"
    )

    documents = loader.load()

    print("\nLOADED DOCUMENT")
    print("=" * 60)

    print(documents[0].page_content)

    print("\nMETADATA")
    print(documents[0].metadata)

    # ---------------------------------------------------------
    # 2. Add custom metadata
    # ---------------------------------------------------------

    documents[0].metadata["doctype"] = "HR_POLICY"

    # ---------------------------------------------------------
    # 3. Create splitter
    # ---------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )

    # ---------------------------------------------------------
    # 4. Split document
    # ---------------------------------------------------------

    chunks = splitter.split_documents(
        documents
    )

    # ---------------------------------------------------------
    # 5. Inspect chunks
    # ---------------------------------------------------------

    print("\nTOTAL CHUNKS:", len(chunks))

    for index, chunk in enumerate(chunks):

        print("\n" + "=" * 60)
        print(f"CHUNK {index + 1}")

        print("\nCONTENT:")
        print(chunk.page_content)

        print("\nMETADATA:")
        print(chunk.metadata)


if __name__ == "__main__":
    main()