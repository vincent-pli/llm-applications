from langchain.text_splitter import RecursiveCharacterTextSplitter
from ._base import SplitChunks

class SplitHtmlSection(SplitChunks):
    def __init__(self, chunk_size, chunk_overlap):
        # Embedding model
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def get_chunk_spliter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )

            return text_splitter



# def chunk_section(section, chunk_size, chunk_overlap):
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", " ", ""],
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#     )
#     chunks = text_splitter.create_documents(
#         texts=[section["text"]], metadatas=[{"source": section["source"]}]
#     )
#     return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]
    

