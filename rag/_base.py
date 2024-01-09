from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict, List
import ray
from pathlib import Path
from functools import wraps
from langchain.text_splitter import TextSplitter
from functools import partial



class Item(BaseModel):
    text: str
    source: str

class DocPath(BaseModel):
    path: Path


def formatit(func):
    """
    Decorator to time a function.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        args_list = list(args)
        args_list[1] = args_list[1]["item"]
        ret = func(*args_list, **kwargs)

        return [
            {"source": item.source, "text": item.text}
            for item in ret
        ]

    return inner

class DataIngest(ABC):
    """Ingest data from raw format to dict of class Item.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
    """

    def __init__(
        self,
        docs_dir: str,
    ):
        self.docs_dir = Path(docs_dir)

    def ingest(self) -> ray.data.Dataset:
        """handle documents to dict of Items.

        Args:
            docs_dir (str): path of documents.
        """
        file_paths = self.load_docs()
        sections_ds = file_paths.flat_map(self.extract_doc)

        return sections_ds


    def load_docs(self) -> ray.data.Dataset:
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        ds = ray.data.from_items(
            [DocPath(path=path) for path in self.docs_dir.rglob("*.html") if not path.is_dir()]
        )

        return ds


    @abstractmethod
    @formatit
    def extract_doc(self, doc_path: DocPath) -> List[Item]:
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        pass


class SplitChunks(ABC):
    def __init__(self, chunk_size, chunk_overlap):
        # Embedding model
        self.splitter = self.get_chunk_spliter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, section):
        chunks = self.splitter.create_documents(
            texts=[section["text"]], metadatas=[{"source": section["source"]}]
        )
        return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]
    
    def chunk(self, docs: ray.data.Dataset) -> ray.data.Dataset:
        chunks_ds = docs.flat_map(self.split)
        return chunks_ds


    @abstractmethod
    def get_chunk_spliter(self, chunk_size: int, chunk_overlap: int) -> TextSplitter:
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        pass
