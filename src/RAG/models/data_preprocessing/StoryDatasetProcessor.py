from datasets import load_dataset
import pandas as pd
from langchain_community.document_loaders.dataframe import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from typing import Tuple, Optional

class StoryDatasetProcessor:
    def __init__(
        self,
        dataset_name: str = "FareedKhan/1k_stories_100_genre",
        columns: list[str] = ["id", "title", "story", "genre"],
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
    ):
        self.dataset_name = dataset_name
        self.columns = columns
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.df: pd.DataFrame = pd.DataFrame()
        self.documents = []
        self.chunked_documents = []

    def load_dataset(self, split: str = "train"):
        """Load dataset from Hugging Face and convert to pandas DataFrame."""
        ds = load_dataset(self.dataset_name)
        self.df = ds[split].to_pandas()[self.columns]
        return self.df

    def load_documents(self, page_content_column: str = "story"):
        """Load documents using LangChain DataFrameLoader."""
        loader = DataFrameLoader(self.df, page_content_column=page_content_column)
        self.documents = loader.load()
        return self.documents

    def split_documents(self):
        """Split documents into chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.chunked_documents = splitter.split_documents(self.documents)
        return self.chunked_documents

    def normalize_metadata(self):
        """Normalize title and genre to lowercase."""
        for doc in self.chunked_documents:
            if "title" in doc.metadata and doc.metadata["title"]:
                doc.metadata["title"] = doc.metadata["title"].strip().lower()
            if "genre" in doc.metadata and doc.metadata["genre"]:
                doc.metadata["genre"] = doc.metadata["genre"].strip().lower()
        return self.chunked_documents
    
    def build_valid_metadata(self) -> tuple[set, set]:
        """
        Compute unique normalized titles and genres from chunked documents.
        Deduplicates automatically and saves them to:
            assets/validation_metadata/valid_metadata.json
        Returns:
            valid_titles (set): Unique normalized titles
            valid_genres (set): Unique normalized genres
        """
        if not self.chunked_documents:
            self.load_dataset()
            self.load_documents()
            self.split_documents()
            self.normalize_metadata()


        # Deduplicate titles and genres
        valid_titles = {doc.metadata['title'] for doc in self.chunked_documents if doc.metadata.get('title')}
        valid_genres = {doc.metadata['genre'] for doc in self.chunked_documents if doc.metadata.get('genre')}
        valid_ids = {doc.metadata['id'] for doc in self.chunked_documents if doc.metadata.get('id')}

        # Compute path automatically
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(project_root, "assets", "validation_metadata")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "valid_metadata.json")

        # Save to JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "titles": list(valid_titles),
                "ids": list(valid_ids),
                "genres": list(valid_genres)
            }, f, ensure_ascii=False, indent=2)

        print(f"Saved valid metadata to: {save_path}")

        return valid_titles, valid_genres

    def process_all(self):
        """Run full pipeline: load, documentify, split, normalize."""
        self.load_dataset()
        self.load_documents()
        self.split_documents()
        self.normalize_metadata()
        self.build_valid_metadata()
        return self.chunked_documents
