"""
title: Llama Index Ollama Github Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitHub repository.
requirements: llama-index-core, llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-readers-github
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import asyncio


class Pipeline:
    class Valves(BaseModel):
        GITHUB_TOKEN: str
        GITHUB_OWNER: str
        GITHUB_REPO: str
        GITHUB_BRANCH: str
        OLLAMA_HOST: str
        EMBED_MODEL: str
        MODEL: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", "your-github-personal-access-token"),
                "GITHUB_OWNER": os.getenv("GITHUB_OWNER", "your-github-owner-name"),
                "GITHUB_REPO": os.getenv("GITHUB_REPO", "your-github-repo-name"),
                "GITHUB_BRANCH": os.getenv("GITHUB_BRANCH", "your-github-repo-branch-name"),
                "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                "EMBED_MODEL": os.getenv("OLLAMA_HOST", "nomic-embed-text"),
                "MODEL": os.getenv("OLLAMA_HOST", "llama3"),
            }
        )

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.readers.github import GithubRepositoryReader, GithubClient

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.EMBED_MODEL,
            base_url=self.valves.OLLAMA_HOST,
        )
        Settings.llm = Ollama(model=self.valves.MODEL)

        global index, documents

        github_token = self.valves.GITHUB_TOKEN
        owner = self.valves.GITHUB_OWNER
        repo = self.valves.GITHUB_REPO
        branch = self.valves.GITHUB_BRANCH

        github_client = GithubClient(github_token=github_token, verbose=True)

        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            filter_file_extensions=(
                [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".svg",
                    ".ico",
                    "json",
                    ".ipynb",
                ],
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
        )

        loop = asyncio.new_event_loop()

        reader._loop = loop

        try:
            # Load data from the branch
            self.documents = await asyncio.to_thread(reader.load_data, branch=branch)
            self.index = VectorStoreIndex.from_documents(self.documents)
        finally:
            loop.close()

        print(self.documents)
        print(self.index)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
