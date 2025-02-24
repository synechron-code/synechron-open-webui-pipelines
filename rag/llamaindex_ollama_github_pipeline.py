"""
title: Llama Index Ollama Github Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitHub repository.
requirements: 
"""

import logging
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
import asyncio

from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from azure.identity import DefaultAzureCredential


name = "GitHub RAG"

def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()


class Pipeline:
    class Valves(BaseModel):
        DISABLED: bool = Field(
            default = True, description="Disable pipeline"
        )
        GITHUB_TOKEN: str = Field(
            default = "", description="GitHub PAT with read access to repo to be accessed"
        )
        GITHUB_OWNER: str = Field(
            default = "", description="GitHub owner for the repo e.g. http://github.com/<owner>/..."
        )
        GITHUB_REPO: str = Field(
            default = "", description="GitHub repo e.g. http://github.com/<owner>/<repo>"
        )
        GITHUB_BRANCH: str = Field(
            default = "main", description="GitHub branch to be analyzed e.g. http://github.com/<owner>/<repo>/tree/<branch>"
        )
        OLLAMA_HOST: str = Field(
            default = "http://ollama:11434", description="Ollama server host URL"
        )
        EMBED_MODEL: str = Field(
            default = "text-embedding-3-large", description="Embedding model name (i.e. OpenAI name)"
        )
        MODEL: str = Field(
            default = "gpt-4o-mini", description="Model name (i.e. OpenAI name)"
        )
        AZURE_OPENAI_API_KEY: Optional[str] = Field(
            default = None, description="Azure OpenAI key, if key is None, DefaultAzureCredential will retrieve key for Managed Identity"
        )
        AZURE_OPENAI_ENDPOINT: Optional[str] = Field(
            default = None, description="Azure OpenAI endpoint, blank Azure endpoint will enable Ollama endpoint and models"
        )
        AZURE_OPENAI_API_VERSION: Optional[str] = Field(
            default = "2025-01-01-preview", description="Azure OpenAI versions"
        )
        AZURE_OPENAI_MODEL_NAME: Optional[str] = Field(
            default = "gpt-4o-mini-payg", description="Deployment name (i.e. name given to model during deployment)"
        )
        AZURE_OPENAI_EMBED_MODEL_NAME: Optional[str] = Field(
            default = "text-embedding-3-large", description="Embedding model deployment name (i.e. name given to model during deployment)"
        )
        INCLUDE_FILE_EXTENSIONS: Optional[str] = Field(
            default= None, description="List of file extensions to include in the filter separated by ';'"
        )
        EXCLUDE_FILE_EXTENSIONS: Optional[str] = Field(
            default=".png;.jpg;.jpeg;.gif;.svg;.ico;.json;.ipynb",
            description="List of file extensions to exclude in the filter separated by ';'",
        )
        INCLUDE_DIRECTORIES: Optional[str] = Field(
            default= None, description="List of directories to include in the filter separated by ';'"
        )
        EXCLUDE_DIRECTORIES: Optional[str] = Field(
            default= None, description="List of directories to exclude in the filter separated by ';'"
        )


    def __init__(self):
        self.type = "manifold"
        self.documents = None
        self.index = None
        self.embed_model = None
        self.llm = None

        global index, documents

        try:
            self.valves = self.Valves()
        except Exception as e:
            logger.exception(f"Error initializing Valves: {e}")

        self.pipelines = self.pipes()
        pass

    def pipes(self) -> list[dict[str, str]]:
        owner = self.valves.GITHUB_OWNER
        repo = self.valves.GITHUB_REPO
        branch = self.valves.GITHUB_BRANCH
        out = [
            {"id": f"{name}:{owner}:{repo}:{branch}", "name": f"{name}:{owner}:{repo}:{branch}"}
        ]
        logger.info(f"llamaindex_ollama_github_pipeline - {name}: {owner}/{repo}/{branch}")
        return out

    def _init_models(self):
        """
        Create models. Requires Azure Credentials. See the DefaultAzureCredential documentation for details
        of the authentication process (it will cascade through multiple authentication methods until it finds one that
        works, including a Workload Identity, an SPN via Env Vars, or az login credentials when running locally).
        """

        if self.valves.AZURE_OPENAI_MODEL_NAME is not None:
            default_credential = DefaultAzureCredential(exclude_environment_credential=True)
            token = default_credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )

            try:
                self.llm = AzureOpenAI(
                    model=self.valves.MODEL,
                    deployment_name=self.valves.AZURE_OPENAI_MODEL_NAME,
                    api_key=self.valves.AZURE_OPENAI_API_KEY or token.token,
                    azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                    api_version=self.valves.AZURE_OPENAI_API_VERSION,
                )
                logger.info(f"Created AzureOpenAI llm: {self.valves.AZURE_OPENAI_MODEL_NAME}")
            except Exception as e:
                return f"Error: {e}"
        else:
            from llama_index.llms.ollama import Ollama
            try:
                self.llm = Ollama(
                    model=self.valves.MODEL,
                    base_url=self.valves.OLLAMA_HOST,
                    request_timeout=300.0
                )
                logger.info(f"Created ollama llm: {self.valves.MODEL}")
            except Exception as e:
                return f"Error: {e}"

        if self.valves.AZURE_OPENAI_EMBED_MODEL_NAME is not None:
            default_credential = DefaultAzureCredential(exclude_environment_credential=True)
            token = default_credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )

            try:
                self.embed_model = AzureOpenAIEmbedding(
                    model=self.valves.EMBED_MODEL,
                    deployment_name=self.valves.AZURE_OPENAI_EMBED_MODEL_NAME,
                    api_key=self.valves.AZURE_OPENAI_API_KEY or token.token,
                    azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                    api_version=self.valves.AZURE_OPENAI_API_VERSION,
                )
                logger.info(f"Created AzureOpenAI embedding: {self.valves.AZURE_OPENAI_EMBED_MODEL_NAME}")
            except Exception as e:
                return f"Error: {e}"
        else:
            from llama_index.embeddings.ollama import OllamaEmbedding
            try:
                self.embed_model = OllamaEmbedding(
                    model_name=self.valves.EMBED_MODEL,
                    base_url=self.valves.OLLAMA_HOST,
                    request_timeout=120.0
                )
                logger.info(f"Created ollama embedding: {self.valves.EMBED_MODEL}")
            except Exception as e:
                return f"Error: {e}"

    async def _init_embeddings(self):
        github_token = self.valves.GITHUB_TOKEN
        owner = self.valves.GITHUB_OWNER
        repo = self.valves.GITHUB_REPO
        branch = self.valves.GITHUB_BRANCH

        if self.valves.DISABLED:
            logger.warning(f"Pipeline disabled")
            return

        # if any are None then exit
        if not github_token or not owner or not repo:
            logger.warning(f"Github parameters must be configured")
            return

        logger.info(f"Start github embedding for {owner}/{repo}/{branch}")
        self._init_models()
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        github_client = GithubClient(github_token=github_token, verbose=True)

        include_file_extensions = self.valves.INCLUDE_FILE_EXTENSIONS.split(";") if self.valves.INCLUDE_FILE_EXTENSIONS else []
        exclude_file_extensions = self.valves.EXCLUDE_FILE_EXTENSIONS.split(";") if self.valves.EXCLUDE_FILE_EXTENSIONS else []
        include_directories = self.valves.INCLUDE_DIRECTORIES.split(";") if self.valves.INCLUDE_DIRECTORIES else []
        exclude_directories = self.valves.EXCLUDE_DIRECTORIES.split(";") if self.valves.EXCLUDE_DIRECTORIES else []

        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            filter_file_extensions=(
                include_file_extensions,
                GithubRepositoryReader.FilterType.INCLUDE,
            ) if include_file_extensions else (
                exclude_file_extensions,
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
            filter_directories=(
                include_directories,
                GithubRepositoryReader.FilterType.INCLUDE,
            ) if include_file_extensions else (
                exclude_directories,
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
        )

        loop = asyncio.new_event_loop()

        reader._loop = loop

        try:
            # Load data from the branch
            self.documents = await asyncio.to_thread(reader.load_data, branch=branch)
            self.index = VectorStoreIndex.from_documents(self.documents, show_progress=True)
        except Exception as e:
            logger.exception(f"Error: {e}")
        finally:
            loop.close()

        if self.index is None:
            logger.error("Vector store index is not initialized")
            return

        logger.info(f"Finished github embedding for {len(self.documents)} documents for {owner}/{repo}/{branch}")

    async def on_valves_updated(self):
        logger.debug(f"on_valves_update: {name}")
        self.pipelines = self.pipes()
        await self._init_embeddings()

    async def on_startup(self):
        logger.info(f"on_startup: {name}")
        self.valves.DISABLED = True
        pass

    async def on_shutdown(self):
        logger.info(f"on_shutdown: {name}")
        # This function is called when the server is stopped.
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        logger.debug(f"message: {messages}")
        logger.debug(f"user_message: {user_message}")

        if self.index is None:
            logger.error("Vector store index is not initialized")
            return None
        try:
            query_engine = self.index.as_query_engine(
                streaming=True,
                similarity_top_k=0,
                vector_store_query_mode="default"
                )
            response = query_engine.query(user_message)
        except Exception as e:
            raise Exception(f"Exception in llamaindex_ollama_github_pipeline: {e}")

        return response.response_gen
