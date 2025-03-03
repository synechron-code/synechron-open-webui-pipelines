"""
title: Llama Index Ollama Github Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitHub repository.
requirements: llama-index-embeddings-ollama, llama-index-embeddings-azure-openai, llama-index-readers-github
"""

import logging
from typing import List, Sequence, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
import asyncio

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from azure.identity import DefaultAzureCredential
K = 1024


name = "GitHubRAG"

def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


class Pipeline:
    class Valves(BaseModel):
        ENABLED: bool = Field(default=False, description="Enable pipeline")
        GITHUB_TOKEN: str = Field(default="", description="GitHub PAT with read access to repo to be accessed")
        GITHUB_OWNERS: str = Field(
            default="", description="List of GitHub owners separated by ';' i.e. http://github.com/<owner>/<repo>/tree/<branch>"
        )
        GITHUB_REPOS: str = Field(default="", description="List of GitHub repos separated by ';' i.e. http://github.com/<owner>/<repo>/tree/<branch>")
        GITHUB_BRANCHES: str = Field(
            default="main",
            description="List of GitHub branches separated by ';' e.g. http://github.com/<owner>/<repo>/tree/<branch>",
        )
        OLLAMA_HOST: str = Field(default="http://ollama:11434", description="Ollama server host URL")
        EMBED_MODEL: str = Field(
            default="text-embedding-3-large", description="Embedding model name (i.e. OpenAI name)"
        )
        MODEL: str = Field(default="gpt-4o-mini", description="Model name (i.e. OpenAI name)")
        AZURE_OPENAI_API_KEY: Optional[str] = Field(
            default=None,
            description="Azure OpenAI key, if key is None, DefaultAzureCredential will retrieve key for Managed Identity",
        )
        AZURE_OPENAI_ENDPOINT: Optional[str] = Field(
            default=None,
            description="Azure OpenAI endpoint, blank Azure endpoint will enable Ollama endpoint and models",
        )
        AZURE_OPENAI_API_VERSION: Optional[str] = Field(
            default="2025-01-01-preview", description="Azure OpenAI versions"
        )
        AZURE_OPENAI_MODEL_NAME: Optional[str] = Field(
            default="gpt-4o-mini-payg", description="Deployment name (i.e. name given to model during deployment)"
        )
        AZURE_OPENAI_EMBED_MODEL_NAME: Optional[str] = Field(
            default="text-embedding-3-large",
            description="Embedding model deployment name (i.e. name given to model during deployment)",
        )
        INCLUDE_FILE_EXTENSIONS: Optional[str] = Field(
            default=None, description="Comma delimited list of file extensions to include in the filter separated by ';'"
        )
        EXCLUDE_FILE_EXTENSIONS: Optional[str] = Field(
            default=".png;.jpg;.jpeg;.gif;.svg;.ico;.json;.ipynb",
            description="Comma delimited list of file extensions to exclude in the filter separated by ';'",
        )
        INCLUDE_DIRECTORIES: Optional[str] = Field(
            default=None, description="Comma delimited list of directories to include in the filter separated by ';'"
        )
        EXCLUDE_DIRECTORIES: Optional[str] = Field(
            default=None, description="Comma delimited list of directories to exclude in the filter separated by ';'"
        )
        DEBUG: bool = Field(default=False, description="Enable debug logging")


    def __init__(self):
        self.type = "manifold"
        self.indexes = None
        self.embed_model = None
        self.llm = None

        self.ext_excludes = {}
        self.ext_includes = {}
        self.dir_excludes = {}
        self.dir_includes = {}

        try:
            self.valves = self.Valves()
        except Exception as e:
            logger.exception(f"Error initializing Valves: {e}")

        if self.valves.DEBUG:
            logger.setLevel(logging.DEBUG)

        self.pipelines = self.pipes()
        pass

    def get_repos(self) -> List[dict]:
        owners = [owner.strip() for owner in self.valves.GITHUB_OWNERS.split(';')]
        repos = [repo.strip() for repo in self.valves.GITHUB_REPOS.split(';')]
        branches = [branch.strip() for branch in self.valves.GITHUB_BRANCHES.split(';')]
        ext_excludes = [ext.strip() for ext in self.valves.EXCLUDE_FILE_EXTENSIONS.split(';')] if self.valves.EXCLUDE_FILE_EXTENSIONS else []
        ext_includes = [ext.strip() for ext in self.valves.INCLUDE_FILE_EXTENSIONS.split(';')] if self.valves.INCLUDE_FILE_EXTENSIONS else []
        dir_excludes = [dir.strip() for dir in self.valves.EXCLUDE_DIRECTORIES.split(';')] if self.valves.EXCLUDE_DIRECTORIES else []
        dir_includes = [dir.strip() for dir in self.valves.INCLUDE_DIRECTORIES.split(';')] if self.valves.INCLUDE_DIRECTORIES else []

        if not (len(owners) == len(repos) == len(branches)):
            logger.error(
                f"The number of owners, repos, and branches must be the same. "
                f"owners: {len(owners)}, repos: {len(repos)}, branches: {len(branches)}"
            )
            return []

        if any(len(lst) != len(owners) for lst in [ext_excludes, ext_includes, dir_excludes, dir_includes] if lst):
            logger.error("The number of file extension and directory filters must match the number of repos.")
            return []

        out = []
        for index, (owner, repo, branch) in enumerate(zip(owners, repos, branches)):
            repo_id = f"{owner}:{repo}:{branch}"
            out.append({"id": repo_id, "name": f"{name}:{repo_id}"})
            self.ext_excludes[repo_id] = ext_excludes[index] if ext_excludes else None
            self.ext_includes[repo_id] = ext_includes[index] if ext_includes else None
            self.dir_excludes[repo_id] = dir_excludes[index] if dir_excludes else None
            self.dir_includes[repo_id] = dir_includes[index] if dir_includes else None

            logger.debug(f"Repo {index}: {repo_id}")
            logger.debug(f"ext_excludes: {self.ext_excludes[repo_id]}")
            logger.debug(f"ext_includes: {self.ext_includes[repo_id]}")
            logger.debug(f"dir_excludes: {self.dir_excludes[repo_id]}")
            logger.debug(f"dir_includes: {self.dir_includes[repo_id]}")

        return out

    def pipes(self) -> list[dict[str, str]]:
        out = self.get_repos()
        for repo_id in out:
            logger.info(f"llamaindex_ollama_github_pipeline - {repo_id['id']}")
        return out

    def _init_models(self):
        """
        Create models. Requires Azure Credentials. See the DefaultAzureCredential documentation for details
        of the authentication process (it will cascade through multiple authentication methods until it finds one that
        works, including a Workload Identity, an SPN via Env Vars, or az login credentials when running locally).
        """

        if self.valves.AZURE_OPENAI_MODEL_NAME is not None:
            default_credential = DefaultAzureCredential(exclude_environment_credential=True)
            token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

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
                self.llm = Ollama(model=self.valves.MODEL, base_url=self.valves.OLLAMA_HOST, request_timeout=300.0)
                logger.info(f"Created ollama llm: {self.valves.MODEL}")
            except Exception as e:
                return f"Error: {e}"

        if self.valves.AZURE_OPENAI_EMBED_MODEL_NAME is not None:
            default_credential = DefaultAzureCredential(exclude_environment_credential=True)
            token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

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
                    model_name=self.valves.EMBED_MODEL, base_url=self.valves.OLLAMA_HOST, request_timeout=120.0
                )
                logger.info(f"Created ollama embedding: {self.valves.EMBED_MODEL}")
            except Exception as e:
                return f"Error: {e}"

    async def _init_embeddings(self):

        if not self.valves.ENABLED:
            logger.warning(f"Pipeline disabled")
            return

        github_token = self.valves.GITHUB_TOKEN
        if not github_token:
            logger.warning(f"Github token must be configured")
            return

        try:
            github_client = GithubClient(github_token=github_token, verbose=self.valves.DEBUG)
        except Exception as e:
            logger.exception(f"Error connecting to GitHub: {self.valves.GITHUB_URL}")
            return

        self.indexes = {}
        repos = self.get_repos()
        for repo_info in repos:
            documents = []
            repo_id = repo_info["id"]
            logger.info(f"Start github embedding for {repo_id}")
            owner, repo, branch = repo_id.split(":")
            self._init_models()
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            Settings.chunk_size = DEFAULT_CHUNK_SIZE
            Settings.chunk_overlap = DEFAULT_CHUNK_OVERLAP

            include_file_extensions = [ext.strip() for ext in self.ext_includes[repo_id].split(",")] if self.ext_includes[repo_id] else []
            exclude_file_extensions = [ext.strip() for ext in self.ext_excludes[repo_id].split(",")] if self.ext_excludes[repo_id] else []
            include_directories = [dir.strip() for dir in self.dir_includes[repo_id].split(",")] if self.dir_includes[repo_id] else []
            exclude_directories = [dir.strip() for dir in self.dir_excludes[repo_id].split(",")] if self.dir_excludes[repo_id] else []

            filter_file_extensions = (
                (
                    include_file_extensions,
                    GithubRepositoryReader.FilterType.INCLUDE,
                )
                if include_file_extensions
                else (
                    (
                        exclude_file_extensions,
                        GithubRepositoryReader.FilterType.EXCLUDE,
                    )
                    if exclude_file_extensions
                    else None
                )
            )
            filter_directories = (
                (
                    include_directories,
                    GithubRepositoryReader.FilterType.INCLUDE,
                )
                if include_directories
                else (
                    exclude_directories,
                    GithubRepositoryReader.FilterType.EXCLUDE,
                )
                if exclude_directories
                else None
            )

            logger.info(f"Filters for {repo_id}:")
            logger.info(f"filter_file_extensions: {filter_file_extensions}")
            logger.info(f"filter_directories: {filter_directories}")

            reader = GithubRepositoryReader(
                github_client=github_client,
                owner=owner,
                repo=repo,
                use_parser=False,
                verbose=self.valves.DEBUG,
                filter_file_extensions=filter_file_extensions,
                filter_directories=filter_directories,
            )

            loop = asyncio.new_event_loop()
            reader._loop = loop

            try:
                # Load data from the branch
                logger.info(f"Start loading documents for {repo_id}.")
                documents = await asyncio.to_thread(reader.load_data, branch=branch)
                if not len(documents):
                    logger.warning(f"Error vectorizing for {repo_id} - 0 documents found for {repo_id}")
                    continue
                logger.info(f"Start vectorizing {len(documents)} documents for {repo_id}.")
                self.indexes[repo_id] = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True
                )
            except Exception as e:
                logger.exception(f"Error vectorizing for {repo_id} - {e}")
            finally:
                loop.close()

            if not self.indexes[repo_id]:
                logger.error(f"Error vectorizing for {repo_id} - VectorStoreIndex is empty")
                continue

            logger.info(f"Finished vectorizing {len(documents)} documents for {repo_id}.")

    async def on_valves_updated(self):
        logger.info(f"on_valves_update: {name}")
        self.pipelines = self.pipes()
        await self._init_embeddings()

    async def on_startup(self):
        logger.info(f"on_startup: {name}")
        self.pipelines = self.pipes()
        self.valves.ENABLED = False
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

        repo_id = model_id
        logger.debug(f"repo_id: {repo_id}")
        logger.debug(f"message: {messages}")
        logger.debug(f"user_message: {user_message}")

        if not self.indexes.get(repo_id, None):
            logger.error(f"Vector store index is not initialized for {repo_id}")
            return None
        try:
            streaming = not self.valves.MODEL.startswith("o")
            query_engine = self.indexes[repo_id].as_query_engine(
                streaming=streaming, similarity_top_k=0, vector_store_query_mode="default"
            )
            response = query_engine.query(user_message)
        except Exception as e:
            raise Exception(f"Exception in llamaindex_ollama_github_pipeline for {repo_id} - {e}")

        if streaming:
            return response.response_gen
        else:
            return response
