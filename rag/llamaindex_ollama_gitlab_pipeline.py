"""
title: Llama Index Ollama Gitlab Pipeline
author: David.Sewell
date: 2025-03-02
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitLab repository.
requirements: python-gitlab, llama-index-embeddings-ollama, llama-index-embeddings-azure-openai, llama-index-readers-gitlab
"""

import logging
from typing import List, Union, Generator, Iterator, Optional, Sequence
from pydantic import BaseModel, Field
import asyncio
import gitlab

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.readers.gitlab import GitLabRepositoryReader, GitLabIssuesReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from azure.identity import DefaultAzureCredential


name = "GitlabRAG"

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
        GITLAB_TOKEN: str = Field(default="", description="Gitlab PAT with read access to repo to be accessed")
        GITLAB_URL: str = Field(default="https://gitlab.com", description="Gitlab URL")
        GITLAB_PROJECT_PATHS: str = Field(
            default="", description="List of Gitlab project paths separated by ';' i.e. http://gitlab.com/<project path>"
        )
        GITLAB_PATHS: Optional[str] = Field(
            default="", description="List of Gitlab project folders separated by ';'"
        )
        GITLAB_REFS: str = Field(
            default="HEAD", description="List of Gitlab References (Branches or Commits) separated by ';' i.e. http://gitlab.com/<project path>/<path>/tree/"
        )
        GITLAB_CODE: bool = Field(
            default=True, description="Enable loading code into knowledge base"
        )
        GITLAB_ISSUES: bool = Field(
            default=True, description="Enable loading issues into knowledge base"
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
            default=None, description="List of file extensions to include in the filter separated by ';'"
        )
        EXCLUDE_FILE_EXTENSIONS: Optional[str] = Field(
            default=".png,.jpg,.jpeg,.gif,.svg,.ico,.json,.ipynb",
            description="List of file extensions to exclude in the filter separated by ';'",
        )
        INCLUDE_DIRECTORIES: Optional[str] = Field(
            default=None, description="List of directories to include in the filter separated by ';'"
        )
        EXCLUDE_DIRECTORIES: Optional[str] = Field(
            default=None, description="List of directories to exclude in the filter separated by ';'"
        )
        DEBUG: bool = Field(default=False, description="Enable debug logging")

    def __init__(self):
        self.type = "manifold"
        self.indexes = None
        self.embed_model = None
        self.llm = None
        self.gitlab_client = None
        self.pipelines = [{"id": f"{name}:NULL", "name": f"{name}:NULL"}]

        try:
            self.valves = self.Valves()
        except Exception as e:
            logger.exception(f"Error initializing Valves: {e}")

        if self.valves.DEBUG:
            logger.setLevel(logging.DEBUG)

        self.pipelines = self.pipes()
        pass

    def _init_gitlab(self):
        gitlab_token = self.valves.GITLAB_TOKEN

        if not gitlab_token:
            logger.error(f"GITLAB_TOKEN must be configured")
            return
        if not self.gitlab_client:
            try:
                self.gitlab_client = gitlab.Gitlab(self.valves.GITLAB_URL, private_token=gitlab_token)
                self.gitlab_client.auth()
            except Exception as e:
                logger.exception(f"Error connecting to Gitlab: {self.valves.GITLAB_URL}")

    def get_project_id(self, project):
        if self.gitlab_client:
            try:
                project_id = self.gitlab_client.projects.get(project).id
                logger.info(f"project: {project} => project id: {project_id}")
            except Exception as e:
                logger.exception(f"error retrieving project id for {project}: {e}")
                project_id = "NULL"
        else:
            logger.error(f"gitlab client not initialized")
            project_id = "NULL"
        return project_id

    def get_repos(self):
        projects = [project.strip() for project in self.valves.GITLAB_PROJECT_PATHS.split(';')]
        paths = [path.strip() for path in self.valves.GITLAB_PATHS.split(';')]
        refs = [ref.strip() for ref in self.valves.GITLAB_REFS.split(';')]
        projects = [project.replace("/", "__") for project in projects]
        paths = [paths.replace("/", "__") for paths in paths]
        if not (len(projects) == len(paths) == len(refs)):
            logger.error(
                f"The number of projects, repos, and branches must be the same. "
                f"projects: {len(projects)}, repos: {len(paths)}, branches: {len(refs)}"
            )
            return [{"id": f"{name}:NULL", "name": f"{name}:NULL"}]

        self._init_gitlab()

        out = []
        for project, path, ref in zip(projects, paths, refs):
            repo_id = f"{project}:{path}:{ref}"
            out.append({"id": repo_id, "name": f"{name}:{repo_id}"})

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

    def _get_vector_index(documents: Sequence[Document], id: str):
        try:
            if not len(documents):
                logger.warning(f"Error vectorizing for {id} - 0 documents found for {id}")
                return None
            logger.info(f"Start vectorizing {len(documents)} documents for {id}.")
            return VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
        except Exception as e:
            logger.exception(f"Error vectorizing for {id} - {e}")
        return None

    async def _init_embeddings(self):

        if not self.valves.ENABLED:
            logger.warning(f"Pipeline disabled")
            return

        self.indexes = {}
        repos = self.get_repos()
        for repo_info in repos:
            documents: Sequence[Document] = []
            repo_id = repo_info["id"]
            logger.info(f"Start creating knowledge base for {repo_id}")
            project, path, ref = repo_id.split(":")
            project = project.replace("__", "/")
            path = path.replace("__", "/")
            self._init_models()
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm

            loop = asyncio.new_event_loop()

            try:
                project_id = self.get_project_id(project)
                if self.valves.GITLAB_CODE:
                    project_reader = GitLabRepositoryReader(
                        gitlab_client=self.gitlab_client,
                        project_id=project_id,
                        verbose=self.valves.DEBUG,
                    )
                    project_reader._loop = loop
                    documents.extend(await asyncio.to_thread(project_reader.load_data, path=path, ref = ref, recursive = True))

                if self.valves.GITLAB_ISSUES:
                    project_issues_reader = GitLabIssuesReader(
                        gitlab_client=self.gitlab_client,
                        project_id=project_id,
                        verbose=self.valves.DEBUG,
                    )
                    project_issues_reader._loop = loop
                    documents.extend(await asyncio.to_thread(project_issues_reader.load_data, state = GitLabIssuesReader.IssueState.ALL))

                self.indexes[repo_id] = Pipeline._get_vector_index(documents, repo_id)
            except Exception as e:
                logger.exception(f"Error creating knowledge base for {repo_id} - {e}")
            finally:
                loop.close()

            if self.indexes[repo_id] is None:
                logger.error(f"Error creating knowledge base for {repo_id} - VectorStoreIndex is empty")
                continue

            logger.info(f"Finished creating knowledge base {len(documents)} documents for {repo_id}.")

    async def on_valves_updated(self):
        logger.debug(f"on_valves_update: {name}")
        self.pipelines = self.pipes()
        await self._init_embeddings()
        pass

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

        repo_id = model_id  # Strip the name element
        logger.debug(f"repo_id: {repo_id}")
        logger.debug(f"message: {messages}")
        logger.debug(f"user_message: {user_message}")
        response = None

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
