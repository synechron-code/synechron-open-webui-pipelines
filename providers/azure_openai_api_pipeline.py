"""
title: Azure OpenAI SDK
author: davidsewell
date: 2025-02-16
version: 0.1
license: MIT
description: A pipeline for integrating with Azure OpenAI using the Azure OpenAI API and Managed Identities.
requirements: typing_extensions>=4.12.2, openai>=1.63.0, azure-ai-inference, azure-identity, azure-core, pydantic>=2.10.6
environment_variables: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_MODEL, AZURE_OPENAI_API_DEBUG
"""

from http.client import HTTPConnection
import logging
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
import os

from openai import AzureOpenAI, ChatCompletion
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

name = "Azure OpenAI API"

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
        # You can add your custom valves here.
        DISABLED: bool = Field(
            default = False, description="Disable pipeline"
        )
        AZURE_OPENAI_API_KEY: Optional[str] = Field(
            default = None, description="Azure OpenAI key, if key is None, DefaultAzureCredential will retrieve key for Managed Identity"
        )
        AZURE_OPENAI_ENDPOINT: Optional[str] = Field(
            default = "your-azure-openai-endpoint-here", description="Azure OpenAI endpoint, blank Azure endpoint will enable Ollama endpoint and models"
        )
        AZURE_OPENAI_API_VERSION: Optional[str] = Field(
            default = "2025-01-01-preview", description="Azure OpenAI versions"
        )
        AZURE_OPENAI_MODELS: Optional[str] = Field(
            default = "gpt-4o-mini-payg", description="List of models separated by ';'"
        )
        AZURE_OPENAI_MODEL_NAMES: Optional[str] = Field(
            default = "text-embedding-3-large", description="List of deployment names separated by ';'"
        )
        ENABLE_DEBUG: bool = Field(
            default = False, description="Enable debug logging"
        )

    def __init__(self):
        self.type = "manifold"
        self.name = name
        self.valves = self.Valves()
        self.set_pipelines()

        self._enable_debug(self.valves.ENABLE_DEBUG)

        if self.valves.DISABLED:
            return

        self.client = self._openai_client()

        pass

    def _enable_debug(self, enable: bool = False):
        if enable:
            logger.setLevel(logging.DEBUG)
            logger.debug("enable debug logging")
            # Enable HTTPConnection debug logging to the console.
            HTTPConnection.debuglevel = 1
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger("openai").setLevel(logging.DEBUG)
            logging.getLogger("urllib3").setLevel(logging.DEBUG)
        else:
            logger.debug("disable debug logging")
            logger.setLevel(logging.INFO)
            # Enable HTTPConnection debug logging to the console.
            HTTPConnection.debuglevel = 0
            logging.basicConfig(level=logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("urllib3").setLevel(logging.INFO)

    def _openai_client(self) -> AzureOpenAI:
        """
        Create an OpenAI client. Requires Azure Credentials. See the DefaultAzureCredential documentation for details
        of the authentication process (it will cascade through multiple authentication methods until it finds one that
        works, including a Workload Identity, an SPN via Env Vars, or az login credentials when running locally).

        :return: AzureOpenAI client
        """
        default_credential = DefaultAzureCredential(exclude_environment_credential=True)
        token = default_credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        )

        try:
            client = AzureOpenAI(
                api_version=self.valves.AZURE_OPENAI_API_VERSION,
                azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                api_key=self.valves.AZURE_OPENAI_API_KEY or token.token
            )
            logger.debug("client created")
        except Exception as e:
            return f"Error: {e}"

        return client

    def set_pipelines(self):
        models = self.valves.AZURE_OPENAI_MODELS.split(";")
        model_names = self.valves.AZURE_OPENAI_MODEL_NAMES.split(";")
        self.pipelines = [
            {"id": model, "name": name} for model, name in zip(models, model_names)
        ]
        logger.info(f"azure_openai_api_pipeline - models: {self.pipelines}")
        pass

    async def on_valves_updated(self):
        logger.debug(f"on_valves_updated: {name}")
        logger.debug(self.valves)
        self.set_pipelines()
        if self.valves.DISABLED:
            return
        self._enable_debug(self.valves.ENABLE_DEBUG)
        self.client = self._openai_client()

    async def on_startup(self):
        # This function is called when the server is started.
        logger.debug(f"on_startup:{name}")
        logger.debug(self.valves)
        if self.valves.DISABLED:
            return
        self.client = self._openai_client()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        logger.debug(f"on_shutdown:{name}")
        pass

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.

        if self.valves.DISABLED:
            return "WARNING: Pipeline DISABLED"

        # remap user field
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"]["id"] if "id" in body["user"] else str(body["user"])

        stream = body.get("stream", False)

        # o1 and o1-mini don't alow stream = True!
        o_model = False
        if model_id.startswith("o"):
            o_model = True
            stream = False

        # Base parameters for the API call
        parameters = {
            "model": model_id,
            "messages": messages,
            "temperature": body.get("temperature", 1),
            "top_p": body.get("top_p", 1),
            "frequency_penalty": body.get("frequency_penalty", 0),
            "presence_penalty": body.get("presence_penalty", 0),
            "user": body.get("user", None)
        }

        if o_model:
            max_tokens = body.get("max_tokens") or body.get("max_completion_tokens", 4000) # fix this
            parameters["max_completion_tokens"] = max_tokens
        else:
            parameters["max_tokens"] = body.get("max_tokens", 4000)
            parameters["stream"] = stream


        response: ChatCompletion = None
        try:
            logger.debug(f"{parameters=}")
            response = self.client.chat.completions.create(**parameters)
            logger.debug(f"{response=}")

            if stream:
                return self.stream_response(response)
            else:
                return response.choices[0].message.content

        except Exception as e:
            if response:
                text = response.choices[0].message.content
                return f"Error: {e} ({text})"
            else:
                return f"Error: {e}"

    def stream_response(self, response: ChatCompletion):
        for chunk in response:
            choices = chunk.choices
            if choices and len(choices) > 0:
                content = choices[0].delta.content
                if content:
                    yield content