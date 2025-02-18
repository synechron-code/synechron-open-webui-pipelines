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
from pydantic import BaseModel
import os

from openai import AzureOpenAI, ChatCompletion
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential


class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_OPENAI_API_KEY: Optional[str] = None
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_API_VERSION: str
        AZURE_OPENAI_MODELS: str
        AZURE_OPENAI_MODEL_NAMES: str
        AZURE_OPENAI_API_DEBUG: Optional[bool] = False

    def __init__(self):
        self.type = "manifold"
        self.name = "Azure OpenAI API: "
        self.valves = self.Valves(
            **{
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", None),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                "AZURE_OPENAI_MODELS": os.getenv("AZURE_OPENAI_MODELS", "gpt-4o-mini"),
                "AZURE_OPENAI_MODEL_NAMES": os.getenv("AZURE_OPENAI_MODEL_NAMES", "gpt-4o-mini"),
                "AZURE_OPENAI_API_DEBUG": os.getenv("AZURE_OPENAI_API_DEBUG", "False").lower() in ("1","true","yes"),
            }
        )

        self._enable_debug(self.valves.AZURE_OPENAI_API_DEBUG)

        self.client = self._openai_client()

        self.set_pipelines()
        pass

    def _enable_debug(self, enable: bool = False):
        if enable:
            print("AzureOpenAI enable debug logging")
            # Enable HTTPConnection debug logging to the console.
            HTTPConnection.debuglevel = 1
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger("openai").setLevel(logging.DEBUG)
            logging.getLogger("urllib3").setLevel(logging.DEBUG)
        else:
            print("AzureOpenAI disable debug logging")
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
            print("AzureOpenAI client created")
        except Exception as e:
            return f"Error: {e}"

        return client

    def set_pipelines(self):
        models = self.valves.AZURE_OPENAI_MODELS.split(";")
        model_names = self.valves.AZURE_OPENAI_MODEL_NAMES.split(";")
        self.pipelines = [
            {"id": model, "name": name} for model, name in zip(models, model_names)
        ]
        print(f"azure_openai_api_pipeline - models: {self.pipelines}")
        pass

    async def on_valves_updated(self):
        print(f"on_valves_update: {__name__}")
        print(self.valves)
        self._enable_debug(self.valves.AZURE_OPENAI_API_DEBUG)
        self.client = self._openai_client()
        self.set_pipelines()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        print(self.valves)
        self.client = self._openai_client()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe: {__name__}")

        print(f"model_id: {model_id}")
        print(f"messages: {messages}")
        print(f"user_message: {user_message}")
        print(f"body: {body}")

        # allowed_params = {'messages', 'temperature', 'role', 'content', 'contentPart', 'contentPartImage',
        #                   'enhancements', 'dataSources', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
        #                   'frequency_penalty', 'logit_bias', 'user', 'function_call', 'funcions', 'tools',
        #                   'tool_choice', 'top_p', 'log_probs', 'top_logprobs', 'response_format', 'seed'}

        # remap user field
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"]["id"] if "id" in body["user"] else str(body["user"])

        stream = body.get("stream", False)

        # o1 and o1-mini don't alow stream = True!
        if model_id in ("o1", "o1-mini"):
            stream = False

        # Base parameters for the API call
        parameters = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
            "temperature": body.get("temperature", 1),
            "max_completion_tokens": body.get("max_tokens", 4000),
            "top_p": body.get("top_p", 1),
            "frequency_penalty": body.get("frequency_penalty", 0),
            "presence_penalty": body.get("presence_penalty", 0),
            "user": body.get("user", None)
        }

        print(f"parameters: {parameters}")

        response: ChatCompletion = None
        try:
            response = self.client.chat.completions.create(**parameters)

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