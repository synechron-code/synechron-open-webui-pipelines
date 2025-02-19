from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel
import requests
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

def dump_response(response):
    print("Status Code:", response.status_code)
    print("Headers:", response.headers)
    print("Content:", response.content)
    print("Text:", response.text)
    print("JSON:", response.json() if response.headers.get('Content-Type') == 'application/json' else "Not a JSON response")
    print("URL:", response.url)
    print("History:", response.history)
    print("Elapsed Time:", response.elapsed)
    print("Request Headers:", response.request.headers)
    print("Request Body:", response.request.body)

class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_OPENAI_API_KEY: Optional[str] = ""
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_API_VERSION: str
        AZURE_OPENAI_MODELS: str
        AZURE_OPENAI_MODEL_NAMES: str

    def __init__(self):
        self.type = "manifold"
        self.name = "Azure OpenAI: "
        self.valves = self.Valves(
            **{
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                "AZURE_OPENAI_MODELS": os.getenv("AZURE_OPENAI_MODELS", "gpt-4o;gpt-4o-mini"),
                "AZURE_OPENAI_MODEL_NAMES": os.getenv("AZURE_OPENAI_MODEL_NAMES", "GPT-4o;GPT-4o-MINI"),
            }
        )
        self.bearer_token_provider = self._get_token()
        self.set_pipelines()
        pass

    def _get_token(self):
        try:
            return get_bearer_token_provider(
                DefaultAzureCredential(exclude_environment_credential=True),
                "https://cognitiveservices.azure.com/.default"
            )
        except Exception as e:
            print(f"Error getting Azure credentials: {e}")
            raise e

    def set_pipelines(self):
        models = self.valves.AZURE_OPENAI_MODELS.split(";")
        model_names = self.valves.AZURE_OPENAI_MODEL_NAMES.split(";")
        self.pipelines = [
            {"id": model, "name": name} for model, name in zip(models, model_names)
        ]
        print(f"azure_openai_manifold_pipeline - models: {self.pipelines}")
        pass

    async def on_valves_updated(self):
        self.set_pipelines()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
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

        headers = {
            "Content-Type": "application/json",
        }

        if self.valves.AZURE_OPENAI_API_KEY:
            headers["api-key"] = self.valves.AZURE_OPENAI_API_KEY
        else:
            headers["Authorization"] = 'Bearer ' + self.bearer_token_provider()

        url = f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/{model_id}/chat/completions?api-version={self.valves.AZURE_OPENAI_API_VERSION}"

        allowed_params = {'messages', 'temperature', 'role', 'content', 'contentPart', 'contentPartImage',
                          'enhancements', 'dataSources', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
                          'frequency_penalty', 'logit_bias', 'user', 'function_call', 'funcions', 'tools',
                          'tool_choice', 'top_p', 'log_probs', 'top_logprobs', 'response_format', 'seed'}

        # Azure OpenAI reasoning models (o-models) do not support the following params
        o_model_not_allowed_params = {"stream", "temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs", "logit_bias", "max_tokens"}

        # remap user field
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"]["id"] if "id" in body["user"] else str(body["user"])

        # Remove not allowed params for o-models
        if model_id.startswith("o"):
            if "max_tokens" in body and "max_completion_tokens" not in body:
                body["max_completion_tokens"] = body["max_tokens"]
            filtered_body = {k: v for k, v in body.items() if k in allowed_params and k not in o_model_not_allowed_params}
        else:
            filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        # log fields that were filtered out as a single line
        if len(body) != len(filtered_body):
            print(f"Dropped params: {', '.join(set(body.keys()) - set(filtered_body.keys()))}")

        # Log the request details
        print(f"Request URL: {url}")
        print(f"Request Headers: {headers}")
        print(f"Request Body: {filtered_body}")
        
        try:
            r = requests.post(
                url=url,
                json=filtered_body,
                headers=headers,
                stream=True,
            )
            
            dump_response(r)

            r.raise_for_status()
            if body["stream"]:
                iter = r.iter_lines()
                return iter
            else:
                return r.json()
        except Exception as e:
            if r:
                text = r.text
                print(f"Error: {e} ({text})")
                return f"Error: {e} ({text})"
            else:
                print(f"Error: {e}")
                return f"Error: {e}"
