# nexuchat-pipelines

Pipelines are a UI-Agnostic OpenAI API Plugin Framework developed by the Open WebUI open source project. They are designed to streamline the development process of OpenAI API plugins by providing a simple and intuitive way to create, manage, and deploy plugins. Pipelines are built on top of the OpenAI API and provide a powerful and flexible way to extend the functionality of the Nexus Chat platform.

Pipelines are python libraries that can be deployed into the Nexus Chat platform to extend its functionality. They can be used to create custom workflows, integrate with external services, and add new features to the platform. Pipelines are designed to be modular and extensible, allowing developers to easily create and deploy new plugins.

There is a large library of open source pipelines available on the Open WebUI community site (https://openwebui.com/)

This repository contains a library of Synechron proprietary pipeline plugins that can be deployed into the nexus chat platform.

### Examples of What You Can Achieve:

- [**Function Calling Pipeline**](https://github.com/open-webui/pipelines/blob/main/examples/filters/function_calling_filter_pipeline.py): Easily handle function calls and enhance your applications with custom logic.
- [**Custom RAG Pipeline**](https://github.com/open-webui/pipelines/blob/main/examples/pipelines/rag/llamaindex_pipeline.py): Implement sophisticated Retrieval-Augmented Generation pipelines tailored to your needs.
- [**Message Monitoring Using Langfuse**](https://github.com/open-webui/pipelines/blob/main/examples/filters/langfuse_filter_pipeline.py): Monitor and analyze message interactions in real-time using Langfuse.
- [**Rate Limit Filter**](https://github.com/open-webui/pipelines/blob/main/examples/filters/rate_limit_filter_pipeline.py): Control the flow of requests to prevent exceeding rate limits.
- [**Real-Time Translation Filter with LibreTranslate**](https://github.com/open-webui/pipelines/blob/main/examples/filters/libretranslate_filter_pipeline.py): Seamlessly integrate real-time translations into your LLM interactions.
- [**Toxic Message Filter**](https://github.com/open-webui/pipelines/blob/main/examples/filters/detoxify_filter_pipeline.py): Implement filters to detect and handle toxic messages effectively.
- **And Much More!**: The sky is the limit for what you can accomplish with Pipelines and Python. [Check out our scaffolds](https://github.com/open-webui/pipelines/blob/main/examples/scaffolds) to get a head start on your projects and see how you can streamline your development process!
