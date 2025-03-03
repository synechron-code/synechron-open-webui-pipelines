# Test a pipeline can run before uploading to server
import asyncio
from rag.llamaindex_ollama_gitlab_pipeline import Pipeline

async def main():
    pipeline = Pipeline()
    await pipeline.on_startup()
    await pipeline.on_valves_updated()

if __name__ == "__main__":
    asyncio.run(main())
