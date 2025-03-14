import os
from dotenv import load_dotenv

load_dotenv()
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
    set_default_openai_client,
)
import asyncio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
custom_openai_client = AsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPENAI_API_KEY,
)

set_default_openai_client(custom_openai_client)
set_tracing_disabled(True)


async def run_multi_agent_models():
    spanish_agent = Agent(
        name="Spanish agent",
        instructions="You only speak Spanish.",
        model="o3-mini",
    )

    english_agent = Agent(
        name="English agent",
        instructions="You only speak English",
        model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=AsyncOpenAI()),
    )

    triage_agent = Agent(
        name="Triage agent",
        instructions="Handoff to the appropriate agent based on the language of the request.",
        handoffs=[spanish_agent, english_agent],
        model="gpt-3.5-turbo",
    )

    result = await Runner.run(triage_agent, "Hola, ¿cómo estás?")
    print(result.final_output)


async def run_other_openai_client_as_agent():
    client = AsyncOpenAI()
    MODEL = "gpt-3.5-turbo"

    @function_tool
    def get_weather(city: str):
        print(f"[debug] Getting weather for {city}")
        return f"The weather in {city} is 75 degrees and sunny."

    agent = Agent(
        name="Weather agent",
        instructions="You only respond in haikus.",
        model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo??")
    print(result.final_output)


def run_agent_default_config():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    result = Runner.run_sync(agent, "Write a haiku about the ocean.")
    print(result.final_output)


if __name__ == "__main__":
    print("Default example:")
    run_agent_default_config()
    print("\nMulti-agent example:")
    asyncio.run(run_multi_agent_models())
    print("\nOther OpenAI client example:")
    asyncio.run(run_other_openai_client_as_agent())
