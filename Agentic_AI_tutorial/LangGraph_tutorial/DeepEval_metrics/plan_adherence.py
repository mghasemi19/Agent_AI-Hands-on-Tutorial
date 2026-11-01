from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from deepeval.models import OpenAIModel
import os
from deepeval.integrations.langchain import CallbackHandler
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, StepEfficiencyMetric, PlanAdherenceMetric
from dotenv import load_dotenv

load_dotenv()

# Use the same OpenAI-compatible endpoint and GPT model as tool_correctness.py.
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key or not api_key.strip():
    raise ValueError("OPENAI_API_KEY is missing or empty")
if not base_url or not base_url.strip():
    raise ValueError("BASE_URL is missing or empty")

model = init_chat_model(
    api_key=api_key,
    base_url=base_url,
    reasoning_effort="none",
    model="gpt-6-luna",
)

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = create_agent(
    model=model,
    tools=[multiply],
    system_prompt="Be concise.",
)

# Goldens are the inputs you want to evaluate.
dataset = EvaluationDataset(goldens=[
    Golden(input="What is 8 multiplied by 6?"),
    Golden(input="What is 7 multiplied by 9?"),
])

judge_model = OpenAIModel(
    model="gpt-4.1-mini",
    api_key=api_key,
    base_url=base_url,
)

metrics = [PlanAdherenceMetric(judge_model)]


# The `TaskCompletionMetric` is passed into the `evals_iterator`.
for golden in dataset.evals_iterator(metrics=metrics):
    agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler()]},
    )