# TaskCompletionMetric will use an LLM judge to decide: "Did the agent actually 
# answer the user's question correctly?" It doesn't need a golden answer; it uses the input as the criterion.

import os
from langchain.chat_models import init_chat_model
from deepeval.models import OpenAIModel
from langchain.agents import create_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, StepEfficiencyMetric, PlanAdherenceMetric
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


model = init_chat_model(api_key=api_key,
                        base_url=base_url,
                        reasoning_effort="none", # This is the critical fix
                        model="gpt-6-luna")

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

metrics = [TaskCompletionMetric(model=judge_model)]


# The `TaskCompletionMetric` is passed into the `evals_iterator`.
# a streaming pattern that evaluates each golden as it runs the agent
for golden in dataset.evals_iterator(metrics=metrics):
    agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler()]},
    )


    '''
┌──────────────────────────────────────────────────────────┐
│  dataset.evals_iterator(metrics=[TaskCompletionMetric])  │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
       ┌──────────────────────────────────────┐
       │  For each golden in dataset:          │
       │                                       │
       │  1. yield golden                      │
       │           │                           │
       │           ▼                           │
       │  Your loop body:                      │
       │    agent.invoke(                     │
       │      messages=[golden.input],         │
       │      callbacks=[CallbackHandler()]    │  ← traces everything
       │    )                                  │
       │           │                           │
       │           ▼                           │
       │  2. DeepEval pulls the trace          │
       │     from CallbackHandler             │
       │           │                           │
       │           ▼                           │
       │  3. TaskCompletionMetric.measure()    │
       │     LLM judges: "task complete?"      │
       │           │                           │
       │           ▼                           │
       │  4. Store result, loop back           │
       └──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Print summary (pass rate, scores, reasons)              │
└──────────────────────────────────────────────────────────┘
'''