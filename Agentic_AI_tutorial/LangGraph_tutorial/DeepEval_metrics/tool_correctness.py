
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import (
    LLMTestCase,
    ToolCall,
    ToolCallParams,
)
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key or not api_key.strip():
    raise ValueError("OPENAI_API_KEY is missing or empty")
if not base_url or not base_url.strip():
    raise ValueError("BASE_URL is missing or empty")


# Step 1: Define tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Step 2: Create agent
model = init_chat_model(api_key=api_key,
                        base_url=base_url,
                        reasoning_effort="none", # This is the critical fix
                        model="gpt-6-luna")

agent = create_agent(
    model=model,
    tools=[multiply],
    system_prompt="Be concise. Use tools for any math questions.",
)


# Step 3: Define goldens
dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="What is 8123 multiplied by 62323?",
            expected_tools=[
                ToolCall(
                    name="multiply",
                    input_parameters={
                        "a": 8123,
                        "b": 62323,
                    },
                )
            ],
        ),
        Golden(
            input="What is 7 multiplied by 92323?",
            expected_tools=[
                ToolCall(
                    name="multiply",
                    input_parameters={
                        "a": 7,
                        "b": 92323,
                    },
                )
            ],
        ),
    ]
)


# Step 4: Execute agent and convert goldens to test cases
for golden in dataset.goldens:

    print(f"\nEvaluating: {golden.input}")

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": golden.input}
            ]
        }
    )

    actual_output = result["messages"][-1].content

    # Collect real tool calls from LangChain messages
    tools_called = []

    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tools_called.append(
                ToolCall(
                    name=tool_call["name"],
                    input_parameters=tool_call["args"],
                )
            )

    # Convert Golden into LLMTestCase
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        tools_called=tools_called,
        expected_tools=golden.expected_tools,
    )

    dataset.add_test_case(test_case)

    print("Agent Response:", actual_output)
    print("Tools Called:", tools_called)
    print("Expected Tools:", golden.expected_tools)


# Step 5: Configure metric
metric = ToolCorrectnessMetric(
    threshold=1.0,
    evaluation_params=[
        ToolCallParams.INPUT_PARAMETERS
    ],
    should_exact_match=True,
    include_reason=True,
    verbose_mode=True,
)

# Step 6: Configure accuracy metric
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import OpenAIModel

judge_model = OpenAIModel(
    model="gpt-4.1-mini",
    api_key=api_key,
    base_url=base_url,
)

accuracy = GEval(
    name="AnswerCorrectness",
    criteria="Determine if the actual_output is mathematically correct for the given input.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.9,
    model=judge_model, 
)



# Step 6: Evaluate test cases
evaluate(
    test_cases=dataset.test_cases,
    metrics=[metric, accuracy],
)
