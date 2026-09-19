# HEPAgent

<img width="1400" height="800" alt="ChatGPT Image Sep 17, 2026, 06_31_01 AM" src="https://github.com/user-attachments/assets/36bdc2f0-0810-4bb2-a2f6-aea457d05456" />
<img width="1448" height="1086" alt="ChatGPT Image Sep 19, 2026, 11_18_52 AM" src="https://github.com/user-attachments/assets/c55e2ca7-e052-4356-b61e-c47e52345409" />


Talk to ROOT in plain English. RooAgent lets you run HEP analyses — histograms, event selection, fitting, significance scans — by typing what you want rather than writing code.

Under the hood, an LLM reads your prompt, picks the right analysis tool, fills in the arguments, and calls PyROOT. You get the result; ROOT does the work.

**Two ways to use it:**

- **LangGraph agent** — a standalone chatbot; pick a provider via `LLM_PROVIDER`: Claude (default) or ChatGPT.
- **MCP server** — drop it into a Claude/Codex CLI session.

## Quick start

Install from PyPI:

```
pip install rooagent
```

Or from source:

```
git clone https://github.com/amanmdesai/RooAgent.git
cd RooAgent
pip install .
```

`pip install .` installs everything needed for both modes — no extras.

## Operating modes

### Claude / Anthropic (default)

Pick a provider via `LLM_PROVIDER` and set its key env var:

| `LLM_PROVIDER` | Key env var | `MODEL` default |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-5` |
| `openai` | `OPENAI_API_KEY` | `gpt-5.5` |

`ROOAGENT_SEED` *(optional)* sets the sampling seed. Default: `7`.

```
export LLM_PROVIDER=anthropic   # default, can omit
export ANTHROPIC_API_KEY="..."
export MODEL=claude-sonnet-5    # optional override
export ROOAGENT_SEED=7          # optional sampling seed
```

To use OpenAI instead:

```
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
```

### Claude CLI via MCP

```
git clone https://github.com/amanmdesai/RooAgent.git
cd RooAgent
pip install .
claude mcp add rooagent -- rooagent-mcp
```

Then navigate to your ROOT files and start Claude:

```
cd /path/to/your/root/files
claude
```

### Codex CLI via MCP

```
git clone https://github.com/amanmdesai/RooAgent.git
cd RooAgent
pip install .

# Load ROOT first, replacing this path with your ROOT installation.
source /path/to/root/bin/thisroot.sh

ROOT_LIBDIR="$(root-config --libdir)"

codex mcp add rooagent \
  --env ROOTSYS="$ROOTSYS" \
  --env PATH="$ROOTSYS/bin:$PATH" \
  --env LD_LIBRARY_PATH="$ROOT_LIBDIR:${LD_LIBRARY_PATH:-}" \
  --env PYTHONPATH="$ROOT_LIBDIR:${PYTHONPATH:-}" \
  -- rooagent-mcp
  
```

If you previously registered RooAgent before loading ROOT, remove and re-add it:

```
codex mcp remove rooagent
```


## What it can do

- Inspection
- Counting
- Histograms
- Plotting
- Fitting
- Variables
- Statistics
- Statistical Inference
- Export

## Requirements

- Python >= 3.10
- CERN ROOT >= 6.34 (built with RooFit/RooStats)

## Dependencies

`pandas`, `numpy`, `scipy`, `matplotlib`, `mcp[cli]`, `typing_extensions`, `langchain-core`, `langgraph`, `langchain-anthropic`, `langchain-openai`

## How to cite

If you use RooAgent in your work, please cite:

```bibtex
@article{Desai:2026nmx,
    author = "Desai, Aman",
    title = "{RooAgent: An LLM Agent for Root-Based High Energy Physics Analysis}",
    eprint = "2605.17318",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "5",
    year = "2026"
}
```
