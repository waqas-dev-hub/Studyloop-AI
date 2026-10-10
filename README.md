# 🧠 StudyLoop AI — Truly Agentic Adaptive Study Planner

> An AI-powered study planner that **diagnoses misconceptions**, **decides teaching strategies**, and **adapts in real-time** — not just a quiz app with LLM calls.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LPU-orange.svg)](https://groq.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-green.svg)](https://smith.langchain.com)

---

## What Makes This Agentic (Not Just Another LLM App)

| Feature | Fixed App | This Agent |
|---------|-----------|------------|
| Wrong answer | Score drops → next question | Diagnoses **why** → teaches the misconception → then retests |
| Topic selection | Lowest score first | LLM **reasons**: "You've failed this 3x — let me explain before quizzing again" |
| Missed days | Spread evenly | Agent **trades off**: drops strong-topic reviews, doubles weak-topic sessions |
| Content quality | Generate and show | **Critic node** verifies content is in your notes, regenerates if hallucinated |
| Plan changes | Automatic | Agent **proposes** → you **approve** (human-in-the-loop) |

### The Agentic Loop

```
observe → diagnose → decide → act → verify → update → repeat
  │          │          │        │       │         │
  │          │          │        │       │         └─ Write to SQLite
  │          │          │        │       └─ Critic self-check
  │          │          │        └─ Call MCP tools (quiz, explain, replan...)
  │          │          └─ LLM chooses teaching action (6 options)
  │          └─ LLM analyzes misconceptions & patterns
  └─ Load mastery, history, time-to-exam from DB
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Agent** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful graph with conditional edges, decision nodes, loops |
| **LLM** | [Groq](https://groq.com) (meta-llama/llama-4-scout-17b-16e-instruct) | Ultra-fast inference via LPU |
| **Tools** | [MCP](https://modelcontextprotocol.io/) | 8 tools the agent calls autonomously |
| **Tracing** | [LangSmith](https://smith.langchain.com) | Full observability of every agent step |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | REST API serving the agent |
| **Frontend** | [Streamlit](https://streamlit.io/) | 5-page interactive dashboard |
| **Database** | SQLite (via SQLAlchemy) | Persistent storage for mastery, plans, history |
| **Doc Processing** | [LangChain](https://langchain.com/) | PDF/TXT loaders, text splitting, prompt templates |

---

## Project Structure

```
langgraph_project/
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── database.py               # SQLAlchemy ORM models (SQLite)
├── models.py                 # Pydantic schemas (API contracts)
├── mcp_server.py             # MCP tool server (8 tools)
│
├── agent/                    # LangGraph agent
│   ├── __init__.py           # Exports create_agent_graph, AgentState
│   ├── state.py              # AgentState TypedDict
│   ├── prompts.py            # All LLM prompt templates
│   ├── nodes.py              # Graph node functions (observe, diagnose, decide, act, verify, update)
│   └── graph.py              # StateGraph wiring with conditional edges
│
├── api/                      # FastAPI backend
│   ├── __init__.py
│   ├── main.py               # App entry point + CORS + startup
│   └── routes.py             # All API route handlers
│
├── streamlit_app.py          # Streamlit frontend (5 pages)
└── uploads/                  # Uploaded study materials
```

---

## Quick Start

### 1. Clone & Install

```bash
cd langgraph_project
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env and add your keys:
#   GROQ_API_KEY=gsk_...
#   LANGCHAIN_API_KEY=lsv2_...
```

### 3. Start the API

```bash
python -m api.main
# API runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Start the Frontend

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### 5. (Optional) Run MCP Server Standalone

```bash
python mcp_server.py
```

---

## LangGraph Agent — How It Works

### Graph Architecture

```
START → observe → diagnose → decide → execute → verify ─┬─→ approval → END (wait for human)
                                          ↑       │      │
                                          └───────┘      └─→ update → should_continue ─┬─→ observe (loop)
                                          (retry if              │                      │
                                           critic fails)         └──────────────────────┴─→ END
```

### Decision Node (`choose_teaching_action`)

The agent picks from 8 actions based on LLM reasoning:

| Action | When the agent chooses it |
|--------|--------------------------|
| `quiz` | Need to assess or reinforce knowledge |
| `explain` | Misconception detected — teach before quizzing again |
| `worked_example` | Student understands concept but fails application |
| `easier_question` | Student frustrated (multiple failures in a row) |
| `explain_back` | Verify deep understanding after an explanation |
| `move_on` | Topic mastery is sufficient, switch topics |
| `replan` | Missed days or mastery data shows plan isn't working |
| `mock_paper` | ≤ 7 days to exam |

### MCP Tools (8 tools)

| Tool | What it does |
|------|-------------|
| `search_notes` | Semantic search over uploaded documents |
| `generate_quiz` | Generate questions at agent-chosen difficulty |
| `grade_answer` | Grade with explanation + misconception detection |
| `get_mastery_data` | Read all scores from SQLite |
| `update_mastery` | Write scores + misconceptions to SQLite |
| `manage_study_plan` | Get/replan/mark study plan entries |
| `generate_mock_paper` | Full weighted mock exam |
| `create_explanation` | Targeted mini-lesson for a misconception |

---

## LangSmith Tracing

Every agent step is automatically traced in LangSmith:

- **observe_state** — What data the agent loaded
- **diagnose_weaknesses** — What misconceptions it found
- **choose_teaching_action** — Why it chose this action (full reasoning)
- **execute_action** — What content it generated
- **critic_verify** — Whether the critic passed or failed
- **update_state** — What was written to the database

View traces at: [smith.langchain.com](https://smith.langchain.com)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/subjects` | Create subject with exam date |
| GET | `/api/subjects` | List all subjects |
| POST | `/api/subjects/{id}/upload` | Upload PDF/TXT study materials |
| POST | `/api/session/start` | Start study session (invokes agent) |
| POST | `/api/session/{id}/answer` | Submit answer (agent grades & decides next) |
| GET | `/api/session/{id}/next` | Get next agent action |
| GET | `/api/subjects/{id}/plan` | Get study plan |
| POST | `/api/subjects/{id}/plan/replan` | Force replan |
| GET | `/api/subjects/{id}/dashboard` | Full analytics dashboard |
| POST | `/api/subjects/{id}/mock` | Generate mock paper |

Full Swagger docs at `http://localhost:8000/docs`

---

## Resume Bullet Points

> **StudyLoop AI** — Built a truly agentic AI study planner using **LangGraph**, **LangChain**, **Groq**, **MCP**, **FastAPI**, and **Streamlit**
> - Designed a 7-node LangGraph agent with conditional routing that diagnoses misconceptions, chooses from 8 pedagogical strategies, and self-verifies via a critic loop
> - Integrated **MCP** (Model Context Protocol) with 8 autonomous tools for quiz generation, grading, plan management, and misconception-targeted teaching
> - Built human-in-the-loop approval for study plan changes and a spaced-repetition algorithm with mastery tracking
> - Added **LangSmith** tracing for full observability of agent decision chains and LLM interactions
> - Deployed as a full-stack app with FastAPI backend, Streamlit dashboard, and SQLite persistence

---

## License

MIT
