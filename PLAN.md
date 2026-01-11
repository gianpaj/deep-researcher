# Multi-Deep Research Platform - Implementation Plan

## Overview

A Python-based multi-source deep research platform that aggregates research from **OpenAI, Google Gemini, Grok (xAI), and Perplexity** APIs using an agentic framework, with output in Markdown, HTML, or PDF formats.

---

## Recommended Stack

| Component | Choice | Justification |
|-----------|--------|---------------|
| Agentic Framework | **LangGraph** | Native parallel execution, explicit state management, production-ready checkpointing |
| PDF Generation | **WeasyPrint** | Pure Python, good CSS support, no external dependencies |
| Configuration | **Pydantic Settings** | Type-safe, .env support, validation |
| HTML Templates | **Jinja2** | Industry standard, powerful templating |

### Why LangGraph?

After analyzing LangGraph, CrewAI, and AutoGen:

| Criteria | LangGraph | CrewAI | AutoGen |
|----------|-----------|--------|---------|
| Parallel Execution | Native scatter-gather | Sequential by default | Manual orchestration |
| State Management | Explicit reducer-driven | Auto-managed | Conversation-based |
| Production Readiness | Battle-tested | Enterprise features | Requires custom deploy |
| Error Recovery | Built-in checkpointing | Limited | Manual |

---

## Project Structure

```
massive-researcher/
├── pyproject.toml                    # Dependencies and metadata
├── .env.example                      # Environment variable template
├── README.md                         # Documentation
├── Makefile                          # Common commands
│
├── src/
│   └── massive_researcher/
│       ├── __init__.py
│       ├── main.py                   # CLI entry point
│       ├── config.py                 # Pydantic configuration
│       │
│       ├── agents/                   # Research agents
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract base agent
│       │   ├── openai_agent.py       # OpenAI Deep Research
│       │   ├── google_agent.py       # Google Gemini Deep Research
│       │   ├── grok_agent.py         # xAI Grok agent
│       │   └── perplexity_agent.py   # Perplexity Search agent
│       │
│       ├── graph/                    # LangGraph orchestration
│       │   ├── __init__.py
│       │   ├── state.py              # Graph state definitions
│       │   ├── nodes.py              # Graph node functions
│       │   └── workflow.py           # Main workflow assembly
│       │
│       ├── synthesizer/              # Result aggregation
│       │   ├── __init__.py
│       │   └── merger.py             # Merge and synthesize results
│       │
│       ├── output/                   # Output generation
│       │   ├── __init__.py
│       │   ├── markdown.py           # Markdown output
│       │   ├── html.py               # HTML template rendering
│       │   └── pdf.py                # PDF generation
│       │
│       ├── templates/                # Jinja2 templates
│       │   ├── base.html
│       │   ├── research_report.html
│       │   └── styles.css
│       │
│       └── utils/                    # Utilities
│           ├── __init__.py
│           ├── rate_limiter.py       # API rate limiting
│           └── retry.py              # Retry logic with backoff
│
├── tests/
│   ├── conftest.py
│   ├── test_agents/
│   ├── test_graph/
│   └── test_output/
│
└── examples/
    ├── basic_research.py
    └── custom_workflow.py
```

---

## Architecture Diagram

```
                                    ┌─────────────────────────────┐
                                    │        User Input           │
                                    │     (Topic + Context)       │
                                    └──────────────┬──────────────┘
                                                   │
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │     LangGraph Workflow      │
                                    │   (Parallel Orchestration)  │
                                    └──────────────┬──────────────┘
                                                   │
       ┌───────────────────┬───────────────────────┼───────────────────────┬───────────────────┐
       │                   │                       │                       │                   │
       ▼                   ▼                       ▼                       ▼                   ▼
┌──────────────┐   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐   ┌──────────────┐
│    OpenAI    │   │    Google    │       │     Grok     │       │  Perplexity  │   │   (Future)   │
│Deep Research │   │Deep Research │       │  (Synthesis) │       │   (Search)   │   │    Claude    │
│              │   │              │       │              │       │              │   │              │
│ o4-mini-dr   │   │ gemini-dr    │       │   grok-4     │       │  sonar-pro   │   │     ...      │
│  $10-30/q    │   │  $2-5/q      │       │   $1-3/q     │       │   $1-3/q     │   │              │
└──────┬───────┘   └──────┬───────┘       └──────┬───────┘       └──────┬───────┘   └──────────────┘
       │                  │                      │                      │
       └──────────────────┴──────────────────────┴──────────────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────────┐
                                    │     Result Synthesizer      │
                                    │  (Merge + Dedupe + Rank)    │
                                    └──────────────┬──────────────┘
                                                   │
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │     Output Generator        │
                                    │                             │
                                    │  ┌────────┬────────┬─────┐  │
                                    │  │Markdown│  HTML  │ PDF │  │
                                    │  └────────┴────────┴─────┘  │
                                    └──────────────┬──────────────┘
                                                   │
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │      Research Report        │
                                    │    (.md / .html / .pdf)     │
                                    └─────────────────────────────┘
```

---

## Key Components

### 1. Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from enum import Enum

class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"

class Settings(BaseSettings):
    # API Keys
    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    google_api_key: SecretStr = Field(..., env="GOOGLE_API_KEY")
    xai_api_key: SecretStr = Field(..., env="XAI_API_KEY")
    perplexity_api_key: SecretStr = Field(..., env="PERPLEXITY_API_KEY")

    # Model Configuration
    openai_model: str = "o4-mini-deep-research-2025-06-26"  # Cheaper default
    google_model: str = "deep-research-pro-preview-12-2025"
    grok_model: str = "grok-4"
    perplexity_model: str = "sonar-pro"

    # Rate Limiting
    openai_requests_per_minute: int = 10
    xai_requests_per_minute: int = 60
    perplexity_requests_per_minute: int = 100

    # Output
    default_output_format: OutputFormat = OutputFormat.MARKDOWN
    output_directory: str = "./output"

    class Config:
        env_file = ".env"
```

### 2. Graph State (`graph/state.py`)

```python
from typing import TypedDict, Annotated, List, Optional
from operator import add
from dataclasses import dataclass
from enum import Enum

class ResearchStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Citation:
    title: str
    url: str
    source: str  # "openai", "grok", "perplexity"
    accessed_at: str

@dataclass
class ResearchResult:
    source: str
    content: str
    citations: List[Citation]
    status: ResearchStatus
    error_message: Optional[str] = None

class ResearchState(TypedDict):
    topic: str
    additional_context: Optional[str]
    results: Annotated[List[ResearchResult], add]  # Reducer for parallel merge
    synthesized_content: Optional[str]
    all_citations: List[Citation]
    errors: Annotated[List[str], add]
```

### 3. Base Agent (`agents/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional, List
import asyncio

class BaseResearchAgent(ABC):
    def __init__(self, api_key: str, model: str, timeout: int = 300):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @abstractmethod
    async def research(self, topic: str, context: Optional[str] = None) -> ResearchResult:
        pass

    @abstractmethod
    def extract_citations(self, raw_response: dict) -> List[Citation]:
        pass

    async def safe_research(self, topic: str, context: Optional[str] = None) -> ResearchResult:
        """Wrapper with timeout and error handling."""
        try:
            return await asyncio.wait_for(
                self.research(topic, context),
                timeout=self.timeout
            )
        except Exception as e:
            return ResearchResult(
                source=self.source_name,
                content="",
                citations=[],
                status=ResearchStatus.FAILED,
                error_message=str(e)
            )
```

### 4. Research Agents

| Agent | API | Model | Features | Cost/Query |
|-------|-----|-------|----------|------------|
| `OpenAIResearchAgent` | Responses API | o4-mini-deep-research | Web search, code interpreter | $10-30 |
| `GoogleResearchAgent` | Interactions API | deep-research-pro-preview | Web search, file search, citations | **$2-5** |
| `GrokResearchAgent` | xAI Chat API | grok-4 | X/Twitter data, synthesis | $1-3 |
| `PerplexityResearchAgent` | OpenAI-compatible | sonar-pro | Multi-query, domain filtering | $1-3 |

#### Google Deep Research Agent (`agents/google_agent.py`)

```python
import time
from google import genai
from .base import BaseResearchAgent

class GoogleResearchAgent(BaseResearchAgent):
    """Agent using Google Gemini Deep Research via Interactions API."""

    def __init__(self, api_key: str, model: str = "deep-research-pro-preview-12-2025"):
        super().__init__(api_key, model, timeout=3600)  # 60 min max
        self.client = genai.Client(api_key=api_key)

    @property
    def source_name(self) -> str:
        return "google"

    async def research(self, topic: str, context: Optional[str] = None) -> ResearchResult:
        prompt = topic
        if context:
            prompt = f"{topic}\n\nContext: {context}"

        # Start background research task
        interaction = self.client.interactions.create(
            input=prompt,
            agent=self.model,
            background=True
        )

        # Poll for completion
        while True:
            interaction = self.client.interactions.get(interaction.id)
            if interaction.status == "completed":
                return ResearchResult(
                    source=self.source_name,
                    content=interaction.outputs[-1].text,
                    citations=self.extract_citations(interaction),
                    status=ResearchStatus.COMPLETED
                )
            elif interaction.status == "failed":
                return ResearchResult(
                    source=self.source_name,
                    content="",
                    citations=[],
                    status=ResearchStatus.FAILED,
                    error_message=interaction.error
                )
            await asyncio.sleep(10)
```

### 5. LangGraph Workflow (`graph/workflow.py`)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def create_research_workflow(providers: List[str] = None):
    if providers is None:
        providers = ["openai", "google", "grok", "perplexity"]

    workflow = StateGraph(ResearchState)

    # Node mapping
    node_functions = {
        "openai": ("openai_research", research_openai),
        "google": ("google_research", research_google),
        "grok": ("grok_research", research_grok),
        "perplexity": ("perplexity_research", research_perplexity),
    }

    # Add selected research nodes (run in parallel)
    for provider in providers:
        if provider in node_functions:
            name, func = node_functions[provider]
            workflow.add_node(name, func)
            workflow.add_edge(START, name)

    workflow.add_node("synthesize", synthesize_results)

    # All converge to synthesis
    for provider in providers:
        if provider in node_functions:
            name, _ = node_functions[provider]
            workflow.add_edge(name, "synthesize")

    workflow.add_edge("synthesize", END)

    return workflow.compile(checkpointer=MemorySaver())
```

### 6. Output Generators

| Format | Library | Features |
|--------|---------|----------|
| Markdown | Native | YAML frontmatter, bibliography section |
| HTML | Jinja2 + markdown | Templated, responsive, styled |
| PDF | WeasyPrint | Print CSS, page numbers, headers |

---

## Implementation Phases

### Phase 1: Project Setup
- [ ] Create `pyproject.toml` with all dependencies
- [ ] Set up `config.py` with Pydantic settings
- [ ] Create `.env.example` with API key placeholders
- [ ] Initialize directory structure

### Phase 2: Agent Implementation
- [ ] Implement `BaseResearchAgent` abstract class
- [ ] Implement `OpenAIResearchAgent` (Deep Research API)
- [ ] Implement `GrokResearchAgent` (xAI SDK)
- [ ] Implement `PerplexityResearchAgent` (Search API)
- [ ] Add `rate_limiter.py` and `retry.py` utilities

### Phase 3: LangGraph Workflow
- [ ] Define `ResearchState` with TypedDict and reducers
- [ ] Implement graph nodes for each agent
- [ ] Build parallel execution workflow
- [ ] Add checkpointing for recovery

### Phase 4: Synthesis Engine
- [ ] Implement `ResultMerger` class
- [ ] Build LLM synthesis prompt
- [ ] Add citation normalization and deduplication

### Phase 5: Output Generation
- [ ] Implement `MarkdownGenerator`
- [ ] Create Jinja2 HTML templates
- [ ] Implement `PDFGenerator` with WeasyPrint
- [ ] Add CSS styling for all formats

### Phase 6: CLI and Polish
- [ ] Build CLI with argparse
- [ ] Add progress indicators
- [ ] Write README documentation
- [ ] Create example scripts

---

## Dependencies

```toml
[project]
name = "massive-researcher"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # Agentic Framework
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",

    # API Clients
    "openai>=1.40.0",           # For OpenAI AND Perplexity (OpenAI-compatible)
    "google-genai>=1.0.0",      # For Google Gemini Deep Research
    "httpx>=0.27.0",            # For xAI/Grok API calls

    # Configuration
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",

    # Output Generation
    "jinja2>=3.1.0",
    "markdown>=3.5.0",
    "pygments>=2.17.0",

    # Utilities
    "tenacity>=8.2.0",
    "structlog>=24.0.0",
    "rich>=13.0.0",             # Progress display
    "diskcache>=5.6.0",         # Result caching
]

[project.optional-dependencies]
pdf = ["weasyprint>=60.0"]      # Requires system deps

[project.scripts]
massive-researcher = "massive_researcher.main:main"
```

---

## CLI Usage

```bash
# Basic research (uses all 4 providers)
massive-researcher "What are the latest advances in quantum computing?"

# Specify output format
massive-researcher "AI in healthcare" --format pdf --output report.pdf

# Select specific providers
massive-researcher "Climate change" --providers openai google perplexity

# Budget-friendly (skip expensive OpenAI, use Google + Perplexity)
massive-researcher "Climate change" --providers google perplexity --lite

# Add context
massive-researcher "Machine learning" \
  --context "Focus on transformer architectures in 2024-2025" \
  --format html

# Dry run to see estimated cost
massive-researcher "Topic" --dry-run
# Output: Estimated cost: $8-15 (OpenAI: $10, Google: $3, Perplexity: $2)
```

---

## Environment Variables

```bash
# .env.example
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...              # From Google AI Studio
XAI_API_KEY=xai-...
PERPLEXITY_API_KEY=pplx-...

# Optional overrides
OPENAI_MODEL=o4-mini-deep-research-2025-06-26
GOOGLE_MODEL=deep-research-pro-preview-12-2025
GROK_MODEL=grok-4
PERPLEXITY_MODEL=sonar-pro
```

---

## Testing Strategy

1. **Unit tests**: Mock API responses for each agent
2. **Integration tests**: Run workflow with real APIs (rate-limited)
3. **Output validation**: Verify each format renders correctly
4. **CLI tests**: All argument combinations work
5. **Manual QA**: Research a topic and review quality

---

## Open Questions

1. **Streaming output**: Should research results stream in real-time as each provider responds?
2. **Web UI**: CLI-only for now, or add a simple web interface?
3. **HTML template**: Any specific design/branding requirements?
4. **Error handling**: Omit failed providers from output, or show errors in report?

---

## Critical Review: Risks, Gaps & Suggestions

### 🚨 Critical Risks

#### 1. **HIGH COST - OpenAI Deep Research**
| Issue | Impact | Severity |
|-------|--------|----------|
| Cost per call: **$10-30 USD** | A single research session with 3 providers could cost $30-50+ | **CRITICAL** |
| No budget controls in plan | Users could accidentally spend hundreds of dollars | **HIGH** |

**Mitigation:**
- Add `max_cost_per_session` config (default: $5)
- Implement cost estimation before execution
- Add `--dry-run` flag to show estimated cost
- Default to `o4-mini-deep-research` ($2/M input) instead of `o3`

#### 2. **xAI DeepSearch NOT Available via API**
| Issue | Impact | Severity |
|-------|--------|----------|
| DeepSearch is X Premium+ only | Plan assumed web search capability that doesn't exist | **CRITICAL** |
| API web search is "Enterprise only (planned)" | Cannot deliver equivalent functionality | **HIGH** |

**Mitigation:**
- Use Grok for **reasoning/synthesis only**, not web search
- Pair Grok with Perplexity results for grounded responses
- Add disclaimer in docs about Grok limitations
- Consider Grok as optional/secondary provider

#### 3. **No Official Perplexity Python SDK**
| Issue | Impact | Severity |
|-------|--------|----------|
| Plan assumed `perplexity` package exists | Must use OpenAI-compatible client instead | **MEDIUM** |
| Community packages are unmaintained | Risk of breaking changes | **MEDIUM** |

**Mitigation:**
- Use `openai` client with custom `base_url="https://api.perplexity.ai"`
- Remove fake `perplexity` dependency from pyproject.toml
- Document the OpenAI-compatible approach

---

### ⚠️ Gaps in Original Plan

#### Gap 1: No Cost Tracking or Limits
**Problem:** Users have no visibility into costs until they get their API bill.

**Solution:** Add to `config.py`:
```python
# Cost Management
max_cost_per_session: float = 5.00  # USD
track_costs: bool = True
cost_warning_threshold: float = 1.00  # Warn when approaching limit
```

Add `utils/cost_tracker.py`:
```python
COST_PER_1K_TOKENS = {
    "o3-deep-research": {"input": 0.01, "output": 0.04},
    "o4-mini-deep-research": {"input": 0.002, "output": 0.008},
    "grok-4": {"input": 0.003, "output": 0.015},
    "sonar-pro": {"input": 0.003, "output": 0.015},
}
```

#### Gap 2: No Caching Layer
**Problem:** Repeated research on similar topics wastes money.

**Solution:** Add `utils/cache.py`:
- Cache research results by topic hash
- Configurable TTL (default: 24 hours)
- Option to force refresh: `--no-cache`

#### Gap 3: No Mock/Test Mode
**Problem:** Cannot develop or test without spending money on real API calls.

**Solution:** Add `--mock` flag and mock responses:
```python
# config.py
mock_mode: bool = Field(default=False, env="MOCK_MODE")
```

#### Gap 4: Missing Fallback Strategy
**Problem:** If one provider fails, entire research may be degraded.

**Solution:** Add fallback configuration:
```python
# If primary fails, try fallback
provider_fallbacks = {
    "openai": ["perplexity"],  # If OpenAI fails, use Perplexity
    "grok": ["openai"],
    "perplexity": ["openai"],
}
```

#### Gap 5: No Progress/Status Feedback
**Problem:** Deep research can take 2-5 minutes. Users see nothing.

**Solution:** Add `rich` for progress display:
```
[████████░░░░░░░░] 50% | OpenAI: Complete | Grok: In Progress | Perplexity: Pending
Estimated cost so far: $2.34
```

#### Gap 6: WeasyPrint System Dependencies
**Problem:** WeasyPrint requires system libraries (cairo, pango, gdk-pixbuf).

**Solution:**
- Document system requirements in README
- Add Dockerfile for containerized usage
- Consider `pdfkit` + `wkhtmltopdf` as alternative
- Make PDF generation optional (graceful degradation)

---

### 📊 Revised API Reality Check

| Provider | Original Assumption | Reality | Action Required |
|----------|---------------------|---------|-----------------|
| **OpenAI** | Standard chat API | Uses Responses API (different endpoint) | Update agent implementation |
| **OpenAI** | Reasonable cost | $10-30 per call | Add cost limits, default to o4-mini |
| **Google** | N/A (not in original) | Interactions API with polling, $2-5/call | **Add as primary provider** |
| **Google** | N/A | Uses `google-genai` SDK, background tasks | Implement async polling |
| **xAI/Grok** | Has web search | Web search is Enterprise-only | Remove web search, use for synthesis |
| **xAI/Grok** | `xai-sdk` package | SDK exists but limited | Verify SDK capabilities |
| **Perplexity** | `perplexity` package | No official SDK | Use OpenAI client with custom base_url |
| **Perplexity** | Search + Chat separate | Both via chat completions | Simplify implementation |

### 🆕 Google Gemini Deep Research (NEW)

| Aspect | Details |
|--------|---------|
| **API** | Interactions API (not Chat Completions) |
| **SDK** | `google-genai` (`from google import genai`) |
| **Agent** | `deep-research-pro-preview-12-2025` |
| **Cost** | **$2-5 per query** (best value!) |
| **Max Time** | 60 minutes per research task |
| **Features** | Web search, file search, detailed citations, structured output |
| **Status** | Public preview (GA expected soon) |

**Key Implementation Notes:**
- Must use `background=True` (async polling required)
- Poll every 10 seconds for completion
- Built-in web search (no additional tools needed)
- Citations included in response

---

### 💡 Suggestions for Improvement

#### Suggestion 1: Add "Lite" Mode
For cost-conscious users:
```bash
massive-researcher "topic" --lite  # Uses o4-mini + sonar (not pro)
```
Estimated cost: $1-3 vs $30-50

#### Suggestion 2: Provider Tiers
```python
PROVIDER_TIERS = {
    "premium": ["openai-o3", "google", "grok-4", "sonar-pro"],  # ~$40-60/query
    "standard": ["google", "grok-4", "sonar-pro"],              # ~$5-10/query (recommended)
    "budget": ["google", "sonar"],                               # ~$3-5/query
    "minimal": ["perplexity"],                                   # ~$1-2/query
}
```

**Recommendation**: Use "standard" tier by default - Google Deep Research provides excellent quality at $2-5/query, making it the best value.

#### Suggestion 3: Incremental Results
Don't wait for all providers - show results as they complete:
```
✓ Perplexity complete (12 citations) - 8 seconds
✓ Grok complete (3 insights) - 15 seconds
⏳ OpenAI Deep Research in progress... (typically 2-5 min)
```

#### Suggestion 4: Add Claude/Anthropic as Provider
Consider adding Claude with web search (via tool use) as a fourth option:
- More predictable pricing
- Strong reasoning capabilities
- Good citation extraction

#### Suggestion 5: Research Depth Levels
```bash
massive-researcher "topic" --depth quick    # 1 provider, fast
massive-researcher "topic" --depth standard # 2 providers
massive-researcher "topic" --depth deep     # All 3, full synthesis
```

---

### 🔧 Revised Dependencies

```toml
[project]
dependencies = [
    # Agentic Framework
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",

    # API Clients (corrected)
    "openai>=1.40.0",           # For OpenAI AND Perplexity (OpenAI-compatible)
    "google-genai>=1.0.0",      # For Google Gemini Deep Research (Interactions API)
    "httpx>=0.27.0",            # For xAI direct API calls

    # Configuration
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",

    # Output Generation
    "jinja2>=3.1.0",
    "markdown>=3.5.0",
    "pygments>=2.17.0",

    # Utilities
    "tenacity>=8.2.0",
    "structlog>=24.0.0",
    "rich>=13.0.0",             # Progress display
    "diskcache>=5.6.0",         # Result caching
]

[project.optional-dependencies]
pdf = ["weasyprint>=60.0"]      # Make PDF optional due to system deps
```

---

### 📋 Revised Project Structure

```
massive-researcher/
├── src/massive_researcher/
│   ├── agents/
│   │   ├── base.py
│   │   ├── openai_agent.py
│   │   ├── google_agent.py     # NEW: Google Gemini Deep Research
│   │   ├── grok_agent.py
│   │   └── perplexity_agent.py
│   └── utils/
│       ├── rate_limiter.py
│       ├── retry.py
│       ├── cost_tracker.py     # NEW: Track API costs
│       ├── cache.py            # NEW: Result caching
│       └── progress.py         # NEW: Rich progress display
├── tests/
│   ├── mocks/                  # NEW: Mock API responses
│   │   ├── openai_mock.json
│   │   ├── google_mock.json    # NEW
│   │   ├── grok_mock.json
│   │   └── perplexity_mock.json
│   └── ...
├── Dockerfile                  # NEW: For WeasyPrint deps
└── ...
```

---

### ✅ Updated Implementation Phases

#### Phase 1: Project Setup (Updated)
- [ ] Create `pyproject.toml` with corrected dependencies
- [ ] Set up `config.py` with **cost limits** and **mock mode**
- [ ] Create `.env.example` with all API keys (including GOOGLE_API_KEY)
- [ ] Add `Dockerfile` for containerized usage
- [ ] Initialize directory structure

#### Phase 2: Core Utilities (NEW)
- [ ] Implement `cost_tracker.py` with per-provider costs
- [ ] Implement `cache.py` with disk-based caching
- [ ] Implement `progress.py` with rich progress bars
- [ ] Add mock response fixtures for testing (all 4 providers)

#### Phase 3: Agent Implementation (Updated)
- [ ] Implement `BaseResearchAgent` with cost tracking hooks
- [ ] Implement `OpenAIResearchAgent` using **Responses API** (not chat)
- [ ] Implement `GoogleResearchAgent` using **Interactions API** with polling
- [ ] Implement `GrokResearchAgent` via **httpx** (synthesis only, no web search)
- [ ] Implement `PerplexityResearchAgent` via **OpenAI client** with custom base_url
- [ ] Add `rate_limiter.py` and `retry.py` utilities

#### Phase 4-6: (Same as before)

---

### 🎯 Risk Mitigation Summary

| Risk | Mitigation | Priority |
|------|------------|----------|
| High API costs | Add cost limits, default to cheaper models | **P0** |
| xAI no web search | Use Grok for synthesis only, document limitation | **P0** |
| No Perplexity SDK | Use OpenAI-compatible client | **P1** |
| WeasyPrint deps | Make PDF optional, add Dockerfile | **P1** |
| No test mode | Add mock fixtures and `--mock` flag | **P1** |
| No caching | Add diskcache-based result caching | **P2** |
| No progress feedback | Add rich progress display | **P2** |

---

## References

- [OpenAI Deep Research API](https://platform.openai.com/docs/guides/deep-research)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Google Gemini Deep Research API](https://ai.google.dev/gemini-api/docs/deep-research)
- [Google Deep Research Blog](https://blog.google/technology/developers/deep-research-agent-gemini-api/)
- [xAI API Documentation](https://docs.x.ai/docs/overview)
- [xAI Search Tools (Enterprise)](https://docs.x.ai/docs/guides/tools/search-tools)
- [Perplexity API (OpenAI-compatible)](https://docs.perplexity.ai/api-reference/chat-completions-post)
- [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)
- [WeasyPrint Documentation](https://weasyprint.org/)
