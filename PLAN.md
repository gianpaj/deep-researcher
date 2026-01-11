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
│       ├── prompts/                  # System prompts for each agent
│       │   ├── __init__.py
│       │   ├── research_base.py      # Base research prompt template
│       │   ├── openai_prompt.py      # OpenAI-specific prompt
│       │   ├── google_prompt.py      # Google-specific prompt
│       │   ├── grok_prompt.py        # Grok-specific prompt
│       │   ├── perplexity_prompt.py  # Perplexity-specific prompt
│       │   └── synthesizer_prompt.py # Final synthesizer prompt
│       │
│       ├── graph/                    # LangGraph orchestration
│       │   ├── __init__.py
│       │   ├── state.py              # Graph state definitions
│       │   ├── nodes.py              # Graph node functions
│       │   └── workflow.py           # Main workflow assembly
│       │
│       ├── synthesizer/              # Result aggregation
│       │   ├── __init__.py
│       │   ├── merger.py             # Merge and synthesize results
│       │   ├── claim_extractor.py    # Extract claims from research
│       │   ├── claim_verifier.py     # Cross-reference and verify claims
│       │   └── grounding.py          # Search-based claim verification
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

---

## System Prompts

### Research Agent Prompts

Each research agent receives a structured system prompt to ensure consistent, high-quality output with proper citations.

#### Base Research Prompt Template (`prompts/research_base.py`)

```python
RESEARCH_SYSTEM_PROMPT_TEMPLATE = """You are an expert research analyst conducting deep research on behalf of a user.

## Your Task
Research the following topic thoroughly and provide a comprehensive, well-structured report.

## Research Guidelines
1. **Accuracy First**: Only include information you can verify. If uncertain, explicitly state the uncertainty level.
2. **Source Quality**: Prioritize authoritative sources (academic papers, official documentation, reputable news outlets).
3. **Recency**: Prefer recent information (2024-2025) unless historical context is needed.
4. **Multiple Perspectives**: Include diverse viewpoints, especially on controversial topics.
5. **Quantitative Data**: Include statistics, numbers, and data points where available.

## Output Structure
Your response MUST follow this structure:

### Executive Summary
[2-3 sentence overview of key findings]

### Key Findings
[Bulleted list of 5-10 main discoveries, each with source attribution]

### Detailed Analysis
[In-depth exploration organized by subtopic]

### Data & Statistics
[Relevant numbers, percentages, trends - with sources]

### Controversies & Limitations
[Areas of disagreement, gaps in research, or limitations]

### Sources
[List all sources with URLs in markdown format]

## Citation Format
- Inline citations: Use [Source Name](URL) format
- Every factual claim MUST have a citation
- Prefer primary sources over secondary

## Quality Checklist
Before submitting, verify:
- [ ] All claims have citations
- [ ] No speculation presented as fact
- [ ] Multiple sources corroborate major claims
- [ ] Recent data is prioritized
- [ ] Controversies are acknowledged
"""
```

#### OpenAI Deep Research Prompt (`prompts/openai_prompt.py`)

```python
OPENAI_RESEARCH_PROMPT = """You are OpenAI's Deep Research agent, powered by advanced reasoning capabilities.

{BASE_RESEARCH_PROMPT}

## OpenAI-Specific Instructions
1. **Use Web Search**: Actively search for current information. Don't rely solely on training data.
2. **Code Analysis**: If the topic involves code/technical concepts, use the code interpreter to verify claims.
3. **Reasoning Chain**: Show your reasoning process for complex analyses.
4. **Cross-Reference**: Verify claims across multiple authoritative sources.

## Your Unique Value
- Deep reasoning on complex topics
- Ability to synthesize information from multiple sources
- Technical accuracy for code/engineering topics
- Structured analytical approach

## Topic to Research
{topic}

{context_section}
"""
```

#### Google Gemini Deep Research Prompt (`prompts/google_prompt.py`)

```python
GOOGLE_RESEARCH_PROMPT = """You are Google's Gemini Deep Research agent with access to comprehensive web search.

{BASE_RESEARCH_PROMPT}

## Google-Specific Instructions
1. **Leverage Google Search**: Use your native search capabilities extensively.
2. **Google Scholar**: For academic topics, prioritize scholarly sources.
3. **Structured Data**: Extract and present data in tables where appropriate.
4. **Multimodal Context**: Consider images, charts, and visual data if relevant.

## Your Unique Value
- Access to Google's comprehensive search index
- Strong performance on factual queries
- Excellent at structured data extraction
- Good at synthesizing diverse source types

## Citation Requirements
- Include URLs for ALL sources
- Prefer .edu, .gov, and established news domains
- Note publication dates for all sources

## Topic to Research
{topic}

{context_section}
"""
```

#### Grok (xAI) Research Prompt (`prompts/grok_prompt.py`)

```python
GROK_RESEARCH_PROMPT = """You are Grok, xAI's research assistant with unique access to X/Twitter data and real-time information.

{BASE_RESEARCH_PROMPT}

## Grok-Specific Instructions
1. **Real-Time Insights**: Provide the most current perspective on the topic.
2. **Social Sentiment**: Where relevant, include public sentiment from X/Twitter discussions.
3. **Contrarian Views**: Don't shy away from unconventional or contrarian perspectives if well-supported.
4. **Direct Communication**: Be direct and clear, avoiding unnecessary hedging.

## Your Unique Value
- Access to real-time X/Twitter discussions and trends
- Contrarian and independent analysis
- Current events and breaking developments
- Public sentiment and discourse analysis

## Important Limitations
- You do NOT have web search in this context
- Focus on synthesis and analysis of information
- Be clear about what you know vs. what requires external verification

## Analysis Focus
Since you cannot search the web, focus on:
1. Analyzing the topic from first principles
2. Providing unique perspectives
3. Identifying key questions that need answering
4. Synthesizing known information creatively

## Topic to Research
{topic}

{context_section}
"""
```

#### Perplexity Research Prompt (`prompts/perplexity_prompt.py`)

```python
PERPLEXITY_RESEARCH_PROMPT = """You are Perplexity's Sonar research agent, specialized in fast, accurate web search and synthesis.

{BASE_RESEARCH_PROMPT}

## Perplexity-Specific Instructions
1. **Speed + Accuracy**: You're optimized for fast, accurate search results.
2. **Citation Density**: Every paragraph should have at least one citation.
3. **Source Diversity**: Include multiple source types (news, academic, official, expert blogs).
4. **Query Decomposition**: Break complex topics into sub-queries for thorough coverage.

## Your Unique Value
- Fast, comprehensive web search
- High citation density
- Real-time information access
- Strong at fact-checking claims

## Search Strategy
For the given topic, consider searching:
1. "{topic}" - main query
2. "{topic} latest research 2025"
3. "{topic} statistics data"
4. "{topic} expert analysis"
5. "{topic} challenges limitations"

## Topic to Research
{topic}

{context_section}
"""
```

---

### Synthesizer Agent Prompt (Final Compilation)

The synthesizer agent reviews all research results, verifies claims, and produces the final report.

#### Synthesizer System Prompt (`prompts/synthesizer_prompt.py`)

```python
SYNTHESIZER_SYSTEM_PROMPT = """You are the Chief Research Synthesizer - an expert at combining multiple research sources into a coherent, verified, and comprehensive report.

## Your Role
You receive research from multiple AI agents (OpenAI, Google, Grok, Perplexity) and must:
1. **Merge** findings into a unified report
2. **Verify** claims by cross-referencing sources
3. **Identify** consensus vs. contradictions
4. **Rank** claim confidence based on source agreement
5. **Produce** a final, publication-ready document

## Claim Verification Process

### Step 1: Extract Claims
For each source, extract distinct factual claims:
- "According to [OpenAI]: Claim X (Source: URL)"
- "According to [Google]: Claim Y (Source: URL)"

### Step 2: Cross-Reference Claims
For each major claim, check if it's:
- ✅ **VERIFIED**: 3+ sources agree (HIGH confidence)
- ⚠️ **LIKELY**: 2 sources agree (MEDIUM confidence)
- ❓ **UNVERIFIED**: Only 1 source (LOW confidence - needs verification)
- ❌ **CONTRADICTED**: Sources disagree (flag for user attention)

### Step 3: Grounding Check
Use web search to verify:
- Claims that appear in only one source
- Statistical claims (numbers, percentages)
- Recent claims (events in last 6 months)
- Controversial or surprising claims

## Output Structure

### 1. Executive Summary
[3-5 sentences summarizing the most important, verified findings]

### 2. Key Findings (Verified)
| Finding | Confidence | Sources | Notes |
|---------|------------|---------|-------|
| Claim 1 | HIGH ✅ | OpenAI, Google, Perplexity | All sources agree |
| Claim 2 | MEDIUM ⚠️ | Google, Perplexity | Not mentioned by others |
| Claim 3 | CONTRADICTED ❌ | OpenAI vs Grok | Requires resolution |

### 3. Detailed Analysis
[Synthesized narrative combining all sources, with inline confidence indicators]

### 4. Source Agreement Matrix
| Topic | OpenAI | Google | Grok | Perplexity | Consensus |
|-------|--------|--------|------|------------|-----------|
| Topic A | ✓ | ✓ | ✓ | ✓ | STRONG |
| Topic B | ✓ | ✓ | - | ✓ | GOOD |
| Topic C | ✓ | ✗ | - | ✓ | DISPUTED |

### 5. Contradictions & Disputes
[Detailed analysis of where sources disagree, with context for each position]

### 6. Gaps & Limitations
[What wasn't covered, what needs more research, known limitations]

### 7. Confidence Assessment
Overall Research Confidence: [HIGH/MEDIUM/LOW]
- Sources used: X/4
- Claim verification rate: X%
- Major contradictions: X

### 8. Complete Bibliography
[Deduplicated, organized by source type]

## Verification Guidelines

### HIGH Confidence Indicators
- Multiple independent sources cite the same information
- Primary sources (official reports, peer-reviewed papers)
- Recent, dated information
- Quantitative data with methodology

### LOW Confidence Indicators
- Single source only
- Undated information
- Secondary/tertiary sources
- Vague or unquantified claims
- Sources with known biases

## Handling Contradictions

When sources disagree:
1. Present both positions fairly
2. Note which sources support each position
3. Check for recency (newer often more accurate)
4. Check source authority (academic > blog)
5. Flag for user if unresolvable
"""

SYNTHESIZER_GROUNDING_PROMPT = """## Grounding Verification Task

I need to verify the following claims extracted from research sources.

### Claims to Verify
{claims_to_verify}

### Instructions
For each claim:
1. Search for independent verification
2. Check if the claim is accurate, partially accurate, or inaccurate
3. Note any updates or corrections
4. Provide confidence level

### Output Format
For each claim:
```
Claim: [original claim]
Verification: [VERIFIED / PARTIALLY VERIFIED / UNVERIFIED / CONTRADICTED]
Evidence: [what you found]
Sources: [URLs]
Confidence: [HIGH / MEDIUM / LOW]
Notes: [any caveats or context]
```
"""
```

---

### Claim Extraction & Verification (`synthesizer/claim_verifier.py`)

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class ClaimConfidence(str, Enum):
    HIGH = "high"           # 3+ sources agree
    MEDIUM = "medium"       # 2 sources agree
    LOW = "low"             # 1 source only
    CONTRADICTED = "contradicted"  # Sources disagree

@dataclass
class ExtractedClaim:
    claim: str
    source: str              # "openai", "google", "grok", "perplexity"
    citation_url: Optional[str]
    category: str            # "statistic", "fact", "opinion", "prediction"

@dataclass
class VerifiedClaim:
    claim: str
    confidence: ClaimConfidence
    supporting_sources: List[str]
    contradicting_sources: List[str]
    grounding_result: Optional[str]  # Result from verification search
    final_assessment: str

class ClaimVerifier:
    """Extracts and verifies claims across multiple research sources."""

    def __init__(self, grounding_client):
        self.grounding_client = grounding_client  # Perplexity or search API

    async def extract_claims(self, research_content: str, source: str) -> List[ExtractedClaim]:
        """Extract distinct factual claims from research content."""
        # Use LLM to extract claims
        extraction_prompt = f"""
        Extract all distinct factual claims from this research.

        For each claim, identify:
        1. The claim itself (one sentence)
        2. The category: statistic, fact, opinion, or prediction
        3. The source URL if cited

        Research from {source}:
        {research_content}

        Output as JSON array:
        [
            {{"claim": "...", "category": "...", "citation_url": "..."}}
        ]
        """
        # Implementation would call LLM here
        pass

    async def cross_reference_claims(
        self,
        all_claims: dict[str, List[ExtractedClaim]]
    ) -> List[VerifiedClaim]:
        """Cross-reference claims across sources to determine confidence."""

        # Group similar claims using semantic similarity
        claim_groups = self._group_similar_claims(all_claims)

        verified_claims = []
        for group in claim_groups:
            sources = set(c.source for c in group)

            if len(sources) >= 3:
                confidence = ClaimConfidence.HIGH
            elif len(sources) == 2:
                confidence = ClaimConfidence.MEDIUM
            else:
                confidence = ClaimConfidence.LOW

            verified_claims.append(VerifiedClaim(
                claim=group[0].claim,  # Representative claim
                confidence=confidence,
                supporting_sources=list(sources),
                contradicting_sources=[],
                grounding_result=None,
                final_assessment=""
            ))

        return verified_claims

    async def ground_uncertain_claims(
        self,
        claims: List[VerifiedClaim]
    ) -> List[VerifiedClaim]:
        """Use search to verify low-confidence or contradicted claims."""

        for claim in claims:
            if claim.confidence in [ClaimConfidence.LOW, ClaimConfidence.CONTRADICTED]:
                # Search for verification
                search_result = await self.grounding_client.search(
                    f"verify: {claim.claim}"
                )
                claim.grounding_result = search_result
                claim.final_assessment = self._assess_grounding(claim, search_result)

        return claims
```

---

### Updated Architecture with Verification

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
       ┌─────────────────────────────────────┼─────────────────────────────────────┐
       │                                     │                                     │
       ▼                                     ▼                                     ▼
┌──────────────┐                     ┌──────────────┐                     ┌──────────────┐
│    OpenAI    │                     │    Google    │                     │  Perplexity  │
│   + Prompt   │                     │   + Prompt   │                     │   + Prompt   │
└──────┬───────┘                     └──────┬───────┘                     └──────┬───────┘
       │                                    │                                    │
       └────────────────────────────────────┼────────────────────────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │      Claim Extractor        │
                              │  (Extract factual claims)   │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │    Cross-Reference Engine   │
                              │ (Compare claims across src) │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │    Grounding Verifier       │
                              │ (Search to verify claims)   │
                              │    [Uses Perplexity API]    │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │   Synthesizer Agent         │
                              │ (Merge + confidence scores) │
                              │    + System Prompt          │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │    Final Report Generator   │
                              │ (MD / HTML / PDF + sources) │
                              └─────────────────────────────┘
```

---

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
│   ├── prompts/                # NEW: System prompts
│   │   ├── research_base.py    # Base template
│   │   ├── openai_prompt.py
│   │   ├── google_prompt.py
│   │   ├── grok_prompt.py
│   │   ├── perplexity_prompt.py
│   │   └── synthesizer_prompt.py
│   ├── synthesizer/            # NEW: Claim verification
│   │   ├── merger.py
│   │   ├── claim_extractor.py
│   │   ├── claim_verifier.py
│   │   └── grounding.py        # Search-based verification
│   └── utils/
│       ├── rate_limiter.py
│       ├── retry.py
│       ├── cost_tracker.py     # NEW: Track API costs
│       ├── cache.py            # NEW: Result caching
│       └── progress.py         # NEW: Rich progress display
├── tests/
│   ├── mocks/                  # NEW: Mock API responses
│   │   ├── openai_mock.json
│   │   ├── google_mock.json
│   │   ├── grok_mock.json
│   │   └── perplexity_mock.json
│   └── ...
├── Dockerfile                  # NEW: For WeasyPrint deps
└── ...
```

---

### ✅ Updated Implementation Phases

#### Phase 1: Project Setup
- [ ] Create `pyproject.toml` with corrected dependencies
- [ ] Set up `config.py` with **cost limits** and **mock mode**
- [ ] Create `.env.example` with all API keys (including GOOGLE_API_KEY)
- [ ] Add `Dockerfile` for containerized usage
- [ ] Initialize directory structure

#### Phase 2: System Prompts (NEW)
- [ ] Create `prompts/research_base.py` with base template
- [ ] Create `prompts/openai_prompt.py` with OpenAI-specific instructions
- [ ] Create `prompts/google_prompt.py` with Google-specific instructions
- [ ] Create `prompts/grok_prompt.py` with Grok-specific instructions
- [ ] Create `prompts/perplexity_prompt.py` with Perplexity-specific instructions
- [ ] Create `prompts/synthesizer_prompt.py` with verification instructions

#### Phase 3: Core Utilities
- [ ] Implement `cost_tracker.py` with per-provider costs
- [ ] Implement `cache.py` with disk-based caching
- [ ] Implement `progress.py` with rich progress bars
- [ ] Add mock response fixtures for testing (all 4 providers)

#### Phase 4: Agent Implementation
- [ ] Implement `BaseResearchAgent` with cost tracking hooks
- [ ] Implement `OpenAIResearchAgent` using **Responses API** (not chat)
- [ ] Implement `GoogleResearchAgent` using **Interactions API** with polling
- [ ] Implement `GrokResearchAgent` via **httpx** (synthesis only, no web search)
- [ ] Implement `PerplexityResearchAgent` via **OpenAI client** with custom base_url
- [ ] Add `rate_limiter.py` and `retry.py` utilities

#### Phase 5: Claim Verification System (NEW)
- [ ] Implement `claim_extractor.py` to extract claims from research
- [ ] Implement `claim_verifier.py` to cross-reference claims across sources
- [ ] Implement `grounding.py` for search-based verification of uncertain claims
- [ ] Implement semantic similarity grouping for claim comparison
- [ ] Add confidence scoring logic

#### Phase 6: Synthesizer & LangGraph Workflow
- [ ] Implement `merger.py` with full synthesis logic
- [ ] Build LangGraph workflow with parallel research nodes
- [ ] Add claim extraction node after research
- [ ] Add cross-reference node
- [ ] Add grounding verification node
- [ ] Add final synthesis node

#### Phase 7: Output Generation
- [ ] Implement `MarkdownGenerator` with confidence indicators
- [ ] Create Jinja2 HTML templates with verification badges
- [ ] Implement `PDFGenerator` with WeasyPrint
- [ ] Add source agreement matrix visualization

#### Phase 8: CLI and Polish
- [ ] Build CLI with argparse
- [ ] Add progress indicators
- [ ] Write README documentation
- [ ] Create example scripts

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
