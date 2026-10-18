import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="LangChain & AI Systems Engineering Mastery",
    layout="wide",
    page_icon="🦜"
)

# AI Engineering Dark Theme Colors & Constants
AI_BLUE = "#3B82F6"
AI_BLUE_HOVER = "#2563EB"
AI_PURPLE = "#8B5CF6"
AI_CYAN = "#06B6D4"
AI_EMERALD = "#10B981"
AI_AMBER = "#F59E0B"
AI_ROSE = "#F43F5E"
BG_DARK = "#0B0F19"
BG_CARD = "#161F30"
BORDER_DARK = "#2D3A4F"
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "ai_readiness_checklist" not in st.session_state:
    st.session_state.ai_readiness_checklist = {
        "LCEL Runnable Protocol (invoke, batch, stream, astream)": True,
        "Hybrid RAG (Dense Embeddings + BM25 Sparse + RRF)": True,
        "Cross-Encoder Reranking & Context Compression": True,
        "Vector DB Index Sizing (HNSW M/efConstruction & PQ)": False,
        "Semantic Caching with Vector Distance Thresholds": False,
        "LangGraph StateGraph & Cyclic Multi-Agent Loops": True,
        "Tool Calling with Strict Pydantic Schema Guardrails": False,
        "Prompt Caching Prefix Optimization (Gemini / Anthropic)": False,
        "Ragas Automated Evaluation (Faithfulness & Precision)": False,
        "LangSmith Tracing, Feedback Collector & Latency Budgets": True,
        "Multi-Tenant Vector DB Namespace Isolation": False,
        "vLLM PagedAttention & Continuous Batching Deployment": False
    }

if "active_mermaid_step" not in st.session_state:
    st.session_state.active_mermaid_step = "Step 4: Vector Retrieval"

# ---------------------------------------------------------
# MODULE 0: OVERVIEW & ARCHITECTURE DASHBOARD
# ---------------------------------------------------------
def render_dashboard():
    st.title("🦜 LangChain & Production AI Systems Blueprint")
    st.caption("Interactive Technical Guide for AI Engineers building Enterprise RAG, Vector DBs & Agentic Workflows at Scale")
    
    # Header KPI stats
    total_checks = len(st.session_state.ai_readiness_checklist)
    done_checks = sum(1 for v in st.session_state.ai_readiness_checklist.values() if v)
    readiness_pct = int((done_checks / total_checks) * 100) if total_checks else 0
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="card">
            <div class="label">RAG Pipeline Latency Target</div>
            <div style="font-size: 22px; font-weight: 700; color: {AI_BLUE}; margin: 4px 0;">&lt; 650 ms</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">P95 end-to-end TTFT including hybrid reranking</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="card">
            <div class="label">Architectural Readiness</div>
            <div style="font-size: 22px; font-weight: 700; color: {AI_EMERALD}; margin: 4px 0;">{readiness_pct}%</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">{done_checks}/{total_checks} core competencies validated</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="card">
            <div class="label">Semantic Cache Savings</div>
            <div style="font-size: 22px; font-weight: 700; color: {AI_PURPLE}; margin: 4px 0;">Up to 68%</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">Token cost & latency reduction at high QPS</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="card">
            <div class="label">Primary Stack</div>
            <div style="font-size: 22px; font-weight: 700; color: {AI_CYAN}; margin: 4px 0;">LangChain / LCEL</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">LangGraph, Vector DBs, Ragas, Flash Models</div>
        </div>
        """, unsafe_allow_html=True)

    c_left, c_right = st.columns([6, 6])
    
    with c_left:
        st.markdown("### 🏛️ Production AI Architecture Stack")
        layers = [
            ("Client Applications & APIs", "Web Client, Chatbot UI, REST/gRPC API, Async Event Consumers", AI_BLUE),
            ("Orchestration & State (LangChain / LangGraph)", "LCEL RunnableSequence, Multi-Agent StateGraph, Checkpoints", AI_PURPLE),
            ("Retrieval & Context Compression", "HyDE Query Expansion, Dense + BM25 Hybrid, Cross-Encoder Rerank", AI_CYAN),
            ("Vector Storage & Knowledge Stores", "Pinecone, Qdrant, Chroma, FAISS, pgvector with HNSW Indices", AI_EMERALD),
            ("LLM Serving & Token Infrastructure", "Gemini 2.5 Flash, Claude 3.5, GPT-4o, vLLM PagedAttention", AI_AMBER),
            ("Observability & Evaluation (LLMOps)", "LangSmith Traces, Ragas Metrics (Faithfulness, Recall), Guardrails", AI_ROSE)
        ]
        
        for name, desc, color in layers:
            st.markdown(f"""
            <div style="background: {BG_CARD}; border-left: 4px solid {color}; border-top: 1px solid {BORDER_DARK}; border-right: 1px solid {BORDER_DARK}; border-bottom: 1px solid {BORDER_DARK}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="font-weight: 700; font-size: 12.5px; color: {color};">{name}</div>
                <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with c_right:
        st.markdown("### 🧭 AI Systems Competency Radar")
        categories = ['LCEL / Runnables', 'RAG Retrieval', 'Vector Stores', 'Agentic Graphs', 'Cost Engineering', 'LLM Evaluation']
        current_ratings = [4, 4, 3, 4, 3, 4]
        target_ratings = [5, 5, 4.5, 4.5, 4.5, 4.5]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=current_ratings + [current_ratings[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Current Proficiency',
            line=dict(color=AI_BLUE, width=2),
            fillcolor='rgba(59, 130, 246, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=target_ratings + [target_ratings[0]],
            theta=categories + [categories[0]],
            name='Production Target (Staff AI Architect)',
            line=dict(color=AI_EMERALD, dash='dash', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5], color=TEXT_MUTED, gridcolor=BORDER_DARK),
                angularaxis=dict(color=TEXT_LIGHT, gridcolor=BORDER_DARK)
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=10, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_LIGHT, size=11),
            height=300
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🚀 Interactive Blueprint Modules")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        **RAG & Retrieval Architecture**
        - **Module 1:** Interactive Mermaid RAG Sequence Diagram
        - **Module 3:** Vector DB Scaling Configurator (Pinecone, FAISS, Chroma, Qdrant)
        - **Module 5:** Advanced Hybrid Retrieval & Reranking
        """)
    with m2:
        st.markdown("""
        **Cost & Core Frameworks**
        - **Module 2:** Slider-based Scaling Cost & Token Calculator
        - **Module 4:** LangChain LCEL & Runnable Protocol
        - **Module 6:** LangGraph Agent State Machines
        """)
    with m3:
        st.markdown("""
        **Production Readiness & Search**
        - **Module 7:** Ragas Evaluation & Production Guardrails
        - **Module 8:** AI Engineering Competency Checklist
        - **Module 9:** AI Tech Radar (Google Search Grounded)
        """)

# ---------------------------------------------------------
# MODULE 1: INTERACTIVE MERMAID RAG SEQUENCE DIAGRAM
# ---------------------------------------------------------
def render_rag_sequence():
    st.title("🔁 Production RAG Pipeline: Interactive Sequence Diagram")
    st.caption("Visualizing the end-to-end lifecycle of an enterprise Retrieval-Augmented Generation request in LangChain")
    
    st.markdown("""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="label" style="color: #60A5FA;">Foundational Architecture Reference</div>
                <div style="font-weight: 700; font-size: 14px; margin-top: 4px;">Standard RAG Pipeline with Query Decomposition, Vector Search, Cross-Encoder Rerank & Guardrails</div>
            </div>
            <div style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 4px; padding: 4px 10px; font-size: 11px; color: #93C5FD; font-family: monospace;">
                P95 SLA: ~550ms
            </div>
        </div>
        <p style="font-size: 12px; color: #94A3B8; margin-top: 8px; line-height: 1.5;">
            Click on any participant node or sequence phase below to inspect its latency budget, LangChain LCEL implementation, failure modes, and production scaling trade-offs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sequence steps details dictionary
    steps_data = {
        "1. Query Rewriter & HyDE": {
            "actor": "QueryRewriter",
            "latency": "25 - 45 ms",
            "role": "Expands ambiguous user queries into multiple search angles and generates hypothetical document embeddings (HyDE) for superior semantic capture.",
            "code": """from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Query expansion & HyDE prompt in LCEL
rewrite_prompt = ChatPromptTemplate.from_template(
    "Given the user question: {question}\\n"
    "Generate 3 diverse sub-queries and a brief hypothetical passage "
    "that would directly answer this question to maximize vector recall."
)

query_rewriter = rewrite_prompt | fast_llm | StrOutputParser()""",
            "failure_modes": "Over-generation causing semantic drift; adding unnecessary latency if simple keyword queries don't need decomposition.",
            "mitigation": "Add a lightweight classifier / regex filter to bypass rewriting on trivial lookups."
        },
        "2. Embedding Model": {
            "actor": "Embedder",
            "latency": "15 - 30 ms",
            "role": "Translates the expanded text query into a dense mathematical vector (e.g. 768 or 1536 dimensions) using normalized floating-point embeddings.",
            "code": """from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Or OpenAIEmbeddings(model="text-embedding-3-small")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_query"
)

# Convert query to dense vector
query_vector = embeddings.embed_query("How to scale multi-tenant vector databases?")""",
            "failure_modes": "Mismatch between query embedding model and index embedding model; embedding model token truncation without warning.",
            "mitigation": "Enforce model version tagging in vector metadata and assert token lengths before vectorization."
        },
        "3. Vector DB (ANN Retrieval)": {
            "actor": "VectorDB",
            "latency": "15 - 35 ms",
            "role": "Performs Approximate Nearest Neighbor (ANN) search over millions of indexed chunks using HNSW or IVF indices with metadata namespace pre-filtering.",
            "code": """from langchain_pinecone import PineconeVectorStore
# Or Qdrant, Chroma, FAISS

vectorstore = PineconeVectorStore(index_name="knowledge-base", embedding=embeddings)

# Retrieve top 20 candidate chunks with tenant filter
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 20,
        "filter": {"tenant_id": {"$eq": "org_42"}, "status": "active"}
    }
)""",
            "failure_modes": "Post-filtering discarding 90% of candidates leaving too few chunks; high dimensionality curse degrading recall.",
            "mitigation": "Use single-stage graph filtering (Qdrant/Pinecone) rather than post-search metadata filtering."
        },
        "4. Cross-Encoder Reranker": {
            "actor": "Reranker",
            "latency": "35 - 60 ms",
            "role": "Takes the top 20 candidate documents and computes deep cross-attention with the query using a cross-encoder model (e.g. Cohere Rerank or BGE-Reranker), selecting top-4 high-precision chunks.",
            "code": """from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import CohereRerank

compressor = CohereRerank(model="rerank-english-v3.0", top_n=4)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Returns top 4 precisely scored documents
high_relevance_docs = compression_retriever.invoke(user_query)""",
            "failure_modes": "Sending too many documents (e.g. k=100) causes reranker throughput bottlenecks and spikes latency.",
            "mitigation": "Cap initial retrieval to k=20 to keep reranking compute under 50ms."
        },
        "5. Prompt Builder (LCEL Format)": {
            "actor": "PromptBuilder",
            "latency": "1 - 3 ms",
            "role": "Injects formatted context chunks, system instructions, and chat history into a ChatPromptTemplate with strict token boundary enforcement.",
            "code": """from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\\n\\n".join([f"--- Source: {d.metadata.get('source')} ---\\n{d.page_content}" for d in docs])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise AI assistant. Answer strictly using the provided context chunks. "
               "Cite chunk sources. If the answer is not in the context, explicitly state you do not know.\\n\\n"
               "CONTEXT:\\n{context}"),
    ("human", "{question}")
])""",
            "failure_modes": "Lost-in-the-middle phenomenon: placing critical documents in the center of long contexts causes LLM attention degradation.",
            "mitigation": "Sort retrieved documents so highest-relevance chunks appear at the very start and very end of the prompt context."
        },
        "6. LLM Inference & Streaming": {
            "actor": "LLM",
            "latency": "250 - 450 ms TTFT",
            "role": "Generates the response tokens conditioned on the retrieved context chunks using speculative decoding and low-temperature parameters.",
            "code": """from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,  # Low temperature for factual RAG
    max_output_tokens=1024,
    streaming=True
)

# Stream chunks to client in real-time
# for chunk in chain.stream({"question": query}):
#     print(chunk, end="", flush=True)""",
            "failure_modes": "Hallucinating extraneous facts beyond retrieved evidence; API timeout during high concurrency.",
            "mitigation": "Implement primary model with automated secondary fallback via `llm.with_fallbacks([backup_llm])`."
        },
        "7. Output Parser & Guardrails": {
            "actor": "OutputParser",
            "latency": "5 - 15 ms",
            "role": "Parses output into typed Pydantic models, verifies citation consistency, and applies hallucination / PII filters before final delivery.",
            "code": """from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import List

class RAGResponse(BaseModel):
    answer: str = Field(description="Direct synthesis based on context")
    citations: List[str] = Field(description="List of document chunk IDs cited")
    confidence_score: float = Field(description="Estimated groundedness score from 0 to 1")

parser = PydanticOutputParser(pydantic_object=RAGResponse)""",
            "failure_modes": "JSON schema parsing exceptions when LLM outputs Markdown code blocks with preamble.",
            "mitigation": "Use `.with_structured_output(RAGResponse)` which directly binds native function calling schemas."
        }
    }

    # Interactive node selector
    col_sel, col_stat = st.columns([8, 4])
    with col_sel:
        step_keys = list(steps_data.keys())
        selected_step = st.selectbox(
            "📍 Highlight Sequence Node & View Technical Deep Dive:",
            step_keys,
            index=2  # default Vector DB
        )
    with col_stat:
        st.markdown(f"""
        <div style="background: {BG_CARD}; border: 1px solid {BORDER_DARK}; border-radius: 6px; padding: 10px 14px; margin-top: 6px;">
            <div style="font-size: 10px; color: {TEXT_MUTED}; text-transform: uppercase; font-weight: 700;">Selected Component SLA</div>
            <div style="font-size: 16px; font-weight: 700; color: {AI_EMERALD};">{steps_data[selected_step]['latency']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Embedded Mermaid.js Sequence Diagram with professional dark theme
    mermaid_code = f"""%%{{init: {{
  'theme': 'base',
  'themeVariables': {{
    'primaryColor': '#1E293B',
    'primaryTextColor': '#F8FAFC',
    'primaryBorderColor': '#3B82F6',
    'lineColor': '#60A5FA',
    'secondaryColor': '#0F172A',
    'tertiaryColor': '#161F30',
    'noteTextColor': '#E2E8F0',
    'noteBkgColor': '#1E293B',
    'noteBorderColor': '#334155',
    'actorBkg': '#161F30',
    'actorBorder': '#3B82F6',
    'actorTextColor': '#FFFFFF',
    'actorLineColor': '#334155',
    'signalColor': '#38BDF8',
    'signalTextColor': '#E2E8F0'
  }}
}}}}%%
sequenceDiagram
    autonumber
    actor User as 👤 Client / App
    participant Orch as 🦜 LCEL Orchestrator
    participant Rewriter as 🔍 Query Rewriter (HyDE)
    participant Embed as 🧮 Embedder (Dense)
    participant VDB as 🗄️ Vector DB (Index)
    participant Rerank as ⚡ Cross-Encoder Reranker
    participant Prompt as 📝 ChatPromptTemplate
    participant LLM as 🤖 LLM (Gemini 2.5)
    participant Guard as 🛡️ Output Guardrail

    User->>Orch: 1. invoke(question = "Scale Vector DB?")
    activate Orch
    
    rect rgb(22, 31, 48)
        note right of Orch: Query Expansion & Dense Vectorization
        Orch->>Rewriter: 2. rewrite_and_expand(question)
        Rewriter-->>Orch: 3. Sub-queries & hypothetical docs
        Orch->>Embed: 4. embed_query(expanded_text)
        Embed-->>Orch: 5. 768-dim query vector
    end

    rect rgb(15, 23, 42)
        note right of Orch: Multi-Stage Hybrid Candidate Retrieval
        Orch->>VDB: 6. similarity_search(vector, k=20, tenant_id)
        VDB-->>Orch: 7. Top 20 raw candidate chunks
        Orch->>Rerank: 8. rerank(candidates, query, top_n=4)
        Rerank-->>Orch: 9. Top 4 high-precision chunks
    end

    rect rgb(22, 31, 48)
        note right of Orch: Prompt Formulation & Generation
        Orch->>Prompt: 10. format(context=top_docs, query)
        Prompt-->>Orch: 11. ChatPromptValue with strict system prompt
        Orch->>LLM: 12. stream(messages, temp=0.1)
        LLM-->>Orch: 13. Candidate token stream
    end

    rect rgb(15, 23, 42)
        note right of Orch: Structured Validation & Citation Check
        Orch->>Guard: 14. validate_citations_and_schema(response)
        Guard-->>Orch: 15. Verified Pydantic output
    end

    Orch-->>User: 16. Synthesized Answer with Cited Document Chunks
    deactivate Orch
"""

    # Interactive HTML component containing Mermaid diagram with client-side click highlighting
    html_mermaid = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
        <style>
            body {{
                background-color: transparent;
                margin: 0;
                padding: 10px;
                font-family: 'Inter', system-ui, sans-serif;
                color: #F8FAFC;
                overflow-x: auto;
            }}
            #mermaid-container {{
                display: flex;
                justify-content: center;
                background: #0B0F19;
                border: 1px solid #2D3A4F;
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            }}
            .mermaid {{
                width: 100%;
                text-align: center;
            }}
            /* Highlighting active actor */
            .actor {{
                cursor: pointer;
                transition: transform 0.2s, filter 0.2s;
            }}
            .actor:hover {{
                filter: drop-shadow(0 0 8px #3B82F6);
            }}
            .messageLine0, .messageLine1 {{
                stroke-width: 1.5px !important;
            }}
            .highlight-banner {{
                background: rgba(59, 130, 246, 0.12);
                border: 1px solid #3B82F6;
                border-radius: 6px;
                padding: 8px 12px;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-size: 11.5px;
            }}
        </style>
    </head>
    <body>
        <div class="highlight-banner">
            <div>
                <span style="color: #60A5FA; font-weight: 700;">⚡ Active Focus:</span>
                <span style="color: #FFFFFF; font-weight: 600;">{selected_step}</span>
            </div>
            <div style="color: #94A3B8; font-family: monospace;">
                Actor: {steps_data[selected_step]['actor']}
            </div>
        </div>
        <div id="mermaid-container">
            <div class="mermaid">
{mermaid_code}
            </div>
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                securityLevel: 'loose',
                theme: 'dark'
            }});
        </script>
    </body>
    </html>
    """

    st.components.v1.html(html_mermaid, height=520, scrolling=True)

    # Detailed Step Inspection Card
    step_info = steps_data[selected_step]
    st.markdown(f"### 🔍 Detailed Inspection: {selected_step}")
    
    col_d1, col_d2 = st.columns([6, 6])
    with col_d1:
        st.markdown(f"""
        <div class="card">
            <div class="label" style="color: {AI_CYAN};">Core Functional Responsibility</div>
            <p style="margin-top: 6px; font-size: 13px; color: #E2E8F0; line-height: 1.5;">{step_info['role']}</p>
            <div style="margin-top: 12px;">
                <span class="label" style="color: {AI_ROSE};">⚠️ Production Failure Mode:</span>
                <div style="font-size: 11.5px; color: #FCA5A5; margin-top: 3px;">{step_info['failure_modes']}</div>
            </div>
            <div style="margin-top: 10px;">
                <span class="label" style="color: {AI_EMERALD};">🛡️ Architectural Mitigation:</span>
                <div style="font-size: 11.5px; color: #6EE7B7; margin-top: 3px;">{step_info['mitigation']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown(f"**LangChain / LCEL Implementation Code:**")
        st.code(step_info['code'], language="python")

# ---------------------------------------------------------
# MODULE 2: SLIDER-BASED SCALING COST & TOKEN CALCULATOR
# ---------------------------------------------------------
def render_cost_calculator():
    st.title("💰 Scaling Cost & Token Consumption Calculator")
    st.caption("Estimate token volume, monthly infrastructure spend, and cache savings when scaling LangChain applications")

    st.markdown("""
    <div class="card">
        <div class="label" style="color: #F59E0B;">Production LLM Economics</div>
        <p style="font-size: 12px; color: #CBD5E1; margin-top: 4px; line-height: 1.5;">
            In production RAG systems, retrieved context chunks (e.g. 5 chunks × 350 tokens = 1,750 tokens) dominate prompt input tokens. 
            Adjust query volumes and token parameters below to simulate exact monthly operational costs across top foundation models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Input Sliders
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("#### 📈 Query Volume & Traffic")
        daily_queries = st.slider(
            "Daily Inbound Queries",
            min_value=1000,
            max_value=2000000,
            value=100000,
            step=5000,
            format="%d queries"
        )
        monthly_queries = daily_queries * 30.5
        st.caption(f"Monthly Queries: **{monthly_queries:,.0f}** (~{daily_queries/86400:.1f} QPS average)")

        cache_hit_rate = st.slider(
            "Semantic / Prompt Cache Hit Rate (%)",
            min_value=0,
            max_value=90,
            value=35,
            step=5,
            help="Percentage of incoming queries answered via similarity cache without hitting the foundation LLM"
        )

    with col_s2:
        st.markdown("#### 📝 Token Sizes per Query")
        user_prompt_tokens = st.slider(
            "Base Question + System Prompt Tokens",
            min_value=50,
            max_value=1500,
            value=350,
            step=25
        )
        
        retrieved_chunks = st.slider(
            "Top-K Retrieved Context Chunks",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        
        tokens_per_chunk = st.slider(
            "Average Tokens per Chunk",
            min_value=100,
            max_value=1000,
            value=350,
            step=50
        )
        
        total_context_tokens = retrieved_chunks * tokens_per_chunk
        total_input_tokens = user_prompt_tokens + total_context_tokens
        st.caption(f"Total Input per Query: **{total_input_tokens:,} tokens** ({total_context_tokens:,} from context)")

    with col_s3:
        st.markdown("#### 📤 Output Tokens & Model")
        output_tokens = st.slider(
            "Average Completion / Output Tokens",
            min_value=50,
            max_value=2500,
            value=400,
            step=50
        )

        model_catalog = {
            "Gemini 2.5 Flash (Recommended for RAG)": {
                "input_per_m": 0.075,
                "output_per_m": 0.30,
                "cached_input_per_m": 0.01875,
                "provider": "Google",
                "context_window": "1M tokens"
            },
            "Gemini 1.5 Pro (Complex Reasoning)": {
                "input_per_m": 1.25,
                "output_per_m": 5.00,
                "cached_input_per_m": 0.3125,
                "provider": "Google",
                "context_window": "2M tokens"
            },
            "GPT-4o (Flagship Multimodal)": {
                "input_per_m": 2.50,
                "output_per_m": 10.00,
                "cached_input_per_m": 1.25,
                "provider": "OpenAI",
                "context_window": "128k tokens"
            },
            "GPT-4o-mini (Lightweight)": {
                "input_per_m": 0.15,
                "output_per_m": 0.60,
                "cached_input_per_m": 0.075,
                "provider": "OpenAI",
                "context_window": "128k tokens"
            },
            "Claude 3.5 Sonnet": {
                "input_per_m": 3.00,
                "output_per_m": 15.00,
                "cached_input_per_m": 0.75,
                "provider": "Anthropic",
                "context_window": "200k tokens"
            },
            "Llama 3 70B (Self-Hosted vLLM on 2x A100 GPUs)": {
                "input_per_m": 0.35, # Blended compute amortization
                "output_per_m": 0.35,
                "cached_input_per_m": 0.05,
                "provider": "Self-Hosted",
                "context_window": "128k tokens"
            }
        }

        selected_model_name = st.selectbox("Foundation Model", list(model_catalog.keys()))
        selected_model = model_catalog[selected_model_name]

    # Math calculations
    effective_queries = monthly_queries * (1 - (cache_hit_rate / 100))
    cached_queries = monthly_queries * (cache_hit_rate / 100)

    # Monthly Token totals
    uncached_input_tokens_m = (effective_queries * total_input_tokens) / 1_000_000
    uncached_output_tokens_m = (effective_queries * output_tokens) / 1_000_000
    total_tokens_billions = (monthly_queries * (total_input_tokens + output_tokens)) / 1_000_000_000

    # Pricing calculation
    cost_input = uncached_input_tokens_m * selected_model["input_per_m"]
    cost_output = uncached_output_tokens_m * selected_model["output_per_m"]
    total_monthly_cost = cost_input + cost_output

    # Without cache cost
    full_input_tokens_m = (monthly_queries * total_input_tokens) / 1_000_000
    full_output_tokens_m = (monthly_queries * output_tokens) / 1_000_000
    cost_without_cache = (full_input_tokens_m * selected_model["input_per_m"]) + (full_output_tokens_m * selected_model["output_per_m"])
    monthly_savings = max(0, cost_without_cache - total_monthly_cost)
    cost_per_query = total_monthly_cost / monthly_queries if monthly_queries else 0

    st.markdown("---")
    st.markdown("### 📊 Monthly Sizing & Financial Summary")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="card">
            <div class="label">Monthly Total Spend</div>
            <div style="font-size: 24px; font-weight: 700; color: {AI_BLUE}; margin: 4px 0;">${total_monthly_cost:,.2f}</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">Annualized: ${total_monthly_cost * 12:,.2f}/yr</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="card">
            <div class="label">Unit Economics (CPM)</div>
            <div style="font-size: 24px; font-weight: 700; color: {AI_EMERALD}; margin: 4px 0;">${cost_per_query * 1000:,.3f}</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">${cost_per_query:,.5f} per query</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="card">
            <div class="label">Monthly Cache Savings</div>
            <div style="font-size: 24px; font-weight: 700; color: {AI_AMBER}; margin: 4px 0;">${monthly_savings:,.2f}</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">{cache_hit_rate}% queries intercepted in &lt;15ms</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="card">
            <div class="label">Token Volume Processed</div>
            <div style="font-size: 24px; font-weight: 700; color: {AI_PURPLE}; margin: 4px 0;">{total_tokens_billions:,.2f}B</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">{uncached_input_tokens_m:,.1f}M Input + {uncached_output_tokens_m:,.1f}M Output</div>
        </div>
        """, unsafe_allow_html=True)

    # Visualization: Scaling projection curves across models
    col_chart1, col_chart2 = st.columns([6, 6])
    
    with col_chart1:
        st.markdown("#### Cost Scaling Curve Across Query Volumes")
        sample_volumes = [10_000, 50_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
        
        df_curves = []
        for v in sample_volumes:
            eff_v = v * (1 - (cache_hit_rate / 100))
            in_m = (eff_v * total_input_tokens) / 1_000_000
            out_m = (eff_v * output_tokens) / 1_000_000
            for m_name, m_spec in model_catalog.items():
                short_name = m_name.split(" ")[0] + " " + m_name.split(" ")[1]
                c = (in_m * m_spec["input_per_m"]) + (out_m * m_spec["output_per_m"])
                df_curves.append({"Monthly Queries": v, "Model": short_name, "Monthly Cost ($)": c})

        df_curves_df = pd.DataFrame(df_curves)
        fig_scale = px.line(
            df_curves_df,
            x="Monthly Queries",
            y="Monthly Cost ($)",
            color="Model",
            markers=True,
            log_x=True,
            log_y=True,
            title="Monthly Cost vs Query Volume (Log-Log Scale)"
        )
        fig_scale.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_LIGHT, size=11),
            legend=dict(orientation="h", y=-0.3),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_scale, use_container_width=True)

    with col_chart2:
        st.markdown("#### Spend Breakdown & Cache Efficiency")
        labels = ["Prompt Input Spend", "Completion Output Spend", "Saved by Caching"]
        values = [cost_input, cost_output, monthly_savings]
        colors = [AI_BLUE, AI_PURPLE, AI_EMERALD]

        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_LIGHT, size=11),
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("""
    <div style="background: #162032; border: 1px solid #29384D; border-radius: 6px; padding: 12px; margin-top: 14px;">
        <div style="font-weight: 700; font-size: 12px; color: #60A5FA; margin-bottom: 4px;">💡 LangChain Production Architecture Tip:</div>
        <div style="font-size: 11.5px; color: #CBD5E1; line-height: 1.5;">
            By integrating <code>langchain.cache.RedisSemanticCache</code> with a cosine distance threshold of <code>0.10</code>, 
            repetitive enterprise queries (e.g. employee FAQ lookups or customer billing questions) are returned in under 12ms without invoking external LLMs, 
            saving thousands of dollars per month while reducing latency by 95%.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 3: VECTOR DB CONFIGURATOR & SCALING TRADE-OFFS
# ---------------------------------------------------------
def render_vector_db_configurator():
    st.title("🗄️ Vector Database Configurator & Scaling Trade-offs")
    st.caption("Benchmark memory footprints, index algorithms, latency, and LangChain integration patterns across vector engines")

    # Database Specs Catalog
    vector_dbs = {
        "Pinecone (Serverless)": {
            "type": "Managed Cloud Serverless",
            "index_algo": "Proprietary HNSW + Graph Partitioning",
            "p95_latency_ms": 22,
            "recall_pct": 98.5,
            "filtering_power": 9.5,
            "scalability": 10.0,
            "ops_simplicity": 10.0,
            "cost_score": 7.0,
            "quantization": "Automated SQ8 / FP16",
            "best_for": "Zero-maintenance production scale, multi-tenant SaaS, enterprise isolation.",
            "langchain_pkg": "langchain_pinecone",
            "snippet": """from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Ingest and query Pinecone serverless index
vectorstore = PineconeVectorStore(
    index_name="production-rag",
    embedding=embeddings,
    namespace="tenant_acme_corp"
)

# Metadata filtered retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": {"doc_category": "finance"}}
)"""
        },
        "FAISS (Facebook AI Similarity Search)": {
            "type": "In-Memory C++ / GPU Library",
            "index_algo": "HNSW / IVF-PQ / Flat",
            "p95_latency_ms": 4,
            "recall_pct": 97.0,
            "filtering_power": 4.5,
            "scalability": 6.5,
            "ops_simplicity": 6.0,
            "cost_score": 9.5,
            "quantization": "IVF-PQ (90% RAM reduction)",
            "best_for": "Local development, low-latency microservices, sidecar containers, ephemeral search.",
            "langchain_pkg": "langchain_community.vectorstores",
            "snippet": """import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Initialize exact or HNSW FAISS index
dimension = 768
index = faiss.IndexHNSWFlat(dimension, 32)
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Persist and reload locally
vectorstore.save_local("faiss_index_dir")
loaded_store = FAISS.load_local("faiss_index_dir", embeddings, allow_dangerous_deserialization=True)"""
        },
        "Chroma": {
            "type": "Embedded / Client-Server",
            "index_algo": "HNSW (hnswlib)",
            "p95_latency_ms": 18,
            "recall_pct": 98.0,
            "filtering_power": 7.5,
            "scalability": 6.0,
            "ops_simplicity": 9.5,
            "cost_score": 9.0,
            "quantization": "FP32 default",
            "best_for": "Rapid prototyping, internal tooling, desktop apps, small-to-medium datasets (<2M vectors).",
            "langchain_pkg": "langchain_chroma",
            "snippet": """from langchain_chroma import Chroma

# Local persistent directory storage
vectorstore = Chroma(
    collection_name="knowledge_store",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add texts and similarity search with score
vectorstore.add_texts(texts=["LangChain LCEL guide", "Vector DB benchmarks"])
results = vectorstore.similarity_search_with_relevance_scores("vector search", k=3)"""
        },
        "Qdrant": {
            "type": "Rust Cloud-Native / Self-Hosted",
            "index_algo": "Custom HNSW with Payload Storage",
            "p95_latency_ms": 12,
            "recall_pct": 99.0,
            "filtering_power": 10.0,
            "scalability": 9.5,
            "ops_simplicity": 8.0,
            "cost_score": 8.5,
            "quantization": "Scalar (SQ8) & Binary (BQ)",
            "best_for": "Heavy metadata filtering, hybrid vector/keyword search, distributed Kubernetes clusters.",
            "langchain_pkg": "langchain_qdrant",
            "snippet": """from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="support_docs",
    embedding=embeddings
)

# Query with combined payload filtering
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"must": [{"key": "permission", "match": {"value": "admin"}}]}
    }
)"""
        },
        "Milvus / Zilliz": {
            "type": "Distributed Kubernetes Vector DB",
            "index_algo": "Knowhere (HNSW, SCaNN, DiskANN)",
            "p95_latency_ms": 15,
            "recall_pct": 98.5,
            "filtering_power": 8.5,
            "scalability": 10.0,
            "ops_simplicity": 5.5,
            "cost_score": 7.5,
            "quantization": "DiskANN & Scalar Quantization",
            "best_for": "Massive enterprise scale (100M to billions of vectors), sharded cluster deployments.",
            "langchain_pkg": "langchain_milvus",
            "snippet": """from langchain_milvus import Milvus

vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={"host": "milvus-standalone", "port": "19530"},
    collection_name="enterprise_vectors",
    auto_id=True
)

retriever = vectorstore.as_retriever(search_kwargs={"expr": "views >= 100", "k": 10})"""
        },
        "pgvector (PostgreSQL)": {
            "type": "Relational Extension",
            "index_algo": "HNSW / IVFFlat",
            "p95_latency_ms": 28,
            "recall_pct": 96.5,
            "filtering_power": 10.0,
            "scalability": 7.0,
            "ops_simplicity": 8.5,
            "cost_score": 9.0,
            "quantization": "Halfvec (FP16) & Bit vectors",
            "best_for": "Teams already on Postgres, unified ACID SQL joins with vector search, single-database architecture.",
            "langchain_pkg": "langchain_postgres",
            "snippet": """from langchain_postgres import PGVector
from sqlalchemy import create_engine

connection = "postgresql+psycopg://user:password@localhost:5432/ragdb"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="legal_contracts",
    connection=connection,
    use_jsonb=True
)

# Integrated SQL + Vector query
docs = vectorstore.similarity_search("liability limitation clause", k=5)"""
        }
    }

    # Configuration Controls
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        selected_vdb = st.selectbox(
            "Select Vector Database Engine:",
            list(vector_dbs.keys()),
            index=0
        )
        vdb_info = vector_dbs[selected_vdb]

    with col_c2:
        vector_count = st.select_slider(
            "Dataset Scale (Number of Vectors):",
            options=[10_000, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000, 50_000_000],
            value=1_000_000,
            format_func=lambda x: f"{x:,.0f} vectors"
        )

    with col_c3:
        dimensions = st.selectbox(
            "Embedding Dimensions:",
            [384, 768, 1536, 3072],
            index=1,
            help="384 (MiniLM), 768 (Gemini/BERT), 1536 (OpenAI small), 3072 (OpenAI large)"
        )

    # Memory and storage sizing math
    # Raw float32 vector: dimensions * 4 bytes
    bytes_per_vector = dimensions * 4
    raw_vector_bytes = vector_count * bytes_per_vector
    raw_vector_gb = raw_vector_bytes / (1024 ** 3)
    
    # HNSW graph overhead multiplier (~1.5x - 2.0x raw vector size)
    index_multiplier = 1.6
    hnsw_index_gb = raw_vector_gb * index_multiplier
    total_uncompressed_ram_gb = raw_vector_gb + hnsw_index_gb
    
    # Quantized sizes
    sq8_compressed_ram_gb = total_uncompressed_ram_gb * 0.35
    pq_compressed_ram_gb = total_uncompressed_ram_gb * 0.15

    st.markdown("---")
    st.markdown("### 📐 Memory Footprint & Hardware Sizing")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="card">
            <div class="label">Raw Vector Data</div>
            <div style="font-size: 20px; font-weight: 700; color: {AI_BLUE}; margin: 4px 0;">{raw_vector_gb:.2f} GB</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">{vector_count:,.0f} × {dimensions} dims (FP32)</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="card">
            <div class="label">HNSW Index Overhead</div>
            <div style="font-size: 20px; font-weight: 700; color: {AI_PURPLE}; margin: 4px 0;">{hnsw_index_gb:.2f} GB</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">Graph neighbor lists (M=16, efC=64)</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="card">
            <div class="label">RAM Required (Uncompressed)</div>
            <div style="font-size: 20px; font-weight: 700; color: {AI_AMBER}; margin: 4px 0;">{total_uncompressed_ram_gb:.2f} GB</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">Full memory-resident index</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="card">
            <div class="label">RAM with SQ8 Quantization</div>
            <div style="font-size: 20px; font-weight: 700; color: {AI_EMERALD}; margin: 4px 0;">{sq8_compressed_ram_gb:.2f} GB</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">65% savings with &lt;1% recall loss</div>
        </div>
        """, unsafe_allow_html=True)

    # Deep Comparative Radar Chart
    st.markdown("---")
    st.markdown("### ⚖️ Architectural Trade-off Radar (6-Factor Analysis)")
    
    r_cats = ['Low Latency (Speed)', 'Recall Precision', 'Metadata Filtering', 'Horizontal Scalability', 'Ops Simplicity', 'Cost Efficiency']
    
    fig_v_radar = go.Figure()
    
    # Trace for selected DB
    sel_vals = [
        10 - (vdb_info['p95_latency_ms'] / 3.5), # higher is faster
        (vdb_info['recall_pct'] - 90) * 1.0,
        vdb_info['filtering_power'],
        vdb_info['scalability'],
        vdb_info['ops_simplicity'],
        vdb_info['cost_score']
    ]
    # normalize between 1 and 10
    sel_vals = [max(2, min(10, v)) for v in sel_vals]

    fig_v_radar.add_trace(go.Scatterpolar(
        r=sel_vals + [sel_vals[0]],
        theta=r_cats + [r_cats[0]],
        fill='toself',
        name=selected_vdb,
        line=dict(color=AI_BLUE, width=3),
        fillcolor='rgba(59, 130, 246, 0.3)'
    ))

    # Add a reference baseline (Pinecone vs Qdrant vs FAISS)
    ref_pinecone = [7.5, 8.5, 9.5, 10.0, 10.0, 7.0]
    fig_v_radar.add_trace(go.Scatterpolar(
        r=ref_pinecone + [ref_pinecone[0]],
        theta=r_cats + [r_cats[0]],
        name='Pinecone Reference',
        line=dict(color='#94A3B8', dash='dot', width=1.5)
    ))

    fig_v_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], color=TEXT_MUTED, gridcolor=BORDER_DARK),
            angularaxis=dict(color=TEXT_LIGHT, gridcolor=BORDER_DARK)
        ),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_LIGHT, size=11),
        height=320,
        margin=dict(l=20, r=20, t=10, b=30)
    )

    col_rad, col_spec = st.columns([6, 6])
    with col_rad:
        st.plotly_chart(fig_v_radar, use_container_width=True)

    with col_spec:
        st.markdown(f"""
        <div class="card">
            <div class="label" style="color: {AI_CYAN};">Profile: {selected_vdb}</div>
            <div style="margin-top: 8px; font-size: 12.5px; line-height: 1.6;">
                <div><strong>Deployment Model:</strong> {vdb_info['type']}</div>
                <div><strong>Indexing Algorithm:</strong> {vdb_info['index_algo']}</div>
                <div><strong>Typical P95 Latency:</strong> ~{vdb_info['p95_latency_ms']} ms</div>
                <div><strong>Recall Accuracy:</strong> ~{vdb_info['recall_pct']}%</div>
                <div><strong>Quantization Support:</strong> {vdb_info['quantization']}</div>
                <div style="margin-top: 6px; color: #60A5FA;"><strong>Recommended Best For:</strong> {vdb_info['best_for']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**LangChain Ingestion & Retriever Snippet ({vdb_info['langchain_pkg']}):**")
        st.code(vdb_info['snippet'], language="python")

# ---------------------------------------------------------
# MODULE 4: LANGCHAIN LCEL & RUNNABLE PROTOCOL
# ---------------------------------------------------------
def render_lcel_deep_dive():
    st.title("🦜 LangChain Expression Language (LCEL) & Runnable Architecture")
    st.caption("Mastering the declarative composition pattern powering modern production LangChain systems")

    st.markdown("""
    <div class="card">
        <div class="label" style="color: #60A5FA;">The Runnable Protocol</div>
        <p style="font-size: 12px; color: #CBD5E1; margin-top: 4px; line-height: 1.5;">
            LCEL replaces legacy imperative chains (like <code>RetrievalQA</code> or <code>LLMChain</code>) with a unified functional interface.
            Every component implements <code>Runnable</code>, providing automatic async execution (<code>ainvoke</code>), batch processing (<code>batch</code>), 
            and token-by-token streaming (<code>stream</code>) with zero boilerplate.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_pipe, tab_stream, tab_fallback, tab_routing = st.tabs([
        "🔗 Pipe Composition (|)",
        "🌊 Streaming & Async",
        "🛡️ Fallbacks & Retries",
        "🔀 Dynamic Branch Routing"
    ])

    with tab_pipe:
        st.markdown("#### The Standard LCEL RAG Composition")
        st.code("""from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Setup Retriever and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = PineconeVectorStore(index_name="docs", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# 2. Define Prompt Template
template = \"\"\"Answer the question based only on the following context:
{context}

Question: {question}\"\"\"
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

# 3. Declarative Chain Assembly with the Pipe Operator (|)
rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Invoke
response = rag_chain.invoke("What are the advantages of LCEL over legacy chains?")
print(response)""", language="python")

    with tab_stream:
        st.markdown("#### Token Streaming with astream & astream_events")
        st.code("""import asyncio

# Synchronous streaming
for chunk in rag_chain.stream("Explain vector indexing in 3 sentences"):
    print(chunk, end="", flush=True)

# Async streaming for FastAPI / WebSockets with event metadata
async def stream_rag_tokens(query: str):
    async for event in rag_chain.astream_events(query, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content  # Yield tokens directly to SSE / WebSocket
        elif kind == "on_retriever_end":
            docs = event["data"]["output"]
            print(f"[Trace] Retrieved {len(docs)} documents")""", language="python")

    with tab_fallback:
        st.markdown("#### Production Fallback Chains for 99.99% Availability")
        st.code("""from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Primary flagship model with aggressive timeout
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    request_timeout=3.0,
    max_retries=1
)

# Secondary failover model
backup_llm = ChatOpenAI(
    model="gpt-4o-mini",
    request_timeout=5.0
)

# Automated failover runnable: if Gemini times out or hits 429, switches to OpenAI instantly
robust_llm = primary_llm.with_fallbacks([backup_llm])

robust_chain = prompt | robust_llm | StrOutputParser()
output = robust_chain.invoke({"context": "...", "question": "..."})""", language="python")

    with tab_routing:
        st.markdown("#### Dynamic Conditional Routing with RunnableBranch")
        st.code("""from langchain_core.runnables import RunnableBranch

# Classify intent
intent_classifier = (
    ChatPromptTemplate.from_template("Classify user query into 'coding', 'billing', or 'general': {query}")
    | primary_llm
    | StrOutputParser()
)

coding_chain = coding_prompt | specialized_coding_llm | StrOutputParser()
billing_chain = billing_prompt | strict_schema_llm | StrOutputParser()
general_chain = general_prompt | fast_llm | StrOutputParser()

# Dynamic router branch
router = RunnableBranch(
    (lambda x: "coding" in x["intent"].lower(), coding_chain),
    (lambda x: "billing" in x["intent"].lower(), billing_chain),
    general_chain  # default fallback
)

full_routed_chain = (
    RunnableParallel({"intent": intent_classifier, "query": lambda x: x["query"]})
    | router
)""", language="python")

# ---------------------------------------------------------
# MODULE 5: ADVANCED RETRIEVAL (HYBRID, RRF, COMPRESSION)
# ---------------------------------------------------------
def render_retrieval_strategies():
    st.title("🔍 Advanced Retrieval: Hybrid Search, Reranking & Compression")
    st.caption("Overcoming dense embedding blind spots with BM25 sparse search, Reciprocal Rank Fusion, and Cross-Encoders")

    st.markdown("""
    <div class="card">
        <div class="label" style="color: #10B981;">Why Pure Vector Search Fails in Enterprise</div>
        <p style="font-size: 12px; color: #CBD5E1; margin-top: 4px; line-height: 1.5;">
            Dense embeddings excel at broad conceptual semantics, but fail miserably at exact keyword lookups (e.g. error code <code>ERR_503_TOKEN_EXPIRED</code>, SKU numbers, or person names). 
            Production systems combine <strong>BM25 Sparse Keyword Search</strong> + <strong>Dense Vector Search</strong> merged via <strong>Reciprocal Rank Fusion (RRF)</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Interactive RRF Score Demo
    st.markdown("### 🧮 Interactive Reciprocal Rank Fusion (RRF) Calculator")
    st.caption("Formula: Score(d) = Σ [ 1 / (k + rank_i(d)) ] where k is smoothing constant (typically k=60)")

    col_r1, col_r2 = st.columns([5, 7])
    with col_r1:
        rrf_k = st.slider("Smoothing Constant (k):", 1, 100, 60)
        st.markdown("**Simulated Document Ranks:**")
        doc_names = ["Doc A (Exact Error Match)", "Doc B (Broad Conceptual Match)", "Doc C (Old Release Notes)", "Doc D (API Reference)"]
        dense_ranks = [4, 1, 6, 2]
        bm25_ranks = [1, 5, 2, 8]

        rrf_scores = []
        for i in range(len(doc_names)):
            score = (1.0 / (rrf_k + dense_ranks[i])) + (1.0 / (rrf_k + bm25_ranks[i]))
            rrf_scores.append(score)

        df_rrf = pd.DataFrame({
            "Document": doc_names,
            "Dense Rank": dense_ranks,
            "BM25 Rank": bm25_ranks,
            "Combined RRF Score": [round(s, 5) for s in rrf_scores]
        }).sort_values(by="Combined RRF Score", ascending=False)
        st.dataframe(df_rrf, use_container_width=True)

    with col_r2:
        fig_rrf = px.bar(
            df_rrf,
            x="Document",
            y="Combined RRF Score",
            color="Combined RRF Score",
            color_continuous_scale="Viridis",
            title="Merged Hybrid RRF Ranking"
        )
        fig_rrf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT))
        st.plotly_chart(fig_rrf, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🛠️ Production Ensemble Retriever Code in LangChain")
    st.code("""from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore

# 1. Sparse keyword retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# 2. Dense semantic retriever
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Hybrid Ensemble with Reciprocal Rank Fusion
# 50% weight to BM25 keyword matching, 50% weight to dense semantic vector
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)

# Returns documents that score well across both lexical and vector dimensions
hybrid_docs = ensemble_retriever.invoke("Exception ERR_401 invalid bearer token in OAuth flow")""", language="python")

# ---------------------------------------------------------
# MODULE 6: AGENTIC WORKFLOWS & LANGGRAPH STATE MACHINES
# ---------------------------------------------------------
def render_langgraph_agents():
    st.title("🤖 Agentic Systems & LangGraph State Machines")
    st.caption("Designing resilient multi-agent cyclic graphs with persistent state, tool calling, and human-in-the-loop approvals")

    st.markdown("""
    <div class="card">
        <div class="label" style="color: #8B5CF6;">From Linear Chains to Stateful Cyclic Graphs</div>
        <p style="font-size: 12px; color: #CBD5E1; margin-top: 4px; line-height: 1.5;">
            Traditional LangChain agents used rigid single-while-loops (ReAct). 
            <strong>LangGraph</strong> generalizes this by modeling agent architectures as <strong>StateGraphs</strong> with typed state, conditional edges, 
            resumable checkpoints, and multi-agent debate/critique loops.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns([6, 6])
    
    with col_g1:
        st.markdown("#### LangGraph Multi-Agent Architecture Pattern")
        st.code("""from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. Define Typed Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research_summary: str
    iterations: int

# 2. Define State Machine Graph
workflow = StateGraph(AgentState)

# 3. Add Processing Nodes
workflow.add_node("agent", call_reasoning_model)
workflow.add_node("tools", ToolNode(tools=[web_search, sql_query]))
workflow.add_node("reflection", critique_response_quality)

# 4. Define Conditional Edges
def route_next_step(state: AgentState):
    if state["iterations"] > 5:
        return "end"
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "reflection"

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", route_next_step, {
    "tools": "tools",
    "reflection": "reflection",
    "end": END
})
workflow.add_edge("tools", "agent")
workflow.add_edge("reflection", END)

# 5. Compile with durable memory checkpointer
from langgraph.checkpoint.memory import MemorySaver
app = workflow.compile(checkpointer=MemorySaver())""", language="python")

    with col_g2:
        st.markdown("#### Core LangGraph Concepts vs Legacy Agents")
        concepts = [
            ("StateGraph", "A directed graph where every node accepts the current state and returns state updates governed by schema reducers."),
            ("Checkpointer (Time Travel)", "Persists state snapshots after each node step (in Redis/Postgres), enabling rollback, inspection, and human approval before actions."),
            ("ToolNode", "Prebuilt node that executes tool calls generated by LLMs with automatic error recovery and schema verification."),
            ("Sub-Graphs & Multi-Agent", "Hierarchical graphs where a Supervisor Agent coordinates specialist worker sub-graphs (e.g. Researcher, Coder, Reviewer)."),
            ("Human-in-the-Loop", "Pauses execution before sensitive nodes (e.g. executing SQL write or email dispatch) until human confirmation is supplied.")
        ]
        for name, desc in concepts:
            st.markdown(f"""
            <div style="background: {BG_CARD}; border: 1px solid {BORDER_DARK}; border-radius: 6px; padding: 10px 12px; margin-bottom: 8px;">
                <div style="font-weight: 700; font-size: 12px; color: {AI_PURPLE};">{name}</div>
                <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 7: EVALUATION & OBSERVABILITY (RAGAS & GUARDRAILS)
# ---------------------------------------------------------
def render_evaluation_observability():
    st.title("🛡️ Production AI Observability & Ragas Evaluation")
    st.caption("Quantifying RAG quality with the RAG Triad, automated LLM-as-a-judge scoring, and schema guardrails")

    st.markdown("""
    <div class="card">
        <div class="label" style="color: #EC4899;">The RAG Evaluation Triad</div>
        <p style="font-size: 12px; color: #CBD5E1; margin-top: 4px; line-height: 1.5;">
            You cannot improve what you cannot measure. Production AI systems monitor three fundamental pillars:
            <strong>Context Relevance</strong> (did retriever fetch noise?), <strong>Faithfulness / Groundedness</strong> (did LLM hallucinate beyond context?), 
            and <strong>Answer Relevance</strong> (did response actually address the question?).
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown("""
        <div class="card">
            <div class="label" style="color: #60A5FA;">1. Context Precision / Recall</div>
            <div style="font-size: 20px; font-weight: 700; color: #38BDF8; margin: 4px 0;">Retriever Quality</div>
            <p style="font-size: 11px; color: #94A3B8;">Measures whether all relevant chunks were fetched with high signal-to-noise ratio in top-k.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_e2:
        st.markdown("""
        <div class="card">
            <div class="label" style="color: #10B981;">2. Faithfulness (Groundedness)</div>
            <div style="font-size: 20px; font-weight: 700; color: #34D399; margin: 4px 0;">Zero Hallucinations</div>
            <p style="font-size: 11px; color: #94A3B8;">Measures if claims in the answer can be mathematically derived from the retrieved context alone.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_e3:
        st.markdown("""
        <div class="card">
            <div class="label" style="color: #F59E0B;">3. Answer Relevance</div>
            <div style="font-size: 20px; font-weight: 700; color: #FBBF24; margin: 4px 0;">User Satisfaction</div>
            <p style="font-size: 11px; color: #94A3B8;">Measures if the generated answer directly addresses the prompt without evasive verbosity.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Automated Ragas Evaluation Script")
    st.code("""from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Prepare benchmark evaluation dataset
eval_data = {
    "question": ["How does HNSW indexing work in vector databases?"],
    "contexts": [["HNSW builds multi-layer graphs where top layers have longer skip links..."]],
    "answer": ["HNSW constructs a hierarchical graph structure for sub-millisecond nearest neighbor search."],
    "ground_truth": ["HNSW uses a multi-layer graph skip-list structure for fast approximate nearest neighbor retrieval."]
}
dataset = Dataset.from_dict(eval_data)

# Run automated LLM-as-a-judge evaluation
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results.to_pandas())""", language="python")

# ---------------------------------------------------------
# MODULE 8: AI ENGINEERING READINESS CHECKLIST
# ---------------------------------------------------------
def render_competency_checklist():
    st.title("📐 AI Systems Engineering Competencies & Readiness")
    st.caption("Self-assessment checklist for Senior AI Engineer & Staff AI Systems Architect roles")

    total_tasks = len(st.session_state.ai_readiness_checklist)
    done_tasks = sum(1 for v in st.session_state.ai_readiness_checklist.values() if v)
    readiness_pct = int((done_tasks / total_tasks) * 100) if total_tasks else 0

    col_bar1, col_bar2 = st.columns([8, 4])
    with col_bar1:
        st.markdown(f"**Overall Architectural Mastery: {readiness_pct}% ({done_tasks}/{total_tasks} competencies mastered)**")
        st.progress(readiness_pct / 100)
    with col_bar2:
        if st.button("Reset / Check All"):
            all_true = not all(st.session_state.ai_readiness_checklist.values())
            for k in st.session_state.ai_readiness_checklist:
                st.session_state.ai_readiness_checklist[k] = all_true
            st.rerun()

    st.markdown("---")
    cols = st.columns(2)
    items = list(st.session_state.ai_readiness_checklist.items())
    
    for idx, (task, state) in enumerate(items):
        with cols[idx % 2]:
            new_val = st.checkbox(task, value=state, key=f"chk_{idx}")
            if new_val != state:
                st.session_state.ai_readiness_checklist[task] = new_val
                st.rerun()

# ---------------------------------------------------------
# MODULE 9: AI SYSTEMS RESEARCH RADAR (SEARCH GROUNDED)
# ---------------------------------------------------------
def render_tech_radar():
    st.title("📡 AI Systems Research Radar & Real-Time Grounding")
    st.caption("Live Google Search grounded research on LangChain, Vector DBs, LLM Serving & Distributed AI Systems")

    st.markdown("""
    <div class="card">
        <div class="label" style="color: #60A5FA;">Google Search Grounded Architectural Intelligence</div>
        <p style="font-size: 12px; color: #CBD5E1; margin-top: 4px; line-height: 1.5;">
            Query cutting-edge developments across official LangChain docs, vector store whitepapers, and production LLM scaling papers.
            Or click the <strong>AI Tech Radar</strong> button in the top right to open the sliding search drawer anytime!
        </p>
    </div>
    """, unsafe_allow_html=True)

    radar_query = st.text_input(
        "Enter AI Engineering Query:",
        value="LangGraph StateGraph multi-agent workflow human in the loop"
    )

    if st.button("🔍 Search AI Knowledge Base", type="primary"):
        with st.spinner("Querying Google Search Grounding & Synthesizing Architectural Analysis..."):
            try:
                import urllib.request
                import urllib.parse
                
                url = f"http://localhost:3000/api/search?q={urllib.parse.quote(radar_query)}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Pyodide-Client'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    res_body = response.read().decode('utf-8')
                    payload = json.loads(res_body)
                    
                    st.markdown("### 📄 Grounded Architectural Synthesis")
                    st.markdown(payload.get("content", "No synthesis returned."))
                    
                    sources = payload.get("sources", [])
                    if sources:
                        st.markdown("---")
                        st.markdown("**🔗 Verified Sources & Reference Documentation:**")
                        source_cols = st.columns(min(len(sources), 3))
                        for idx, src in enumerate(sources):
                            with source_cols[idx % 3]:
                                st.markdown(f"- [{src.get('title', 'Source')}]({src.get('uri', '#')})")

            except Exception as e:
                st.info("Tip: Click the 'AI Tech Radar' button in the top right header for instant sliding research drawer with real-time web search grounding.")

# ---------------------------------------------------------
# MAIN NAVIGATION ROUTER
# ---------------------------------------------------------
pages = {
    "🏠 Overview & Blueprint Dashboard": render_dashboard,
    "🔁 1. RAG Sequence Diagram (Interactive)": render_rag_sequence,
    "💰 2. Scaling Cost & Token Calculator": render_cost_calculator,
    "🗄️ 3. Vector DB Configurator & Trade-offs": render_vector_db_configurator,
    "🦜 4. LangChain LCEL & Runnables": render_lcel_deep_dive,
    "🔍 5. Hybrid Search & Reranking": render_retrieval_strategies,
    "🤖 6. LangGraph & Multi-Agent Systems": render_langgraph_agents,
    "🛡️ 7. AI Observability & Ragas Eval": render_evaluation_observability,
    "📐 8. Competencies & Readiness Checklist": render_competency_checklist,
    "📡 9. AI Systems Research Radar": render_tech_radar
}

# Sidebar Branding & Navigation
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; padding: 4px;">
    <div style="width: 32px; height: 32px; background: linear-gradient(135deg, {AI_BLUE}, {AI_PURPLE}); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px; color: white; box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);">🦜</div>
    <div>
        <div style="font-weight: 700; font-size: 13px; color: #FFFFFF; letter-spacing: -0.02em;">LANGCHAIN & AI SYSTEMS</div>
        <div style="font-size: 10px; color: {TEXT_MUTED}; text-transform: uppercase;">Production Engineering Guide</div>
    </div>
</div>
""", unsafe_allow_html=True)

selection = st.sidebar.radio("Architecture Modules:", list(pages.keys()))

st.sidebar.markdown("---")
total_t = len(st.session_state.ai_readiness_checklist)
done_t = sum(1 for v in st.session_state.ai_readiness_checklist.values() if v)
readiness = int((done_t / total_t) * 100) if total_t else 0

st.sidebar.markdown(f"""
<div style="margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
    <span class="label">Candidate Readiness</span>
    <span style="font-size: 11px; font-weight: 700; color: {AI_BLUE}; font-family: monospace;">{readiness}%</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.progress(readiness / 100)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background: #162032; border: 1px solid #2D3A4F; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: #60A5FA; letter-spacing: 0.05em; margin-bottom: 4px;">📡 AI Tech Radar</div>
    <div style="font-size: 11px; color: #94A3B8; margin-bottom: 8px;">Semantic Google Search Grounding for modern LangChain & AI system breakthroughs.</div>
    <button onclick="if(window.parent && window.parent.toggleSearchSidebar){{window.parent.toggleSearchSidebar(true)}}else if(window.toggleSearchSidebar){{window.toggleSearchSidebar(true)}}" style="width: 100%; background: linear-gradient(135deg, {AI_BLUE}, {AI_PURPLE}); color: white; border: none; border-radius: 4px; padding: 6px 10px; font-size: 10.5px; font-weight: 700; text-transform: uppercase; cursor: pointer;">Open Radar Search 🔍</button>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("Designed for AI Engineers & LLM Architects • Powered by Pyodide, Express & Gemini 3.8")

# Execute page function
pages[selection]()
