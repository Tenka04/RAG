"""
Agent Graph: LangGraph-powered Multi-Agent & Tool Orchestration Engine.

Includes:
  - Supervisor / Intent Router Node
  - Document RAG Agent Node (connected to ChromaDB)
  - Web Search Agent Tool (DuckDuckGo + Wikipedia live search)
  - Calculator / Python Logic Tool (Safe AST computation)
  - Evaluator Node (Corrective RAG: fallback to web search if local docs lack answers)
  - Response Synthesizer Node (Local Ollama / OpenRouter streaming)
  - Real-time SSE streaming bridge compatible with Next.js frontend & DevInspector
"""

import os
import sys
import time
import math
import ast
import operator
import urllib.parse
from typing import TypedDict, List, Dict, Any, Optional, Generator
from typing_extensions import Annotated
import requests

# LangGraph & LangChain Core imports
from langgraph.graph import StateGraph, START, END

# Ensure backend directory is in sys.path
from pathlib import Path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from rag_engine import RAGEngine


# ==============================================================================
# 1. State Definition
# ==============================================================================

class AgentState(TypedDict):
    query: str
    model: str
    top_k: int
    selected_action: str  # "rag", "web_search", "calculator", "direct_chat"
    routing_reason: str
    documents: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    final_answer: str
    timing_steps: Annotated[List[Dict[str, Any]], operator.add]
    bottlenecks: Annotated[List[str], operator.add]
    agent_status_updates: Annotated[List[str], operator.add]


# ==============================================================================
# 2. Tool Implementations
# ==============================================================================

SAFE_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
    "pi": math.pi,
    "e": math.e,
}


def safe_math_eval(expr: str) -> float:
    """Safely evaluate mathematical expressions using AST without insecure eval()."""
    clean_expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    # Remove common prefixes like 'what is', 'calculate', '='
    for prefix in ["what is", "calculate", "solve", "evaluate", "compute", "="]:
        if clean_expr.lower().startswith(prefix):
            clean_expr = clean_expr[len(prefix):].strip()

    tree = ast.parse(clean_expr.strip(), mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in SAFE_MATH_OPS:
                return SAFE_MATH_OPS[op_type](left, right)
            raise ValueError(f"Unsupported math operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in SAFE_MATH_OPS:
                return SAFE_MATH_OPS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        elif isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", "")
            if func_name in SAFE_MATH_FUNCS:
                args = [_eval(arg) for arg in node.args]
                return SAFE_MATH_FUNCS[func_name](*args)
            raise ValueError(f"Unsupported math function: {func_name}")
        elif isinstance(node, ast.Name) and node.id in SAFE_MATH_FUNCS:
            return SAFE_MATH_FUNCS[node.id]
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return _eval(tree)


def web_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Live web search using DuckDuckGo Instant Answers and Wikipedia."""
    results: List[Dict[str, Any]] = []

    # 1. DuckDuckGo Instant Answer API
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("AbstractText", "")
            source_url = data.get("AbstractURL", "")
            if abstract:
                results.append({
                    "content": abstract,
                    "metadata": {
                        "source": source_url or "https://duckduckgo.com",
                        "title": data.get("Heading", "DuckDuckGo Answer"),
                        "type": "web_search",
                        "chunk_id": "ddg_0"
                    },
                    "score": 0.95,
                    "raw_distance": 0.05
                })

            for i, topic in enumerate(data.get("RelatedTopics", [])[:max_results]):
                if isinstance(topic, dict) and "Text" in topic:
                    first_url = topic.get("FirstURL", "https://duckduckgo.com")
                    results.append({
                        "content": topic["Text"],
                        "metadata": {
                            "source": first_url,
                            "title": topic.get("Text", "")[:40] + "...",
                            "type": "web_search",
                            "chunk_id": f"ddg_rel_{i}"
                        },
                        "score": 0.88,
                        "raw_distance": 0.12
                    })
    except Exception:
        pass

    # 2. Wikipedia Search fallback/enrichment
    if len(results) < 2:
        try:
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&utf8=1"
            headers = {"User-Agent": "DeepRAG-LangGraph/1.0 (https://github.com/)"}
            resp = requests.get(wiki_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for i, item in enumerate(data.get("query", {}).get("search", [])[:max_results]):
                    snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
                    title = item.get("title", "")
                    clean_title = title.replace(" ", "_")
                    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(clean_title)}"
                    results.append({
                        "content": f"{title}: {snippet}",
                        "metadata": {
                            "source": page_url,
                            "title": f"Wikipedia: {title}",
                            "type": "wikipedia",
                            "chunk_id": f"wiki_{i}"
                        },
                        "score": 0.90,
                        "raw_distance": 0.10
                    })
        except Exception:
            pass

    return results[:max_results]


# ==============================================================================
# 3. Graph Nodes
# ==============================================================================

class RAGAgentGraph:
    """Encapsulates the LangGraph multi-agent pipeline and tool orchestration."""

    def __init__(self, engine: RAGEngine):
        self.engine = engine
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("router", self.router_node)
        builder.add_node("rag_agent", self.rag_node)
        builder.add_node("web_search_agent", self.web_search_node)
        builder.add_node("calculator_agent", self.calculator_node)
        builder.add_node("synthesizer", self.synthesizer_node)

        # Edge from START to router
        builder.add_edge(START, "router")

        # Conditional edges from router
        builder.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "rag": "rag_agent",
                "web_search": "web_search_agent",
                "calculator": "calculator_agent",
                "direct_chat": "synthesizer"
            }
        )

        # Conditional edge from RAG agent (Corrective RAG)
        builder.add_conditional_edges(
            "rag_agent",
            self._evaluate_rag_decision,
            {
                "synthesize": "synthesizer",
                "fallback_web": "web_search_agent"
            }
        )

        # Edges from tools directly to synthesizer
        builder.add_edge("web_search_agent", "synthesizer")
        builder.add_edge("calculator_agent", "synthesizer")
        builder.add_edge("synthesizer", END)

        return builder.compile()

    # --------------------------------------------------------------------------
    # Node logic
    # --------------------------------------------------------------------------

    def router_node(self, state: AgentState) -> Dict[str, Any]:
        """Supervisor router: Evaluates intent and picks the appropriate agent/tool."""
        t_start = time.perf_counter()
        query = state["query"].strip()
        q_lower = query.lower()

        # 1. Casual Chat Check
        if self.engine._is_casual_message(query):
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "selected_action": "direct_chat",
                "routing_reason": "Casual conversational greeting or identity question.",
                "timing_steps": [{
                    "name": "1. Supervisor Router",
                    "duration_ms": duration_ms,
                    "details": "Classified as casual chat -> direct synthesis"
                }],
                "agent_status_updates": ["Supervisor: Casual conversation detected. Bypassing retrieval."]
            }

        # 2. Math / Calculator Check (Heuristic + safe parse)
        is_math = False
        math_symbols = ["+", "*", "/", "%", "^", "sqrt", "pi", "sin", "cos", "log", "="]
        has_symbols = any(sym in query for sym in math_symbols)
        has_digits = any(c.isdigit() for c in query)
        math_words = ["calculate", "solve", "compute", "what is", "how much is", "plus", "times", "divided by"]

        if (has_symbols and has_digits) or any(q_lower.startswith(w) for w in math_words if has_digits):
            # Test if safe_math_eval can parse or partially parse
            try:
                # Extract potential math part
                raw_test = query
                for prefix in ["what is", "calculate", "solve", "how much is", "compute", "value of"]:
                    if raw_test.lower().startswith(prefix):
                        raw_test = raw_test[len(prefix):].strip()
                raw_test = raw_test.rstrip("?").strip()
                _ = safe_math_eval(raw_test)
                is_math = True
            except Exception:
                is_math = False

        if is_math:
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "selected_action": "calculator",
                "routing_reason": "Mathematical formula or computational query detected.",
                "timing_steps": [{
                    "name": "1. Supervisor Router",
                    "duration_ms": duration_ms,
                    "details": "Classified as math/computation -> Python Calculator Tool"
                }],
                "agent_status_updates": ["Supervisor: Mathematical intent detected -> Routing to Calculator Tool."]
            }

        # 3. Knowledge Base vs Web Search
        doc_count = self.engine.collection.count()
        web_keywords = ["latest news", "current price", "today", "weather", "who is the current", "wikipedia"]
        is_explicit_web = any(kw in q_lower for kw in web_keywords)

        if doc_count > 0 and not is_explicit_web:
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "selected_action": "rag",
                "routing_reason": f"Knowledge base contains {doc_count} chunks. Querying local vector store.",
                "timing_steps": [{
                    "name": "1. Supervisor Router",
                    "duration_ms": duration_ms,
                    "details": f"Classified as document inquiry -> Local RAG Agent ({doc_count} chunks indexed)"
                }],
                "agent_status_updates": [f"Supervisor: Querying Document Knowledge Base ({doc_count} chunks in ChromaDB)."]
            }
        else:
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            reason = "No documents indexed in knowledge base" if doc_count == 0 else "Query requests live/current external information"
            return {
                "selected_action": "web_search",
                "routing_reason": f"{reason}. Routing to live Web Search Tool.",
                "timing_steps": [{
                    "name": "1. Supervisor Router",
                    "duration_ms": duration_ms,
                    "details": "Routing to Web Search Tool"
                }],
                "agent_status_updates": ["Supervisor: Routing to Web Search Tool for external information."]
            }

    def rag_node(self, state: AgentState) -> Dict[str, Any]:
        """RAG Agent Node: Performs semantic search against ChromaDB."""
        t_start = time.perf_counter()
        query = state["query"]
        top_k = state.get("top_k", 3)

        retrieval_data = self.engine.retrieve(query, top_k=top_k, verbose=False)
        raw_sources = retrieval_data.get("chunks", [])
        # Apply semantic distance threshold
        sources = [s for s in raw_sources if s.get("raw_distance", 2.0) <= 1.45]

        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        step = {
            "name": "2. ChromaDB RAG Agent",
            "duration_ms": duration_ms,
            "details": f"Retrieved {len(sources)}/{len(raw_sources)} valid chunks (embedding: {retrieval_data.get('embedding_ms', 0):.1f}ms)"
        }

        bottlenecks = []
        if retrieval_data.get("embedding_ms", 0) > 1000:
            bottlenecks.append(f"Query embedding latency was high ({retrieval_data.get('embedding_ms'):.1f} ms).")

        return {
            "documents": sources,
            "timing_steps": [step],
            "bottlenecks": bottlenecks,
            "agent_status_updates": [f"RAG Agent: Retrieved {len(sources)} grounded context chunk(s) from ChromaDB."]
        }

    def web_search_node(self, state: AgentState) -> Dict[str, Any]:
        """Web Search Agent Tool: Fetches live data from DuckDuckGo / Wikipedia."""
        t_start = time.perf_counter()
        query = state["query"]

        results = web_search(query, max_results=3)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "documents": results,
            "tool_results": [{"tool": "web_search", "count": len(results)}],
            "timing_steps": [{
                "name": "2. Web Search Agent Tool",
                "duration_ms": duration_ms,
                "details": f"Found {len(results)} live search result(s) via DuckDuckGo/Wikipedia"
            }],
            "agent_status_updates": [f"Web Agent: Retrieved {len(results)} live web snippet(s)."]
        }

    def calculator_node(self, state: AgentState) -> Dict[str, Any]:
        """Calculator Tool: Computes exact math expressions."""
        t_start = time.perf_counter()
        query = state["query"]

        try:
            raw_test = query
            for prefix in ["what is", "calculate", "solve", "how much is", "compute", "value of"]:
                if raw_test.lower().startswith(prefix):
                    raw_test = raw_test[len(prefix):].strip()
            raw_test = raw_test.rstrip("?").strip()
            calc_val = safe_math_eval(raw_test)
            res_str = f"Calculated Result: {raw_test} = {calc_val}"
            error = None
        except Exception as e:
            calc_val = None
            res_str = f"Math Calculation Error: {str(e)}"
            error = str(e)

        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        tool_item = {
            "tool": "calculator",
            "expression": query,
            "result": calc_val,
            "summary": res_str,
            "error": error
        }

        calc_doc = {
            "content": res_str,
            "metadata": {
                "source": "Python Safe Calculator Tool",
                "title": "Exact Mathematical Computation",
                "type": "tool_execution",
                "chunk_id": "calc_0"
            },
            "score": 1.0,
            "raw_distance": 0.0
        }

        return {
            "tool_results": [tool_item],
            "documents": [calc_doc],
            "timing_steps": [{
                "name": "2. Calculator Agent Tool",
                "duration_ms": duration_ms,
                "details": f"Evaluated safely: {res_str}"
            }],
            "agent_status_updates": [f"Calculator Agent: Computed result -> {calc_val}"]
        }

    def synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesizer Node: Formulates grounded context for response generation."""
        t_start = time.perf_counter()
        query = state["query"]
        docs = state.get("documents", [])
        tool_results = state.get("tool_results", [])

        # Build prompt
        prompt = self.engine.build_prompt(query, docs if docs else None)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "timing_steps": [{
                "name": "3. Prompt & Context Assembler",
                "duration_ms": duration_ms,
                "details": f"Assembled prompt with {len(docs)} context chunk(s) and {len(tool_results)} tool result(s)"
            }],
            "agent_status_updates": ["Synthesizer: Formulating grounded generation..."]
        }

    # --------------------------------------------------------------------------
    # Conditional edge routers
    # --------------------------------------------------------------------------

    def _route_decision(self, state: AgentState) -> str:
        return state.get("selected_action", "direct_chat")

    def _evaluate_rag_decision(self, state: AgentState) -> str:
        """Corrective RAG: If ChromaDB has 0 matching chunks, fallback to Web Search."""
        docs = state.get("documents", [])
        if len(docs) > 0:
            return "synthesize"
        # Zero local matches: Trigger Corrective RAG
        return "fallback_web"


# ==============================================================================
# 4. Real-Time SSE Streaming Bridge
# ==============================================================================

def stream_agent_execution(
    engine: RAGEngine,
    query: str,
    model_override: Optional[str] = None,
    top_k: int = 3
) -> Generator[Dict[str, Any], None, None]:
    """
    Executes the LangGraph multi-agent workflow while streaming Server-Sent Events (SSE).
    Fully compatible with the Next.js frontend and Developer Inspector.

    Yields:
      - {"type": "agent_action", "agent": "...", "status": "...", "details": "..."}
      - {"type": "sources", "sources": [...]}
      - {"type": "token", "token": "..."}
      - {"type": "dev_telemetry", "telemetry": {...}}
      - {"type": "done"}
    """
    t_start = time.perf_counter()
    active_model = model_override or engine.llm_model

    # Initialize LangGraph Agent
    agent_graph = RAGAgentGraph(engine)

    # Initial state
    state: AgentState = {
        "query": query,
        "model": active_model,
        "top_k": top_k,
        "selected_action": "rag",
        "routing_reason": "",
        "documents": [],
        "tool_results": [],
        "final_answer": "",
        "timing_steps": [],
        "bottlenecks": [],
        "agent_status_updates": []
    }

    # Instant sub-millisecond answer for local system date/time
    quick_ans = engine._get_quick_system_answer(query)
    if quick_ans:
        yield {
            "type": "agent_action",
            "agent": "supervisor",
            "status": "System Clock Tool: Resolving instant date/time...",
            "details": quick_ans
        }
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "token": quick_ans}
        total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
        telemetry = {
            "query": query,
            "total_time_ms": total_time_ms,
            "retrieval_time_ms": 0.0,
            "embedding_time_ms": 0.0,
            "llm_time_ms": total_time_ms,
            "ttft_ms": total_time_ms,
            "tokens_count": len(quick_ans.split()),
            "tokens_per_second": 999.0,
            "timing_steps": [{"name": "1. Instant Clock Tool", "duration_ms": total_time_ms, "details": "System clock"}],
            "bottlenecks": [],
            "model_used": "System Clock Tool (Instant ⚡)",
            "used_fallback": False,
            "fallback_reason": "",
            "prompt_preview": query,
            "chunks_retrieved_count": 0,
            "raw_chunks": []
        }
        yield {"type": "dev_telemetry", "telemetry": telemetry}
        yield {"type": "done"}
        return

    # --------------------------------------------------------------------------
    # Step 1: Supervisor / Router
    # --------------------------------------------------------------------------
    yield {
        "type": "agent_action",
        "agent": "supervisor",
        "status": "Supervisor: Analyzing query intent...",
        "details": query
    }

    router_res = agent_graph.router_node(state)
    state["selected_action"] = router_res["selected_action"]
    state["routing_reason"] = router_res["routing_reason"]
    state["timing_steps"].extend(router_res.get("timing_steps", []))
    action = state["selected_action"]

    yield {
        "type": "agent_action",
        "agent": "supervisor",
        "status": f"Routing to {action.upper()}: {router_res['routing_reason']}",
        "details": router_res["routing_reason"]
    }

    # --------------------------------------------------------------------------
    # Step 2: Agent / Tool Execution
    # --------------------------------------------------------------------------
    if action == "rag":
        yield {
            "type": "agent_action",
            "agent": "rag",
            "status": "Querying ChromaDB vector index...",
            "details": f"Top-K={top_k}"
        }
        rag_res = agent_graph.rag_node(state)
        state["documents"] = rag_res["documents"]
        state["timing_steps"].extend(rag_res.get("timing_steps", []))
        if rag_res.get("bottlenecks"):
            state["bottlenecks"].extend(rag_res["bottlenecks"])

        # Corrective RAG Check: If no documents found in ChromaDB, search Web!
        if len(state["documents"]) == 0:
            yield {
                "type": "agent_action",
                "agent": "evaluator",
                "status": "Corrective RAG: No matching docs in local KB. Triggering live Web Search...",
                "details": "Corrective loop"
            }
            web_res = agent_graph.web_search_node(state)
            state["documents"] = web_res["documents"]
            state["tool_results"].extend(web_res.get("tool_results", []))
            state["timing_steps"].extend(web_res.get("timing_steps", []))

    elif action == "web_search":
        yield {
            "type": "agent_action",
            "agent": "web_search",
            "status": "Searching the web (DuckDuckGo + Wikipedia)...",
            "details": query
        }
        web_res = agent_graph.web_search_node(state)
        state["documents"] = web_res["documents"]
        state["tool_results"].extend(web_res.get("tool_results", []))
        state["timing_steps"].extend(web_res.get("timing_steps", []))

    elif action == "calculator":
        yield {
            "type": "agent_action",
            "agent": "calculator",
            "status": "Evaluating mathematical calculation...",
            "details": query
        }
        calc_res = agent_graph.calculator_node(state)
        state["documents"] = calc_res["documents"]
        state["tool_results"].extend(calc_res.get("tool_results", []))
        state["timing_steps"].extend(calc_res.get("timing_steps", []))

    elif action == "direct_chat":
        state["documents"] = []

    # Yield sources early to frontend
    sources = state["documents"]
    yield {"type": "sources", "sources": sources}

    # --------------------------------------------------------------------------
    # Step 3: Response Synthesis & LLM Streaming
    # --------------------------------------------------------------------------
    yield {
        "type": "agent_action",
        "agent": "synthesizer",
        "status": f"Synthesizing response with model: {active_model}...",
        "details": f"Context sources: {len(sources)}"
    }

    prompt = engine.build_prompt(query, sources if sources else None)

    # Stream LLM tokens
    t_llm_start = time.perf_counter()
    first_token_time: Optional[float] = None
    token_count = 0
    full_answer_parts: List[str] = []

    try:
        # Check if direct synthesis or model override
        if active_model == "direct_synthesis":
            direct_ans = engine._generate_fallback_synthesis(query, sources)
            for word in direct_ans.split(" "):
                yield {"type": "token", "token": word + " "}
                time.sleep(0.01)
                token_count += 1
            full_answer = direct_ans
        elif engine._is_openrouter_model(active_model):
            # OpenRouter Cloud Streaming
            stream = engine._generate_openrouter_stream(active_model, prompt)
            for chunk in stream:
                if first_token_time is None:
                    first_token_time = round((time.perf_counter() - t_llm_start) * 1000, 2)
                token_count += 1
                full_answer_parts.append(chunk)
                yield {"type": "token", "token": chunk}
            full_answer = "".join(full_answer_parts)
        else:
            # Local Ollama Streaming on GPU
            if engine._check_local_ollama_available():
                stream = engine.local_client.chat(
                    model=active_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    keep_alive="24h",
                    options={"num_ctx": 2048, "num_predict": 1024}
                )
            else:
                import ollama
                stream = ollama.chat(
                    model=active_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )

            in_reasoning = False
            line_buf = ""
            lines_delivered = 0
            target_prog_lines = 2

            def _extract_tokens():
                nonlocal in_reasoning
                for chunk in stream:
                    msg = getattr(chunk, "message", None) or (chunk.get("message", {}) if isinstance(chunk, dict) else {})
                    thinking = getattr(msg, "thinking", None) or (msg.get("thinking") if isinstance(msg, dict) else None)
                    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                    if thinking:
                        if not in_reasoning:
                            in_reasoning = True
                            yield "> **Thinking Process:**\n> "
                        yield thinking.replace("\n", "\n> ")
                    if content:
                        if in_reasoning:
                            in_reasoning = False
                            yield "\n\n"
                        yield content

            for token_text in _extract_tokens():
                if first_token_time is None:
                    first_token_time = round((time.perf_counter() - t_llm_start) * 1000, 2)
                token_count += 1
                full_answer_parts.append(token_text)

                # Progressive line-by-line delivery for smooth visual onset
                if lines_delivered < target_prog_lines:
                    line_buf += token_text
                    if "\n" in line_buf or len(line_buf) >= 60:
                        yield {"type": "token", "token": line_buf}
                        line_buf = ""
                        lines_delivered += 1
                        time.sleep(0.02)
                else:
                    if line_buf:
                        yield {"type": "token", "token": line_buf}
                        line_buf = ""
                    # Instant text delivery for subsequent tokens
                    yield {"type": "token", "token": token_text}

            if line_buf:
                yield {"type": "token", "token": line_buf}
            full_answer = "".join(full_answer_parts)

    except Exception as exc:
        fallback_msg = f"\n\n*[Agent fallback: Generation error with {active_model}: {str(exc)}]*"
        yield {"type": "token", "token": fallback_msg}
        full_answer = "".join(full_answer_parts) + fallback_msg

    t_llm_end = time.perf_counter()
    llm_duration_ms = round((t_llm_end - t_llm_start) * 1000, 2)
    ttft_ms = first_token_time or 0.0

    state["timing_steps"].append({
        "name": "4. LLM Generation & Token Stream",
        "duration_ms": llm_duration_ms,
        "details": f"model={active_model}, tokens={token_count}, ttft={ttft_ms:.1f}ms"
    })

    # --------------------------------------------------------------------------
    # Step 4: Dev Telemetry
    # --------------------------------------------------------------------------
    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    tps = round((token_count / (llm_duration_ms / 1000)), 1) if llm_duration_ms > 0 else 0.0

    raw_chunks_formatted = [
        {
            "id": s.get("metadata", {}).get("chunk_id", i),
            "source": s.get("metadata", {}).get("source", "Unknown"),
            "score": s.get("score", "N/A"),
            "raw_distance": s.get("raw_distance", 0.0),
            "content_preview": (s.get("content", "")[:120] + "..."),
            "char_count": len(s.get("content", "")),
            "metadata": s.get("metadata", {})
        }
        for i, s in enumerate(sources)
    ]

    telemetry = {
        "query": query,
        "total_time_ms": total_time_ms,
        "retrieval_time_ms": sum(s["duration_ms"] for s in state["timing_steps"] if "RAG" in s["name"] or "Web" in s["name"]),
        "embedding_time_ms": 0.0,
        "llm_time_ms": llm_duration_ms,
        "ttft_ms": ttft_ms,
        "tokens_count": token_count,
        "tokens_per_second": tps,
        "timing_steps": state["timing_steps"],
        "bottlenecks": state["bottlenecks"],
        "model_used": f"{active_model} [LangGraph Agent]",
        "used_fallback": False,
        "fallback_reason": "",
        "prompt_preview": prompt[:400] + "..." if len(prompt) > 400 else prompt,
        "chunks_retrieved_count": len(sources),
        "raw_chunks": raw_chunks_formatted
    }

    yield {"type": "dev_telemetry", "telemetry": telemetry}
    yield {"type": "done"}
