"""ReAct-style agent loop.

This module is deliberately framework-light: tools are simple objects exposing
``run(input: str) -> str`` and the LLM client only needs a ``complete`` method.
That keeps the orchestration transparent and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Protocol

from research_agent.config import settings


class Tool(Protocol):
    """Minimal tool interface used by the agent."""

    name: str
    description: str

    def run(self, input: str) -> str:
        """Execute the tool and return an observation string."""


class LLMClient(Protocol):
    """Small completion interface to avoid binding the agent to a framework."""

    def complete(self, prompt: str) -> str:
        """Return the next reasoning/action message."""


@dataclass
class AgentStep:
    """Trace entry for a single ReAct iteration."""

    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    """Final agent response plus full reasoning trace."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)


class StubLLMClient:
    """Deterministic fallback LLM for local smoke tests.

    It is not a substitute for a production LLM. The class exists so the API and
    tests can run without credentials, while making the limitation explicit.
    """

    def complete(self, prompt: str) -> str:
        if "Observation:" in prompt:
            return (
                "Thought: I have enough retrieved context to answer.\n"
                "Final Answer: I completed the available research step. "
                "Configure LLM_PROVIDER=openai for deeper multi-step reasoning."
            )
        return (
            "Thought: I should search the local paper index first.\n"
            'Action: rag_retrieve\nAction Input: {"query": "latest relevant papers"}'
        )


class OpenAILLMClient:
    """OpenAI-compatible chat completion adapter.

    The import is lazy so local users can run non-LLM parts of the project
    without installing the OpenAI SDK. Providers such as AIHubmix can be used
    by setting OPENAI_BASE_URL in the environment.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.openai_model

    def complete(self, prompt: str) -> str:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        try:
            from openai import APIConnectionError, APITimeoutError, OpenAI  # type: ignore
            from openai import AuthenticationError, BadRequestError, RateLimitError
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for LLM_PROVIDER=openai. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise research assistant using ReAct. "
                            "Return either Thought/Action/Action Input or Final Answer."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        except AuthenticationError as exc:
            raise RuntimeError("LLM authentication failed. Check OPENAI_API_KEY in .env.") from exc
        except BadRequestError as exc:
            raise RuntimeError(
                f"LLM request was rejected. Check MODEL_NAME/OPENAI_MODEL='{self.model}' "
                f"and provider compatibility. Provider said: {exc.message}"
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError("LLM provider rate limit reached. Retry later or change model/key.") from exc
        except APITimeoutError as exc:
            raise RuntimeError(
                f"LLM request timed out after {settings.request_timeout_seconds}s. "
                "Increase REQUEST_TIMEOUT_SECONDS or check provider availability."
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                f"LLM connection failed. Check OPENAI_BASE_URL='{settings.openai_base_url}', "
                "local proxy/VPN/firewall settings, and whether the provider endpoint is reachable."
            ) from exc
        return response.choices[0].message.content or ""


def build_default_llm() -> LLMClient:
    """Factory for the configured LLM backend."""

    if settings.llm_provider.lower() == "openai":
        return OpenAILLMClient()
    return StubLLMClient()


class ReActAgent:
    """A transparent ReAct implementation with history tracking."""

    FINAL_RE = re.compile(r"Final Answer\s*:\s*(?P<answer>.*)", re.I | re.S)
    ACTION_RE = re.compile(
        r"Action\s*:\s*(?P<action>[A-Za-z0-9_\-]+)\s*"
        r"Action Input\s*:\s*(?P<input>.*)",
        re.I | re.S,
    )
    THOUGHT_RE = re.compile(r"Thought\s*:\s*(?P<thought>.*?)(?:\nAction|\nFinal Answer|$)", re.I | re.S)

    def __init__(
        self,
        tools: list[Tool],
        llm: LLMClient | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm or build_default_llm()
        self.max_steps = max_steps or settings.agent_max_steps
        self.history: list[AgentStep] = []

    def run(self, query: str) -> AgentResult:
        """Run the ReAct loop until a final answer or max step limit."""

        self.history = []
        scratchpad = ""

        for _ in range(self.max_steps):
            prompt = self._build_prompt(query=query, scratchpad=scratchpad)
            llm_output = self.llm.complete(prompt).strip()

            final = self._parse_final(llm_output)
            if final is not None:
                thought = self._parse_thought(llm_output)
                if thought:
                    self.history.append(AgentStep(thought=thought))
                return AgentResult(answer=final, steps=self.history)

            thought = self._parse_thought(llm_output) or "No explicit thought provided."
            action, action_input = self._parse_action(llm_output)
            if action is None:
                observation = "Invalid agent output: expected Action or Final Answer."
                self.history.append(AgentStep(thought=thought, observation=observation))
                scratchpad += self._format_trace(thought, None, None, observation)
                continue

            observation = self._execute_tool(action, action_input)
            step = AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            self.history.append(step)
            scratchpad += self._format_trace(thought, action, action_input, observation)

        return AgentResult(
            answer="Reached maximum reasoning steps before producing a final answer.",
            steps=self.history,
        )

    def _build_prompt(self, query: str, scratchpad: str) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        )
        return f"""Answer the user query using the available tools.

Available tools:
{tool_descriptions}

Use exactly one of these formats:
Thought: explain the next step
Action: tool_name
Action Input: input string or JSON string

Thought: explain why you are done
Final Answer: final response to the user

User Query:
{query}

Previous steps:
{scratchpad}
"""

    def _execute_tool(self, action: str, action_input: str) -> str:
        tool = self.tools.get(action)
        if tool is None:
            return f"Unknown tool '{action}'. Available tools: {', '.join(self.tools)}"
        try:
            return tool.run(action_input)
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            return f"Tool '{action}' failed: {exc}"

    def _parse_final(self, text: str) -> str | None:
        match = self.FINAL_RE.search(text)
        return match.group("answer").strip() if match else None

    def _parse_action(self, text: str) -> tuple[str | None, str]:
        match = self.ACTION_RE.search(text)
        if not match:
            return None, ""
        raw_input = match.group("input").strip()
        return match.group("action").strip(), self._normalize_action_input(raw_input)

    def _parse_thought(self, text: str) -> str | None:
        match = self.THOUGHT_RE.search(text)
        return match.group("thought").strip() if match else None

    def _normalize_action_input(self, value: str) -> str:
        """Compact JSON action inputs while preserving plain strings."""

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return json.dumps(parsed, ensure_ascii=False)

    def _format_trace(
        self,
        thought: str,
        action: str | None,
        action_input: str | None,
        observation: str,
    ) -> str:
        trace = f"Thought: {thought}\n"
        if action:
            trace += f"Action: {action}\nAction Input: {action_input}\n"
        trace += f"Observation: {observation}\n\n"
        return trace
