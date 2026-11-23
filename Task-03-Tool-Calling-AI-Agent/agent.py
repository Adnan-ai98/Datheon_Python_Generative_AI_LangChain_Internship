from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from tools import ALL_TOOLS

# Terminal colors (ANSI)
YELLOW, CYAN, MAGENTA, GREEN, RED, BOLD, RESET = (
    "\033[93m", "\033[96m", "\033[95m", "\033[92m", "\033[91m", "\033[1m", "\033[0m"
)


def _text(content) -> str:
    """Gemini kabhi content ko blocks ki list bana deta hai. Sirf asli text lo,
    thinking blocks nahi."""
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") not in ("thinking", "reasoning"):
                parts.append(p.get("text", ""))
        return " ".join(x for x in parts if x)
    return content or ""


def _thinking(content) -> str:
    """Gemini ki thinking summary (agar on ho) nikalo."""
    if not isinstance(content, list):
        return ""
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") in ("thinking", "reasoning"):
            parts.append(p.get("thinking") or p.get("reasoning") or p.get("text") or "")
    return " ".join(x for x in parts if x).strip()


class ReasoningPrinter(BaseCallbackHandler):
    """Thought -> Action -> Observation chain console par dikhata hai."""

    def __init__(self):
        self.step = 0
        self._last_thought = None

    def on_agent_action(self, action, **kwargs):
        thought = ""
        for msg in getattr(action, "message_log", []) or []:
            content = getattr(msg, "content", "")
            thought = _text(content).strip() or _thinking(content)
        if thought and thought != self._last_thought:
            self.step += 1
            if len(thought) > 400:
                thought = thought[:400] + "..."
            print(f"\n{BOLD}{YELLOW}Thought {self.step}:{RESET} {thought}")
            self._last_thought = thought
        print(f"{BOLD}{CYAN}Action:{RESET} {action.tool}({action.tool_input})")

    def on_tool_end(self, output, **kwargs):
        text = str(getattr(output, "content", output))
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"{BOLD}{MAGENTA}Observation:{RESET} {text}")

    def on_tool_error(self, error, **kwargs):
        print(f"{BOLD}{RED}Tool error:{RESET} {error}")


class Agent:
    def __init__(self):
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY missing. .env file check karein.")

        # Gemini 3 ke liye temperature default hi rakhna recommended hai
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            include_thoughts=config.SHOW_THOUGHTS,  # reasoning summary for Thought lines
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)  # tools bind
        self.executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            max_iterations=config.MAX_STEPS,
            handle_parsing_errors=True,
        )
        self.history = []

    def run(self, question: str) -> str:
        result = self.executor.invoke(
            {"input": question, "chat_history": self.history},
            config={"callbacks": [ReasoningPrinter()]},
        )
        answer = _text(result["output"])
        print(f"\n{BOLD}{GREEN}Final Answer:{RESET} {answer}\n")
        self.history += [HumanMessage(content=question), AIMessage(content=answer)]
        return answer
