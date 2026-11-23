import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
from agent import Agent

st.set_page_config(page_title="AI Research Agent", page_icon="🤖")
st.title("🤖 Tool-Calling AI Agent")

ss = st.session_state


# ---------- Session state ----------
def make_agent(idx: int, history=None) -> Agent:
    agent = Agent(config.MODELS[idx])
    agent.history = list(history or [])
    return agent


if "messages" not in ss:
    ss.messages = []          # [{"role", "content", "steps"}]
if "model_idx" not in ss:
    ss.model_idx = 0
if "agent" not in ss:
    try:
        ss.agent = make_agent(0)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: `{config.MODELS[ss.model_idx]}`")
    if st.button("🗑️ Clear chat"):
        ss.messages = []
        ss.model_idx = 0
        ss.agent = make_agent(0)
        st.rerun()

# ---------- Chat history ----------
for m in ss.messages:
    with st.chat_message(m["role"]):
        if m.get("steps"):
            with st.expander("🔍 Agent steps"):
                for s in m["steps"]:
                    st.markdown(s)
        st.markdown(m["content"])


# ---------- Run one turn ----------
def run_turn(prompt: str):
    """Returns (answer, steps). If an error happens before any output and a
    fallback model is configured, it retries with the next model."""
    while True:
        status = st.status("Thinking...", expanded=True)
        placeholder = st.empty()
        steps, buf, think, got_output = [], "", "", False

        def add(text):
            steps.append(text)
            status.markdown(text)

        def flush_thought():
            nonlocal buf, think
            if think:
                add(f"💭 *{think.strip()[:400]}*")
                think = ""
            if buf.strip():
                add(f"**Thought:** {buf.strip()}")
            buf = ""
            placeholder.empty()

        try:
            for kind, data in ss.agent.stream(prompt):
                got_output = True
                if kind == "token":
                    buf += data
                    placeholder.markdown(buf + "▌")
                elif kind == "thinking":
                    think += data
                elif kind == "tool_start":
                    flush_thought()
                    name, args = data
                    add(f"**Action:** `{name}({args})`")
                elif kind == "tool_end":
                    text = data if len(data) <= 500 else data[:500] + "..."
                    add(f"**Observation:** {text}")
            placeholder.markdown(buf)
            status.update(label="Done", state="complete", expanded=False)
            return buf, steps
        except Exception as e:
            status.update(label="Error", state="error")
            last = ss.model_idx >= len(config.MODELS) - 1
            if got_output or last:
                raise
            ss.model_idx += 1
            ss.agent = make_agent(ss.model_idx, ss.agent.history)
            st.warning(
                f"Model failed ({e}). Now trying `{config.MODELS[ss.model_idx]}`."
            )


# ---------- Chat input ----------
if prompt := st.chat_input("Ask me anything..."):
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer, steps = run_turn(prompt)
        except Exception as e:
            answer, steps = f"⚠️ Error: {e}", []
            st.error(answer)

    ss.agent.history += [HumanMessage(content=prompt), AIMessage(content=answer)]
    ss.messages.append({"role": "assistant", "content": answer, "steps": steps})