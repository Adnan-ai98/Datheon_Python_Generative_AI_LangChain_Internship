# 🤖 Streamlit Tool-Calling AI Agent

A responsive web chat app built with **Streamlit** and **LangChain** that wraps a tool-calling AI agent powered by **Google Gemini**. It streams the final answer token by token, keeps the chat history, and shows the agent's reasoning (Thought → Action → Observation) in real time.

## Features

- **Real-time streaming**: the final answer appears token by token with a typing cursor.
- **Tool calling**: the agent decides when to use a tool:
  - `web_search`: real-time web search (news, facts, current data)
  - `calculator`: exact math evaluation
  - `read_file`: reads local files from the `data/` folder
- **Visible reasoning**: every step (Thought, Action, Observation) is shown in a collapsible "Agent steps" panel.
- **Chat history**: conversation is kept in `st.session_state`, so the agent remembers earlier messages.
- **Session management**: a "Clear chat" button resets the conversation and the agent.
- **Resilient search**: multiple search backends with automatic fallback and query shortening.
- **Optional model fallback**: configure backup models in `.env`.

## Tech Stack

- Python 3
- Streamlit
- LangChain (`langchain`, `langchain-classic`, `langchain-core`, `langchain-community`)
- Google Gemini via `langchain-google-genai`
- DDGS (multi-backend web search)
- python-dotenv

## Project Structure

```
streamlit-ai-agent/
├── data/
│   └── sample.txt          # Sample file for the read_file tool
├── tools/
│   ├── __init__.py         # Exports ALL_TOOLS
│   ├── calculator.py       # Math tool
│   ├── file_reader.py      # Local file reader tool
│   └── web_search.py       # Web search tool
├── agent.py                # Agent class (CLI run + streaming)
├── app.py                  # Streamlit web UI
├── config.py               # Settings, models, system prompt
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the project

```bash
git clone <your-repo-url>
cd streamlit-ai-agent
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux: `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add your Google API key (get one from [Google AI Studio](https://aistudio.google.com/apikey)):

```env
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
SHOW_THOUGHTS=true
# Optional: comma separated backup models
FALLBACK_MODELS=
```

| Variable          | Description                                              | Default                 |
|-------------------|----------------------------------------------------------|-------------------------|
| `GOOGLE_API_KEY`  | Your Gemini API key (required)                           | none                    |
| `GEMINI_MODEL`    | Primary model, always tried first                        | `gemini-3.5-flash-lite` |
| `SHOW_THOUGHTS`   | Show Gemini's reasoning summary in the steps panel       | `true`                  |
| `FALLBACK_MODELS` | Backup models used if the primary fails before any output| empty                   |

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Example Questions

Try these in the chat to see each tool in action.

**Calculator**
- What is 15% of 2,480?
- Calculate (1250 * 12) / 7 and round to 2 decimals.
- If I invest 5000 at 8% yearly compound interest for 10 years, what is the final amount?

**File reader** (uses `data/sample.txt`)
- Read sample.txt and summarize it in 3 bullet points.
- What are the main topics mentioned in sample.txt?
- How many words are in sample.txt?

**Web search**
- Who is the CEO of Anthropic?
- What is the latest news about artificial intelligence?
- What is the current USD to PKR exchange rate?

**Multi-step (several tools together)**
- Read sample.txt, then calculate how many characters its first line has.
- Find the current USD to PKR rate and convert 250 USD to PKR.
- Search for the current price of 1 Bitcoin in USD and calculate the value of 0.35 Bitcoin.

**Conversation memory**
- Ask: "Who is the CEO of Anthropic?" then follow up with "Which company did he work at before?"
- Ask: "What is 45 * 12?" then follow up with "Now divide that by 6."

**Edge cases**
- Ask for a file that does not exist: "Read missing.txt".
- Ask something that needs no tool: "Explain what an AI agent is in two sentences."
- Disconnect the internet and ask a current-events question to see the graceful "search unavailable" message.

## How It Works

1. **UI (`app.py`)**: uses `st.chat_message` and `st.chat_input`. Messages are stored in `st.session_state.messages`, and the `Agent` object is stored in `st.session_state.agent`, so both survive Streamlit reruns.
2. **Agent (`agent.py`)**: a LangChain `AgentExecutor` with a tool-calling agent. The `stream()` method uses `astream_events` and yields tokens, thinking, tool calls and tool results to the UI.
3. **Streaming**: text written before a tool call is shown as a "Thought" inside the status box. Only the final answer is streamed into the chat bubble.
4. **Memory**: after each turn, the question and answer are appended to `agent.history`, which is passed back as `chat_history` on the next turn.
5. **Config (`config.py`)**: model names, step limit and the system prompt live in one place.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GOOGLE_API_KEY is missing` | Create `.env` from `.env.example` and add your key. |
| Model returns 404 | Check the exact model ID in Google AI Studio and set `GEMINI_MODEL` in `.env`. |
| `SEARCH_UNAVAILABLE` | Network or DNS problem. Try a VPN, change DNS to `8.8.8.8` / `1.1.1.1`, and run `pip install -U ddgs`. |
| Search returns no results | Rephrase with simpler keywords. |
| Changes not showing | Stop Streamlit (Ctrl+C) and run `streamlit run app.py` again. |

## Security

- Never commit your `.env` file. It is listed in `.gitignore`.
- If a key was ever pushed to GitHub, regenerate it in Google AI Studio.

## License

MIT (or your preferred license).