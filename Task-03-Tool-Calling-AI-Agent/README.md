Tool-Calling AI Agent

A console AI agent built with LangChain and Google Gemini that autonomously decides which tool to use to solve compound research questions. It shows its full reasoning chain in the terminal: Thought → Action → Observation → Final Answer.

Features
Autonomous tool selection: the LLM decides which tool to call, in what order, and when it has enough information to answer.
Three tools:
web_search: real-time web search via DuckDuckGo (ddgs) with automatic fallback across multiple search backends.
calculator: safe math evaluator built on Python's ast module (no eval()), so the model never does arithmetic in its head.
read_file: reads local text files from the data/ folder only (path traversal is blocked).
Visible reasoning chain: every step is printed with colors in the console.
Conversation memory: follow-up questions such as "what is 10% of that amount?" work.
Configurable: model name and keys come from .env.
How it works
        You (console)
             |
          main.py
             |
   AgentExecutor (agent.py)  <---->  Gemini (tools bound to the LLM)
             |
     +-------+---------+
     |       |         |
 web_search calculator read_file
     |       |         |
     +-------+---------+
             |
   Observation -> back to the LLM -> next step or Final Answer
The question and chat history go to Gemini together with the tool definitions.
If Gemini requests a tool, the agent runs it and returns the result as an Observation.
The loop repeats (up to MAX_STEPS) until Gemini answers without a tool call. That answer is the Final Answer.
Project structure
tool-calling-ai-agent/
├── data/
│   └── sample.txt          # sample file the agent can read
├── tools/
│   ├── __init__.py         # tool registry
│   ├── calculator.py       # safe math tool
│   ├── file_reader.py      # local file reader (data/ only)
│   └── web_search.py       # DuckDuckGo search with backend fallback
├── agent.py                # LangChain agent + reasoning printer
├── config.py               # settings and system prompt
├── main.py                 # console entry point
├── requirements.txt
├── .env.example
└── README.md
Tech stack
Python 3.10+
LangChain (langchain, langchain-classic, langchain-community)
langchain-google-genai (Gemini)
ddgs (DuckDuckGo search)
python-dotenv
Setup
bash
# 1. Clone
git clone https://github.com/<your-username>/tool-calling-ai-agent.git
cd tool-calling-ai-agent

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux

Open .env and add your Gemini API key (get one from Google AI Studio):

env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
Usage
bash
python main.py

Commands inside the app:

Command	Action
exit	Quit
reset	Clear conversation memory
Example queries
Read sample.txt, calculate the total revenue, and then compute 15% tax on that total.
What is the latest stable version of Python?
Search for the current price of Bitcoin, then calculate the value of 0.5 BTC in USD.
Read sample.txt, find the revenue of Product A, search the web for the current USD to PKR exchange rate, and convert that revenue into PKR.
Example run
You: Read sample.txt, find the revenue of Product A, search the web for the current
     USD to PKR exchange rate, and convert that revenue into PKR.

Action: read_file({'filename': 'sample.txt'})
Observation: Q3 2026 Sales Report
             Product A revenue: 12500 USD
             ...

Action: web_search({'query': 'USD to PKR exchange rate current'})
Observation: [1] US dollars to Pakistani rupees Exchange Rate ...
             [2] Currency Rates in Pakistan Open Market Dollar Rate - Forex.pk ...

Action: calculator({'expression': '12500 * 275.87'})
Observation: 3448375.0

Final Answer: The revenue of Product A is 12,500 USD. Based on the current exchange
rate of 1 USD to 275.87 PKR, the converted revenue is 3,448,375 PKR.

The agent chose all three tools on its own and chained them in the right order. Thought lines are printed when the model returns reasoning summaries (controlled by SHOW_THOUGHTS).

Configuration
Variable	Default	Description
GOOGLE_API_KEY	none	Gemini API key (required)
GEMINI_MODEL	gemini-3.5-flash-lite	Gemini model name
SHOW_THOUGHTS	true	Request and print Gemini's reasoning summaries as Thought lines

Other settings (MAX_STEPS, system prompt) are in config.py.

Security notes
The calculator parses expressions with ast and allows only numbers, basic operators and a whitelist of math functions. Anything else is rejected.
read_file resolves paths and refuses anything outside data/ (for example ../.env), and only opens .txt, .md, .csv, .json, .log files.
.env is listed in .gitignore. Never commit your API key.
Troubleshooting
Problem	Fix
GOOGLE_API_KEY missing	Make sure the file is named .env (not .env.example) and sits in the project root.
404 model not found	Change GEMINI_MODEL in .env to another available Gemini model.
429 quota exceeded	Free-tier limit reached. Wait a little and retry.
SEARCH_UNAVAILABLE	Your network blocks the search engines (DNS or timeout). Try a VPN or set DNS to 8.8.8.8 / 1.1.1.1.
Agent stopped due to max iterations	Ask a simpler question or increase MAX_STEPS in config.py.
Limitations
Search results are short snippets. The agent cannot open full web pages, so numbers that are not in the snippet may be missed.
Web data (prices, exchange rates) comes from search snippets and can differ slightly between sources.
read_file supports plain text formats only (no PDF or DOCX).
Future improvements
A fetch_url tool to read full web pages
PDF and DOCX support in the file reader
Tavily search as an alternative provider
Streaming output and a simple web UI
License

MIT