from agent import Agent

BANNER = """Tool-Calling AI Agent
Tools: web_search | calculator | read_file
Commands: 'exit' to quit, 'reset' to clear memory
"""


def main():
    print(BANNER)
    agent = Agent()
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            agent = Agent()
            print("Memory cleared.\n")
            continue
        try:
            agent.run(question)
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
