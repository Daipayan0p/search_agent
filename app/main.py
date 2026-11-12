
from uuid import uuid4

from app.agent import SearchMindAgent


def main():
    session_id = "default"
    agent = SearchMindAgent(session_id=session_id)

    print("\nWelcome to SearchMind RAG Agent!")
    print("Commands:")
    print("  history - View conversation history")
    print("  new     - Start a new conversation")
    print("  backfill - Index up to 100 pending SQLite messages (safe to repeat)")
    print("  clear-cache - Invalidate all cached answers")
    print("  exit    - Quit\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not question:
                continue

            command = question.lower()

            if command in ("exit", "quit"):
                print("Goodbye!")
                break

            if command == "history":
                for message in agent.history():
                    print(
                        f"[{message['timestamp']}] "
                        f"{message['role']}: "
                        f"{message['content']}"
                    )
                continue

            if command == "new":
                agent.session_id = str(uuid4())
                print(
                    "\nStarted new conversation: "
                    f"{agent.session_id}\n"
                )
                continue

            if command in {"backfill", "clear-cache"}:
                try:
                    if command == "backfill":
                        print(agent.backfill())
                    else:
                        agent.cache.invalidate()
                        print("Answer cache cleared.")
                except Exception:
                    print("Maintenance failed; SQLite records are preserved.")
                continue

            try:
                answer = agent.ask(question)
                print(f"\nSearchMind: {answer}\n")

            except Exception:
                # SDK exception messages can contain credentials or request bodies.
                print("\nRequest failed. Check API configuration and service availability.\n")

    finally:
        agent.close()


if __name__ == "__main__":
    main()
