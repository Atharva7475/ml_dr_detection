# Autonomous Execution Guidelines

## Core Principles
1. **Full Autonomy**: Make technical, architectural, and operational decisions autonomously without stopping to ask trivial questions or prompting the user for approval on small tasks.
2. **Never Use `ask_question`**: Do not use the `ask_question` tool or block interactive modals unless explicitly commanded by the user. If an ambiguity arises, choose the industry-standard or most robust technical solution, execute it, and explain the rationale in the final response.
3. **End-to-End Task Completion**: Complete multi-step tasks end-to-end, including debugging, testing, generating necessary files, and verifying correctness before concluding the turn.
4. **Clean, Non-Blocking Execution**: For background commands, handle timeouts and state gracefully.
