# AGENTS Instructions

## Behavior & Execution Protocol
- **Autonomous Decision Making**: Never interrupt the user with interactive approval prompts (`ask_question` or similar modals) for standard code generation, edits, file creation, or testing. Always make confident, production-grade technical decisions independently.
- **Verification**: Always execute automated tests and verification steps to validate changes before reporting completion.
- **Documentation**: Provide clear, concise, GitHub-formatted markdown with clickable links (`file:///...`) for all modified files.
