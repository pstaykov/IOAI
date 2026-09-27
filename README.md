# IOAI
practice material for the IOAI

## Learning system

Ported from [amosblomqvist/learn](https://github.com/amosblomqvist/learn) (a pi config) to Claude Code:

- `/teach` (`.claude/skills/teach`) - probe level, plan a dependency map, then teach node by node with graded quizzes
- `/visualize` (`.claude/skills/visualize`) - minimal mermaid/SVG diagrams for lessons
- `researcher` agent (`.claude/agents/researcher.md`) - verifies facts before they are taught

Start a session with e.g. `/teach backpropagation`. The pi-specific extensions (quiz, ask-user-question, md-log, visual-tools) are replaced by `AskUserQuestion` and markdown notes in `notes/`.
