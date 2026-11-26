# Copilot Workspace Instructions

## 1. Strict Scope & Directory Boundaries
- **Target Directory Only:** Modify ONLY files inside the specific folder(s) or file(s) explicitly named in the user's prompt.
- **Read-Only Outside Scope:** You may read files outside the target directory for context, types, or references, but you MUST NOT modify, create, move, or delete them unless explicitly instructed.
- **Ask Before Expanding Scope:** If a change seems to require editing a file outside the specified folder, STOP and ask the user for confirmation before making any edits there.

## 2. Minimal & Targeted Edits
- **No Unsolicited Refactoring:** Do not refactor, reformat, rename variables/functions, or reorganize code outside the exact lines or logic needed for the requested task.
- **Preserve Existing Behavior:** Keep existing functions, interfaces, and unrelated features intact unless asked to change them.
- **Avoid Over-Engineering:** Keep solutions straightforward and scoped to the request. Do not introduce extra abstractions, helper layers, or new dependencies unless necessary and approved.

## 3. Verification & Reporting
- **Summarize Touched Files:** At the end of a task, list every file you created, modified, or deleted, along with a brief note of what changed.
- **Flag Cross-Folder Impacts:** If any shared dependency or caller in another folder might be affected by the change, mention it as a note rather than silently editing those files.