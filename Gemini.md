# Python Coding Standards

## 1. Pythonic Principles
- Write code that prioritizes **readability over cleverness**.
- Prefer clear, explicit logic over dense one-liners.
- Optimize for maintainability and ease of review.

---

## 2. PEP 8 Compliance
- Use **4 spaces** for indentation (never tabs).
- Limit lines to **79 characters** (88 if auto-formatted).
- Separate logical sections with blank lines.
- One import per line; group standard, third-party, and local imports.
- Naming conventions:
  - `snake_case` for variables and functions
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

---

## 3. Naming Conventions
- Use **descriptive, intention-revealing names**.
- Avoid vague identifiers (`data`, `temp`, `value`, `obj`).
- Boolean names must read clearly (`is_active`, `has_permission`).
- Avoid single-letter names outside of short loops.

---

## 4. Built-ins and Standard Library
- Prefer Python built-ins over custom implementations.
- Use the standard library (`itertools`, `collections`, `functools`, `pathlib`, etc.) whenever possible.
- Do not reimplement common patterns (sorting, filtering, grouping).

---

## 5. Comprehensions
- Use list, dict, and set comprehensions when intent is clear.
- Avoid deeply nested comprehensions.
- If comprehension reduces readability, use explicit loops.

---

## 6. Function Design
- Functions must perform **one logical task**.
- Keep functions short and focused.
- Avoid excessive parameters; prefer keyword arguments.
- Favor pure functions without side effects.

---

## 7. Explicit Over Implicit
- Avoid magic values; define named constants.
- Make assumptions visible in code.
- Prefer explicit control flow over hidden behavior.

---

## 8. Error Handling
- Catch **specific exceptions only**.
- Never use bare `except`.
- Do not suppress exceptions without justification.
- Error messages must be meaningful and actionable.

---

## 9. Type Hints
- Use type hints for all public functions and methods.
- Be precise and intentional with types.
- Use `Optional`, `Union`, and generics appropriately.
- Type hints are part of the documentation contract.

---

## 10. Docstrings
- Add docstrings to all public modules, classes, and functions.
- Explain **intent, behavior, and edge cases**, not obvious logic.
- Keep docstrings concise and relevant.

---

## 11. State Management
- Avoid global mutable state.
- Pass dependencies explicitly via arguments.
- Encapsulate state within classes when necessary.
- Globals are allowed only for constants.

---

## 12. Consistency
- Follow existing project patterns and conventions.
- Avoid introducing new styles without strong justification.
- Consistency across the codebase is mandatory.

---

## 13. Testability
- Code must be designed for testability.
- Avoid hidden dependencies and tight coupling.
- Separate business logic from I/O and side effects.
- Untestable code is considered incomplete.

---

## 14. Code Hygiene
- Remove unused imports, variables, and functions.
- Do not leave commented-out or dead code.
- Rely on version control for history, not comments.

---