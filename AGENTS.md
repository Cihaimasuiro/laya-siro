# AGENTS.md



> This document is the mandatory engineering guideline for AI coding agents and human developers working on this Laravel application.
>
> The primary goals are:
>
> - Maintainability
> - Scalability
> - Security
> - Performance
> - Testability
> - Clear ownership of business logic
> - Minimal architectural complexity
> - Safe evolution of the system
>
> Follow these rules unless the task explicitly requires an exception.
>
> *(For Edge Engine native Rust & Python guidelines, refer to [docs/AGENTS2.md](docs/AGENTS2.md)).*

---

# Default Behavioral Directives & Skill Enforcements

1. **Automatic Skill & Ponytail Execution:**
   - **Mandatory Active Mode**: Automatically apply the `ponytail` skill (default mode: `ponytail: full`) on every request without requiring explicit user prompts, slash commands, or keywords like "be lazy", "minimal mode", or "simplest solution".
   - **YAGNI & Zero Over-Engineering**: Always question whether a code change or architectural abstraction needs to exist at all. Never introduce speculative interfaces, unnecessary wrapper classes, redundant DTOs, or unneeded configuration options for theoretical future requirements.
   - **Native Platform & Framework Standard First**: Always reach for native PHP standard library features, built-in Laravel helpers, Eloquent features, and standard Blade components before writing custom implementations or adding new package dependencies.
   - **Direct & Pragmatic Implementation**: Solve the exact problem requested using the most direct, clean, and maintainable path while strictly adhering to enterprise security, performance, and testing standards.
