---
name: laya-decision
description: Fast (sub-200ms) quantized neural decision classifier and security routing gate for Windows laptop automation and maintenance. Use when you need to categorize a user request into a safe, bounded action or verify safety thresholds before executing terminal commands.
---

# Laya-Decision Skill

A local, low-latency (140–200 ms) decision classifier powered by an INT8 quantized ModernBERT ONNX model in pure Rust (`laya-siro.exe`).

## Role in Agent Workflows

Use `laya-decision` to classify user automation requests into bounded, allowlisted action labels before running terminal commands. Laya-Siro does **not** generate free-form text or unrestricted shell code; it acts as a typed decision and safety guardrail engine.

### Available Action Targets

| Action Label | Type | Description | Guardrail Policy |
| :--- | :--- | :--- | :--- |
| `system_triage` | Read-only | Top RAM WorkingSet process query | Executes directly without confirmation |
| `laravel_test` | Read-only | `php artisan test` backend suite | Executes directly without confirmation |
| `laravel_route_list` | Read-only | `php artisan route:list` inspection | Executes directly without confirmation |
| `edge_engine_check` | Read-only | `cargo check -j 2` in `clients/edge-engine-rs` | Executes directly without confirmation |
| `git_diff` | Read-only | `git diff --stat` working tree summary | Executes directly without confirmation |
| `git_status` | Read-only | `git status --short` in current workspace | Executes directly without confirmation |
| `git_log` | Read-only | `git log -n 5 --oneline` commit history | Executes directly without confirmation |
| `git_stash_list` | Read-only | `git stash list` stashed changes query | Executes directly without confirmation |
| `git_pull_ff_only` | Mutating | `git pull --ff-only` | Refuses dirty tree; dry-run preview default; requires `--execute` + `[y/N]` |
| `npm_build` | Mutating | `npm run build` frontend asset compilation | Dry-run preview default; requires `--execute` + `[y/N]` |
| `laravel_clear_cache` | Mutating | `php artisan optimize:clear` | Dry-run preview default; requires `--execute` + `[y/N]` |
| `clean_temp` | Mutating | Inspects or clears `AppData\Local\Temp` | Dry-run preview default; requires `--execute` + `[y/N]` |
| `organize_downloads` | Mutating | Categorizes files in `Downloads/` | Safeguard locked to preview |
| `block_unsafe` / `hard_blocked` | Safety | Flags destructive / prohibited commands (`migrate`, `db:seed`, `db:wipe`, `key:generate`, `app:prune`, `sync_to_server`) | Permanently blocked; zero execution |

---

## How to Invoke via CLI or Script

### Option 1: PowerShell Wrapper (Recommended for Agents)

```powershell
powershell -ExecutionPolicy Bypass -File .agents/skills/laya-decision/scripts/siro_decide.ps1 -Prompt "find which processes are hogging all my RAM"
```

### Option 2: Direct Binary Execution (`--json`)

```powershell
target\release\laya-siro.exe --json "find which processes are hogging all my RAM"
```

Example JSON Response:

```json
{
  "action": "system_triage",
  "confidence": 0.8863,
  "escalate": false,
  "mutating": false,
  "latency_ms": 134.35,
  "routing_source": "NEURAL_MODEL",
  "status": "SUCCESS",
  "error": null,
  "should_ask_user": false
}
```

---

## Mandatory Safety Rules for AI Agents

1. **Check `should_ask_user`:**
   If `should_ask_user: true` (or `confidence < 0.75`, or `escalate: true`), the agent **MUST NOT** proceed automatically. You must ask the user for explicit confirmation before taking any action.
2. **Mutating Action Safeguard:**
   If `mutating: true`, the default mode is dry-run preview only. Never invoke with `--execute` unless the user explicitly requested live changes.
3. **Hard-Blocked Security Boundary:**
   If `action: "hard_blocked"`, the request matched a prohibited policy pattern (`migrate:fresh`, `git push --force`, `system32`, etc.). Never attempt alternative workarounds to execute these commands.
