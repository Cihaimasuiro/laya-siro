# AGENTS.md

> Mandatory engineering guidelines for AI coding agents and human developers working on **Laya-Siro** (Windows Low-Memory AI Decision & Automation Engine).

---

## 1. Project Overview & Architecture

* **Engine:** Pure Rust binary (`laya-siro.exe`) running local neural inference on an INT8 quantized ModernBERT ONNX model via Microsoft ONNX Runtime (`ort`).
* **Environment Constraints:** Strictly optimized for **8 GB RAM / Intel 11th Gen Core i5 (i5-1155G7)** running Windows 11.
* **Toolchain:** `stable-x86_64-pc-windows-gnu` / MinGW with dynamic loading (`load-dynamic`) of `onnxruntime.dll`.
* **Resource Ceiling:** All compilations must use `-j 2` (`cargo check -j 2`, `cargo build --release -j 2`) to prevent RAM exhaustion and system lockups while IDE/Chrome is open.

---

## 2. Safety & Guardrail Invariants (Non-Negotiable)

1. **Dry-Run by Default:**
   * Any action that alters filesystem or system state (`clean_temp`, `organize_downloads`) MUST run in read-only / preview mode by default.
   * Mutating execution is only permitted when `--execute` is explicitly passed.

2. **Interactive Human Confirmation:**
   * Passing `--execute` DOES NOT bypass human approval. The binary must prompt `[y/N]` interactively in the terminal before performing any mutation.
   * If stdin delivers anything other than `y` or `yes`, abort immediately without touching the filesystem.

3. **Fallback Routing Is Strictly Read-Only:**
   * When neural model weights are missing, the heuristic fallback router is strictly read-only.
   * The fallback path must never execute mutating actions, even if `--execute` is passed on the command line.

4. **Minimum Confidence Floor (>= 75.0%):**
   * Model decisions with confidence `< 0.75` or where the act gate triggers abstain/escalate must be aborted automatically.

5. **Security Hard-Block Policy (Absensi-Cam Data & Secret Protection):**
   * Database mutations, migrations, seeders, keys, pruning, and server syncs (`migrate`, `db:seed`, `db:wipe`, `key:generate`, `app:prune`, `app:tombstone`, `sync_to_server`, `deploy`) are permanently prohibited in code.
   * Destructive system commands (`git push --force`, `system32`, `format`, `wipe`, `rmdir /s`, `del /f /s`) are permanently prohibited.
   * Hard-blocked requests terminate immediately (0 ms) without shell invocation.

6. **Dirty-Tree Git Pull Protection:**
   * `git pull --ff-only` must check `git status --short` first and refuse execution if any modified or untracked files exist, safeguarding uncommitted face template or attendance code.

7. **Audit Trail:**
   * Every routing attempt, confidence score, routing source, mode, and execution outcome must be logged to `logs/automation.log`.
   * `logs/` must remain local and ignored in `.gitignore`.

---

## 3. Machine-Readable Agent Integration

* When AI agents (such as Antigravity) invoke the engine for decision routing, they must pass `--json` to receive structured JSON:
  ```json
  {
    "action": "system_triage",
    "confidence": 0.8863,
    "escalate": false,
    "mutating": false,
    "latency_ms": 134.35,
    "routing_source": "NEURAL_MODEL",
    "status": "SUCCESS",
    "error": null
  }
  ```
* Agents must honor the `should_ask_user` flag emitted by `.agents/skills/laya-decision/scripts/siro_decide.ps1` before taking any automated actions.
