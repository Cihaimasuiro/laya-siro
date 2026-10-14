# Laya-Siro: Windows Low-Memory Decision & Automation Engine

A high-performance, low-memory Windows decision and automation engine running [Laya](https://huggingface.co/convaiinnovations/laya) in Rust via [ONNX Runtime](https://onnxruntime.ai/).

Specifically optimized for **8 GB RAM / Intel 11th Gen Core i5 (i5-1155G7)** environments with tight memory budgets.

---

## Architecture & Memory Guardrails

1. **Pattern B (Short-Lived CLI)**:
   - Evaluates state and routes decisions in **~167 ms** (down from 5.8s via sequence slicing).
   - Executes the targeted Windows automation action natively.
   - Exits immediately, leaving **0 MB of idle background RAM**.
2. **Sub-200ms Engine Optimizations**:
   - `with_intra_threads(4)`: Fully utilizes all 4 physical Tiger Lake cores on Intel Core i5-1155G7.
   - `GraphOptimizationLevel::Level3`: Fuses quantized INT8 attention nodes into AVX-512 VNNI vector instructions.
   - Sequence Length Slicing (`model_fast.onnx`): Eliminates $O(N^2)$ quadratic attention over-padding (128 vs 1024), cutting latency by 35x.
   - Managed memory arena: Prevents OS-level `VirtualAlloc` fragmentation during tensor execution.
3. **Exact 5-Tensor Graph Contract**:
   - Accurately constructs: `input_ids`, `attention_mask`, `marker_pos` (index of each option's `[MASK]`), `marker_mask` (boolean mask), and `qtype` (`0` for Choice).
   - Extracts both `logits` (option probability distribution scaled by RLCD temperature) and `act_logits` (2-way answer/abstain gate).
4. **Dynamic Runtime Linking (`load-dynamic`)**:
   - Bypasses MSVC static linker constraints, allowing clean compilation on Windows GNU/MinGW and MSVC toolchains alike.

---

## Quick Start

### 1. Download Model & Dependencies (Step 2)

Run the included PowerShell helper script to download the INT8 quantized model (~317 MB), tokenizer, and configuration from Hugging Face:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
```

### 2. Run the Automation CLI

Run the CLI with any plain English task. When weights are downloaded, Laya routes with 5-tensor calibrated inference; before weights are downloaded, an intelligent heuristic fallback allows testing immediately:

```bash
# 1. System & RAM Triage
cargo run -- "find which processes are hogging all my RAM"

# 2. Downloads Folder Organization
cargo run -- "tidy up my downloads folder by sorting files"

# 3. Temporary File Cache Inspection
cargo run -- "delete temporary files in Windows cache"

# 4. Guardrail Safety Abort
cargo run -- "delete system32 and wipe C: drive"
```

### 3. Global Windows PowerShell Shortcut (`siro`)

Install the global `siro` command into your PowerShell profile to trigger automation tasks from any folder on your laptop:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_shortcut.ps1
```

Once installed, invoke it anywhere:

```powershell
siro "delete temporary files in Windows cache"
siro "find which processes are hogging all my RAM"
```

### 4. Safe Compilation Flag

To ensure compiler operations do not exceed free memory:

```bash
cargo check -j 2
cargo build -j 2
```
