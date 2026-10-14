use std::env;
use std::fs::{self, OpenOptions};
use std::io::{stdin, stdout, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Array2};
use ort::{
    environment::Environment,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct RlConfig {
    #[serde(default = "default_temperature")]
    temperature_by_options: Vec<f32>,
}

fn default_temperature() -> Vec<f32> {
    vec![1.05]
}

/// Structured JSON output emitted when --json flag is passed
#[derive(Debug, Serialize)]
struct JsonDecisionResponse<'a> {
    action: &'a str,
    confidence: f32,
    escalate: bool,
    mutating: bool,
    latency_ms: f64,
    routing_source: &'static str,
    status: &'static str,
    error: Option<String>,
}

fn emit_json_and_exit<'a>(
    action: &'a str,
    confidence: f32,
    escalate: bool,
    mutating: bool,
    latency_ms: f64,
    routing_source: &'static str,
    status: &'static str,
    error: Option<String>,
) {
    let resp = JsonDecisionResponse {
        action,
        confidence,
        escalate,
        mutating,
        latency_ms,
        routing_source,
        status,
        error,
    };
    if let Ok(json_str) = serde_json::to_string_pretty(&resp) {
        println!("{}", json_str);
    }
}

/// Core neural automation options trained in Laya model
const AUTOMATION_OPTIONS: &[&str] = &[
    "system_triage",
    "organize_downloads",
    "clean_temp",
    "block_unsafe",
];

const ROUTER_QUESTION: &str = "Which Windows automation action should be executed for this request?";

/// Strict hard-coded forbidden command patterns (always blocked regardless of model confidence)
const HARD_BLOCKED_PATTERNS: &[&str] = &[
    // Data, migrations, and database state protection (strictly prohibited for absensi-cam)
    "migrate",
    "db:seed",
    "db:wipe",
    "key:generate",
    "app:prune",
    "app:tombstone",
    "app:push-model-update",
    "sync_to_server",
    "deploy",
    "drop database",
    // Git destructive operations
    "git push --force",
    "git push -f",
    "git reset --hard",
    // System wipe & destructive filesystem commands
    "system32",
    "format c:",
    "wipe c:",
    "rmdir /s",
    "del /f /s",
];

/// Minimum confidence floor required to execute any predicted action
const MINIMUM_CONFIDENCE_THRESHOLD: f32 = 0.75;

fn check_hard_blocked(prompt: &str) -> Option<&'static str> {
    let lower = prompt.to_lowercase();
    for &pattern in HARD_BLOCKED_PATTERNS {
        if lower.contains(pattern) {
            return Some(pattern);
        }
    }
    None
}

/// Deterministic fast-path allowlist for safe developer commands and strictly gated operations
fn check_allowlisted_fast_path(prompt: &str) -> Option<(&'static str, &'static str, bool)> {
    let lower = prompt.to_lowercase();
    let trimmed = lower.trim();

    // Tier 1: Laravel Read-Only Checks
    if trimmed.contains("artisan test") || trimmed == "test" || trimmed.contains("run tests") || trimmed.contains("backend tests") {
        Some(("laravel_test", "Run Laravel Backend Test Suite (php artisan test)", false))
    } else if trimmed.contains("route:list") || trimmed.contains("list routes") || trimmed.contains("show routes") {
        Some(("laravel_route_list", "Inspect Laravel Registered Routes (php artisan route:list)", false))

    // Tier 2: Edge Engine & Protocol Checks (Read-Only)
    } else if trimmed.contains("edge-engine") || trimmed.contains("check edge") || trimmed.contains("edge check") {
        Some(("edge_engine_check", "Verify Edge Engine Compilation (cargo check -j 2)", false))
    } else if trimmed.contains("protocol test") || trimmed.contains("test protocol") || trimmed.contains("packages/protocol") {
        Some(("protocol_test", "Test Terminal Protocol (cargo test in packages/protocol)", false))

    // Tier 3: Git Read-Only Checks
    } else if trimmed.contains("git diff") || trimmed.contains("diff stat") || trimmed == "diff" {
        Some(("git_diff", "Query Git Working Tree Diff Summary (git diff --stat)", false))
    } else if trimmed.contains("git status") || trimmed == "status" || trimmed.contains("repo status") || trimmed.contains("uncommitted") {
        Some(("git_status", "Query Git Working Tree Status (git status --short)", false))
    } else if trimmed.contains("git log") || trimmed.contains("commit history") || trimmed.contains("recent commits") || trimmed.contains("git history") {
        Some(("git_log", "Query Recent Git Commit Log (git log -n 5 --oneline)", false))
    } else if trimmed.contains("git stash") || trimmed.contains("stash list") || trimmed.contains("stashed changes") {
        Some(("git_stash_list", "Query Stashed Git Changes (git stash list)", false))

    // Tier 4: Mutating Actions (Strictly Gated with Dry-Run Default and [y/N] Confirmation)
    } else if trimmed.contains("git pull") || trimmed.contains("pull changes") || trimmed == "pull" {
        Some(("git_pull_ff_only", "Fast-Forward Git Pull (git pull --ff-only, dirty-tree protected)", true))
    } else if trimmed.contains("npm run build") || trimmed.contains("build assets") || trimmed.contains("build frontend") {
        Some(("npm_build", "Compile Frontend Assets (npm run build)", true))
    } else if trimmed.contains("optimize:clear") || trimmed.contains("clear cache") || trimmed.contains("artisan cache:clear") {
        Some(("laravel_clear_cache", "Clear Laravel Caches (php artisan optimize:clear)", true))
    } else {
        None
    }
}

fn is_action_mutating(action: &str) -> bool {
    match action {
        "system_triage" | "git_status" | "git_log" | "git_stash_list" | "git_diff"
        | "laravel_test" | "laravel_route_list" | "edge_engine_check" | "protocol_test"
        | "block_unsafe" => false,

        "clean_temp" | "organize_downloads" | "git_pull_ff_only" | "npm_build"
        | "laravel_clear_cache" => true,

        _ => true,
    }
}

fn confirm_action(action: &str, details: &str) -> bool {
    print!("\n[CONFIRMATION REQUIRED] Mutating action '{}' will be executed.\nTarget: {}\nProceed? [y/N]: ", action, details);
    let _ = stdout().flush();

    let mut input = String::new();
    if stdin().read_line(&mut input).is_ok() {
        let trimmed = input.trim().to_lowercase();
        trimmed == "y" || trimmed == "yes"
    } else {
        false
    }
}

fn current_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let days = now / 86400;
    let secs_of_day = now % 86400;
    let hours = secs_of_day / 3600;
    let mins = (secs_of_day % 3600) / 60;
    let secs = secs_of_day % 60;

    let mut year = 1970;
    let mut remaining_days = days;
    loop {
        let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_year = if is_leap { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_month = [
        31, if is_leap { 29 } else { 28 }, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    ];
    let mut month = 1;
    for &dim in &days_in_month {
        if remaining_days < dim {
            break;
        }
        remaining_days -= dim;
        month += 1;
    }
    let day = remaining_days + 1;
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", year, month, day, hours, mins, secs)
}

fn log_automation_event(
    models_dir: &Path,
    prompt: &str,
    action: &str,
    confidence: f32,
    routing_source: &str,
    mode: &str,
    status: &str,
) {
    let logs_dir = if let Some(parent) = models_dir.parent() {
        if parent.as_os_str().is_empty() {
            PathBuf::from("logs")
        } else {
            parent.join("logs")
        }
    } else {
        PathBuf::from("logs")
    };
    let _ = fs::create_dir_all(&logs_dir);
    let log_path = logs_dir.join("automation.log");

    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&log_path) {
        let ts = current_timestamp();
        let _ = writeln!(
            file,
            "[{}] PROMPT: {:?} | ACTION: {} | CONFIDENCE: {:.2}% | SOURCE: {} | MODE: {} | STATUS: {}",
            ts,
            prompt,
            action,
            confidence * 100.0,
            routing_source,
            mode,
            status
        );
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_usage(&args[0]);
        return Ok(());
    }

    let mut execute_mode = false;
    let mut json_mode = false;
    let mut prompt_parts = Vec::new();

    for arg in &args[1..] {
        if arg == "--execute" || arg == "-e" {
            execute_mode = true;
        } else if arg == "--json" {
            json_mode = true;
        } else {
            prompt_parts.push(arg.clone());
        }
    }
    let input_prompt = prompt_parts.join(" ");

    if input_prompt.trim().is_empty() {
        print_usage(&args[0]);
        return Ok(());
    }

    if !json_mode {
        println!("==================================================");
        println!(" [Laya-Siro] Windows Low-Memory Automation Engine ");
        println!("==================================================");
        println!("Prompt : \"{}\"", input_prompt);
        println!("Mode   : {}\n", if execute_mode { "EXECUTE (--execute active)" } else { "DRY-RUN (Preview default)" });
    }

    let models_dir = resolve_models_dir();

    // 1. Guardrail Gate 1: Strict Hard-Block Policy Check
    if let Some(pattern) = check_hard_blocked(&input_prompt) {
        log_automation_event(
            &models_dir,
            &input_prompt,
            "hard_blocked",
            1.0,
            "SECURITY_POLICY",
            if execute_mode { "EXECUTE" } else { "DRY_RUN" },
            "HARD_BLOCKED",
        );

        if json_mode {
            emit_json_and_exit(
                "hard_blocked",
                1.0,
                true,
                false,
                0.0,
                "SECURITY_POLICY",
                "HARD_BLOCKED",
                Some(format!("Prohibited command pattern detected: '{}'", pattern)),
            );
        } else {
            println!("\n[SECURITY HARD BLOCK] Prohibited command pattern detected: '{}'", pattern);
            println!("Request '{}' was permanently rejected by security policy.", input_prompt);
            println!("No changes or shell executions were performed.");
        }
        return Ok(());
    }

    // 2. Guardrail Gate 2: Deterministic Fast-Path Allowlist
    if let Some((action, desc, mutating)) = check_allowlisted_fast_path(&input_prompt) {
        log_automation_event(
            &models_dir,
            &input_prompt,
            action,
            1.0,
            "DETERMINISTIC_FAST_PATH",
            if mutating {
                if execute_mode { "EXECUTE" } else { "DRY_RUN" }
            } else {
                "READ_ONLY"
            },
            "SUCCESS",
        );

        if json_mode {
            emit_json_and_exit(
                action,
                1.0,
                false,
                mutating,
                0.0,
                "DETERMINISTIC_FAST_PATH",
                if mutating && !execute_mode { "PREVIEW_ONLY" } else { "SUCCESS" },
                None,
            );
            return Ok(());
        }

        println!(">>> Detected Allowlisted Intent: {} <<<", desc);
        println!("Action     : {}", action);
        println!("Confidence : 100.00% (Deterministic Allowlist)");
        println!("Safety     : {}\n", if mutating { "Mutating (Requires --execute and [y/N] confirmation)" } else { "Passed (Read-Only)" });

        if mutating {
            if !execute_mode {
                println!("[DRY-RUN DEFAULT] Action '{}' is mutating. Running in preview mode.", action);
                execute_windows_action(action, &input_prompt, false)?;
                println!("\n[HINT] To apply actual changes, re-run with '--execute' and confirm the prompt [y/N].");
                return Ok(());
            }

            if !confirm_action(action, desc) {
                log_automation_event(
                    &models_dir,
                    &input_prompt,
                    action,
                    1.0,
                    "DETERMINISTIC_FAST_PATH",
                    "EXECUTE",
                    "USER_CANCELLED",
                );
                println!("\n[ABORTED] Action cancelled by user. No system changes were made.");
                return Ok(());
            }
        }

        execute_windows_action(action, &input_prompt, execute_mode)?;
        return Ok(());
    }

    // 3. Neural Decision Path (Laya INT8 ONNX Model)
    let fast_model = models_dir.join("model_fast.onnx");
    let base_model = models_dir.join("model.onnx");
    let model_path = if fast_model.exists() {
        if !json_mode {
            println!("[Fast Inference Engine] Using optimized sequence-length model: {:?}", fast_model);
        }
        fast_model
    } else {
        base_model
    };
    let tokenizer_path = models_dir.join("tokenizer.json");
    let config_path = models_dir.join("rl_agent_config.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        if !json_mode {
            println!("[WARN] Model weights not found in {:?}.", models_dir);
            println!("Please download the INT8 weights first by running:");
            println!("  powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1\n");
            println!("[FALLBACK] Running rule-based dry-run dispatch test...");
        }
        let (action, conf) = fallback_heuristic_routing(&input_prompt);
        let mutating = is_action_mutating(action);

        log_automation_event(
            &models_dir,
            &input_prompt,
            action,
            conf,
            "HEURISTIC_FALLBACK",
            if mutating { "DRY_RUN" } else { "READ_ONLY" },
            if mutating { "FALLBACK_PREVIEW_ONLY" } else { "SUCCESS" },
        );

        if json_mode {
            emit_json_and_exit(
                action,
                conf,
                false,
                mutating,
                0.0,
                "HEURISTIC_FALLBACK",
                if mutating { "PREVIEW_ONLY" } else { "SUCCESS" },
                if mutating {
                    Some("Fallback mode prohibits mutating execution. Dry-run preview only.".to_string())
                } else {
                    None
                },
            );
        } else {
            println!("Predicted Action : {}", action);
            println!("Confidence Score : {:.2}%\n", conf * 100.0);
            if mutating {
                println!("[SAFEGUARD] Fallback mode is strictly read-only. Mutating action '{}' locked to preview.", action);
            }
            execute_windows_action(action, &input_prompt, false)?;
        }
        return Ok(());
    }

    let dylib_path = models_dir.join("onnxruntime.dll");
    let env = if dylib_path.exists() {
        ort::init_from(&dylib_path)
            .map_err(|e| anyhow!("Failed to load onnxruntime.dll: {}", e))?
            .with_name("Laya-Siro")
            .build()
            .map_err(|e| anyhow!("Failed to build ORT environment: {}", e))?
    } else {
        ort::init()
            .with_name("Laya-Siro")
            .build()
            .map_err(|e| anyhow!("Failed to build ORT environment: {}", e))?
    };

    let mut session = build_low_memory_session(&env, &model_path)?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
    let temperature = load_temperature(&config_path, AUTOMATION_OPTIONS.len());

    let result = evaluate_laya_decision(
        &mut session,
        &tokenizer,
        &input_prompt,
        ROUTER_QUESTION,
        AUTOMATION_OPTIONS,
        temperature,
    )?;

    if !json_mode {
        println!("--------------------------------------------------");
        println!(">>> Real ONNX Neural Inference Diagnostic Data <<<");
        println!("--------------------------------------------------");
        println!("Inference Latency: {:.2} ms", result.latency_ms);
        println!("Model Input Contract: 5 tensors (input_ids, attention_mask, marker_pos, marker_mask, qtype)");
        println!("\nOption Evaluation (Calibrated via RLCD temp = {:.2}):", temperature);
        for (opt, (raw, prob)) in AUTOMATION_OPTIONS.iter().zip(result.raw_logits.iter().zip(result.calibrated_probs.iter())) {
            let is_chosen = *opt == result.action;
            let prefix = if is_chosen { "-> [CHOSEN]" } else { "   " };
            println!("{} {:<20} | Raw Logit: {:>6.3} | Calibrated: {:>6.2}%", prefix, opt, raw, prob * 100.0);
        }
        println!("\nAct Gate (act_logits) : [Answer: {:.4}, Escalate: {:.4}]", result.act_scores.0, result.act_scores.1);
        println!("Escalate / Abstain    : {}", if result.should_escalate { "YES" } else { "NO" });
        println!("Final Routing Action  : {}", result.action);
        println!("Confidence Score      : {:.2}%", result.confidence * 100.0);
        println!("--------------------------------------------------\n");
    }

    // 4. Guardrail Gate 3: Confidence & Act Gate Floor Check (>= 75.0%)
    if result.should_escalate || result.confidence < MINIMUM_CONFIDENCE_THRESHOLD {
        log_automation_event(
            &models_dir,
            &input_prompt,
            result.action,
            result.confidence,
            "NEURAL_MODEL",
            if execute_mode { "EXECUTE" } else { "DRY_RUN" },
            "BLOCKED_LOW_CONFIDENCE",
        );

        let reason = if result.should_escalate {
            "Model act gate triggered escalate/abstain flag".to_string()
        } else {
            format!("Confidence score ({:.2}%) is below the safety threshold of {:.2}%", result.confidence * 100.0, MINIMUM_CONFIDENCE_THRESHOLD * 100.0)
        };

        if json_mode {
            emit_json_and_exit(
                result.action,
                result.confidence,
                result.should_escalate,
                is_action_mutating(result.action),
                result.latency_ms,
                "NEURAL_MODEL",
                "BLOCKED_LOW_CONFIDENCE",
                Some(reason),
            );
        } else {
            println!("[ABORT] {}.", reason);
            println!("Action '{}' was blocked from automatic execution for system safety.", result.action);
        }
        return Ok(());
    }

    // 5. Guardrail Gate 4: Mutating Action Execution & Confirmation
    let mutating = is_action_mutating(result.action);

    if json_mode {
        log_automation_event(
            &models_dir,
            &input_prompt,
            result.action,
            result.confidence,
            "NEURAL_MODEL",
            if execute_mode { "EXECUTE" } else { "DRY_RUN" },
            if mutating && !execute_mode { "PREVIEW_ONLY" } else { "SUCCESS" },
        );

        emit_json_and_exit(
            result.action,
            result.confidence,
            result.should_escalate,
            mutating,
            result.latency_ms,
            "NEURAL_MODEL",
            if mutating && !execute_mode { "PREVIEW_ONLY" } else { "SUCCESS" },
            None,
        );
        return Ok(());
    }

    if mutating {
        if !execute_mode {
            log_automation_event(
                &models_dir,
                &input_prompt,
                result.action,
                result.confidence,
                "NEURAL_MODEL",
                "DRY_RUN",
                "PREVIEW_COMPLETED",
            );

            println!("[DRY-RUN DEFAULT] Action '{}' is mutating. Running in preview mode.", result.action);
            execute_windows_action(result.action, &input_prompt, false)?;
            println!("\n[HINT] To apply actual changes, re-run with '--execute' and confirm the prompt [y/N].");
            return Ok(());
        }

        // Execute mode requires interactive confirmation
        if !confirm_action(result.action, &input_prompt) {
            log_automation_event(
                &models_dir,
                &input_prompt,
                result.action,
                result.confidence,
                "NEURAL_MODEL",
                "EXECUTE",
                "USER_CANCELLED",
            );

            println!("\n[ABORTED] Action cancelled by user. No system changes were made.");
            return Ok(());
        }
    }

    // 6. Execute selected action
    log_automation_event(
        &models_dir,
        &input_prompt,
        result.action,
        result.confidence,
        "NEURAL_MODEL",
        if mutating { "EXECUTE" } else { "READ_ONLY" },
        "SUCCESS",
    );

    execute_windows_action(result.action, &input_prompt, execute_mode)?;

    Ok(())
}

fn print_usage(prog_name: &str) {
    println!("Usage: {} [--json] [--execute] \"<request text>\"", prog_name);
    println!("\nFlags:");
    println!("  --json           Output structured JSON result for agent skills and automated scripts");
    println!("  --execute, -e    Enable execution of mutating actions (defaults to safe dry-run preview)");
    println!("\nExamples:");
    println!("  # JSON Output for AI Agent Integration:");
    println!("  {} --json \"find which processes are hogging all my RAM\"", prog_name);
    println!("  {} --json \"git status\"", prog_name);
    println!("\n  # Safe Read-Only Queries (Run directly):");
    println!("  {} \"find which processes are hogging all my RAM\"", prog_name);
    println!("  {} \"git status\"", prog_name);
    println!("  {} \"git log\"", prog_name);
    println!("\n  # Mutating Actions (Dry-run by default):");
    println!("  {} \"delete temporary files in Windows cache\"", prog_name);
    println!("  {} --execute \"delete temporary files in Windows cache\"  # Prompts [y/N]", prog_name);
    println!("\n  # Hard-Blocked Policies (Always rejected):");
    println!("  {} \"delete system32 and wipe C: drive\"", prog_name);
}

fn resolve_models_dir() -> PathBuf {
    if let Ok(dir) = env::var("LAYA_MODELS_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return p;
        }
    }
    let cwd_models = PathBuf::from("models");
    if cwd_models.exists() {
        return cwd_models;
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            let rel = parent.join("../../models");
            if rel.exists() {
                return rel;
            }
            let adj = parent.join("models");
            if adj.exists() {
                return adj;
            }
        }
    }
    PathBuf::from("models")
}

fn build_low_memory_session(env: &Environment, model_path: &Path) -> Result<Session> {
    let mut builder = Session::builder(env)
        .map_err(|e| anyhow!("Failed to create SessionBuilder: {}", e))?;

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("Failed to set optimization level: {}", e))?;
    builder = builder
        .with_intra_threads(4)
        .map_err(|e| anyhow!("Failed to set intra threads: {}", e))?;
    builder = builder
        .with_inter_threads(1)
        .map_err(|e| anyhow!("Failed to set inter threads: {}", e))?;

    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("Failed to commit session from {:?}: {}", model_path, e))?;

    Ok(session)
}

struct NeuralDecisionResult<'a> {
    action: &'a str,
    confidence: f32,
    should_escalate: bool,
    latency_ms: f64,
    raw_logits: Vec<f32>,
    calibrated_probs: Vec<f32>,
    act_scores: (f32, f32),
}

fn evaluate_laya_decision<'a>(
    session: &mut Session,
    tokenizer: &Tokenizer,
    state: &str,
    question: &str,
    options: &'a [&'a str],
    temperature: f32,
) -> Result<NeuralDecisionResult<'a>> {
    let cls_id = tokenizer.token_to_id("[CLS]").unwrap_or(1);
    let sep_id = tokenizer.token_to_id("[SEP]").unwrap_or(2);
    let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(4);

    let mut input_ids: Vec<i64> = vec![cls_id as i64];

    let q_enc = tokenizer
        .encode(question, false)
        .map_err(|e| anyhow!("Tokenize question: {}", e))?;
    input_ids.extend(q_enc.get_ids().iter().map(|&id| id as i64));
    input_ids.push(sep_id as i64);

    let mut marker_positions: Vec<i64> = Vec::with_capacity(options.len());
    for opt in options {
        marker_positions.push(input_ids.len() as i64);
        input_ids.push(mask_id as i64);

        let opt_enc = tokenizer
            .encode(*opt, false)
            .map_err(|e| anyhow!("Tokenize option: {}", e))?;
        input_ids.extend(opt_enc.get_ids().iter().map(|&id| id as i64));
        input_ids.push(sep_id as i64);
    }

    input_ids.push(sep_id as i64);
    let state_enc = tokenizer
        .encode(state, false)
        .map_err(|e| anyhow!("Tokenize state: {}", e))?;
    input_ids.extend(state_enc.get_ids().iter().map(|&id| id as i64));
    input_ids.push(sep_id as i64);

    let pad_id = tokenizer
        .token_to_id("<pad>")
        .or_else(|| tokenizer.token_to_id("[PAD]"))
        .unwrap_or(0);

    let num_options = options.len();
    let max_seq_len = session
        .inputs()
        .iter()
        .find(|i| i.name() == "input_ids")
        .and_then(|i| match i.dtype() {
            ort::value::ValueType::Tensor { ref shape, .. } if shape.len() > 1 && shape[1] > 0 => {
                Some(shape[1] as usize)
            }
            _ => None,
        })
        .unwrap_or(128);
    let max_options = 20;

    let mut attention_mask: Vec<i64> = vec![1; input_ids.len().min(max_seq_len)];
    if input_ids.len() > max_seq_len {
        input_ids.truncate(max_seq_len);
    }
    input_ids.resize(max_seq_len, pad_id as i64);
    attention_mask.resize(max_seq_len, 0);

    let mut marker_mask: Vec<bool> = vec![true; num_options.min(max_options)];
    marker_positions.resize(max_options, 0);
    marker_mask.resize(max_options, false);

    let input_ids_arr = Array2::from_shape_vec((1, max_seq_len), input_ids)?;
    let attn_mask_arr = Array2::from_shape_vec((1, max_seq_len), attention_mask)?;
    let marker_pos_arr = Array2::from_shape_vec((1, max_options), marker_positions)?;
    let marker_mask_arr = Array2::from_shape_vec((1, max_options), marker_mask)?;
    let qtype_arr = Array1::from_vec(vec![0_i64]);

    let infer_start = std::time::Instant::now();
    let outputs = session.run(inputs![
        "input_ids" => TensorRef::from_array_view(&input_ids_arr)?,
        "attention_mask" => TensorRef::from_array_view(&attn_mask_arr)?,
        "marker_pos" => TensorRef::from_array_view(&marker_pos_arr)?,
        "marker_mask" => TensorRef::from_array_view(&marker_mask_arr)?,
        "qtype" => TensorRef::from_array_view(&qtype_arr)?,
    ])?;
    let latency_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let raw_logits_extracted = if let Some(t) = outputs.get("logits") {
        let (_, v) = t.try_extract_tensor::<f32>()?;
        v.to_vec()
    } else if let Some((_, t)) = outputs.iter().next() {
        let (_, v) = t.try_extract_tensor::<f32>()?;
        v.to_vec()
    } else {
        return Err(anyhow!("No outputs returned from ONNX session"));
    };

    let raw_logits: Vec<f32> = raw_logits_extracted[..num_options.min(raw_logits_extracted.len())].to_vec();
    let probs = calibrated_softmax(&raw_logits, temperature);

    let mut act_scores = (0.0_f32, 0.0_f32);
    let mut should_escalate = false;
    if let Some(t) = outputs.get("act_logits") {
        if let Ok((_, v)) = t.try_extract_tensor::<f32>() {
            if v.len() >= 2 {
                act_scores = (v[0], v[1]);
                if v[1] > v[0] {
                    should_escalate = true;
                }
            }
        }
    }

    let mut best_idx = 0;
    let mut best_prob = -1.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        if p > best_prob {
            best_prob = p;
            best_idx = i;
        }
    }

    Ok(NeuralDecisionResult {
        action: options[best_idx],
        confidence: best_prob,
        should_escalate,
        latency_ms,
        raw_logits,
        calibrated_probs: probs,
        act_scores,
    })
}

fn calibrated_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temp = if temperature > 0.0 { temperature } else { 1.0 };
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) / temp).exp())
        .collect();
    let sum: f32 = exps.iter().sum();

    if sum > 0.0 {
        exps.into_iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

fn load_temperature(config_path: &Path, num_options: usize) -> f32 {
    if let Ok(content) = fs::read_to_string(config_path) {
        if let Ok(config) = serde_json::from_str::<RlConfig>(&content) {
            if let Some(&t) = config.temperature_by_options.get(num_options) {
                return t;
            }
        }
    }
    1.05
}

fn fallback_heuristic_routing(prompt: &str) -> (&'static str, f32) {
    let lower = prompt.to_lowercase();
    if lower.contains("ram") || lower.contains("cpu") || lower.contains("process") || lower.contains("triage") {
        ("system_triage", 0.92)
    } else if lower.contains("download") || lower.contains("sort") || lower.contains("organize") {
        ("organize_downloads", 0.89)
    } else if lower.contains("temp") || lower.contains("clean") || lower.contains("cache") {
        ("clean_temp", 0.88)
    } else if lower.contains("system32") || lower.contains("wipe") || lower.contains("format") || lower.contains("delete all") {
        ("block_unsafe", 0.99)
    } else {
        ("system_triage", 0.51)
    }
}

/// Native Windows Automation Dispatcher with Safe Dry-Run and Allowlisted Git Execution
fn execute_windows_action(action: &str, prompt: &str, execute_mode: bool) -> Result<()> {
    println!("[DISPATCHER] Target Action: '{}'", action);

    match action {
        "system_triage" => {
            println!("\n>>> Running Windows System & Process Triage (Top WorkingSet)...");
            let output = Command::new("powershell")
                .arg("-NoProfile")
                .arg("-Command")
                .arg("Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 6 -Property ProcessName, @{Name='RAM_MB';Expression={[math]::Round($_.WorkingSet64 / 1MB, 1)}}, Id | Format-Table -AutoSize")
                .output()
                .context("Failed to run PowerShell process query")?;

            println!("{}", String::from_utf8_lossy(&output.stdout));
            println!("[DONE] System triage completed.");
        }

        "clean_temp" => {
            let temp_dir = env::var("TEMP").unwrap_or_else(|_| "C:\\Windows\\Temp".to_string());
            println!("\n>>> Windows Temp Cache: {:?}", temp_dir);

            let mut total_bytes: u64 = 0;
            let mut file_count = 0;
            let mut deleted_count = 0;
            let mut deleted_bytes: u64 = 0;

            if let Ok(entries) = fs::read_dir(&temp_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Ok(meta) = path.metadata() {
                            total_bytes += meta.len();
                            file_count += 1;

                            if execute_mode {
                                if fs::remove_file(&path).is_ok() {
                                    deleted_count += 1;
                                    deleted_bytes += meta.len();
                                }
                            }
                        }
                    }
                }
            }

            let size_mb = (total_bytes as f64) / (1024.0 * 1024.0);
            if execute_mode {
                let freed_mb = (deleted_bytes as f64) / (1024.0 * 1024.0);
                println!("[EXECUTED] Deleted {} unlocked files ({:.2} MB freed).", deleted_count, freed_mb);
                println!("Locked files currently in use by Windows apps were safely preserved.");
            } else {
                println!("Scanned {} files ({:.2} MB).", file_count, size_mb);
                println!("[DRY-RUN DEFAULT] No files deleted. Pass '--execute' to perform cleanup.");
            }
        }

        "organize_downloads" => {
            let user_profile = env::var("USERPROFILE").unwrap_or_else(|_| "C:\\Users\\Default".to_string());
            let downloads_dir = PathBuf::from(&user_profile).join("Downloads");

            println!("\n>>> Inspecting Downloads folder: {:?}", downloads_dir);
            if !downloads_dir.exists() {
                println!("[INFO] Downloads folder not found at path: {:?}", downloads_dir);
                return Ok(());
            }

            let mut count = 0;
            if let Ok(entries) = fs::read_dir(&downloads_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        count += 1;
                        if count <= 5 {
                            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("none");
                            println!("  [Found File] {:?} (ext: .{})", path.file_name().unwrap_or_default(), ext);
                        }
                    }
                }
            }

            println!("Total scanned files in Downloads: {}", count);
            if execute_mode {
                println!("[SAFEGUARD] Auto-move is currently disabled. Files in Downloads remain untouched.");
            } else {
                println!("[DRY-RUN DEFAULT] No files moved. Pass '--execute' to organize files into subfolders.");
            }
        }

        "git_status" => {
            println!("\n>>> Running Read-Only Git Status (git status --short)...");
            let output = Command::new("git")
                .arg("status")
                .arg("--short")
                .output();

            match output {
                Ok(out) if out.status.success() => {
                    let text = String::from_utf8_lossy(&out.stdout);
                    if text.trim().is_empty() {
                        println!("Working tree clean (no modified or untracked files).");
                    } else {
                        println!("{}", text);
                    }
                    println!("[DONE] Read-only Git status check completed.");
                }
                Ok(out) => {
                    println!("[WARN] Git command returned non-zero status:");
                    println!("{}", String::from_utf8_lossy(&out.stderr));
                }
                Err(e) => {
                    println!("[ERROR] Failed to execute git: {}", e);
                }
            }
        }

        "git_log" => {
            println!("\n>>> Running Read-Only Git Log (git log -n 5 --oneline)...");
            let output = Command::new("git")
                .arg("log")
                .arg("-n")
                .arg("5")
                .arg("--oneline")
                .output();

            match output {
                Ok(out) if out.status.success() => {
                    println!("{}", String::from_utf8_lossy(&out.stdout));
                    println!("[DONE] Read-only Git log check completed.");
                }
                Ok(out) => {
                    println!("[WARN] Git log returned error:");
                    println!("{}", String::from_utf8_lossy(&out.stderr));
                }
                Err(e) => {
                    println!("[ERROR] Failed to execute git: {}", e);
                }
            }
        }

        "git_stash_list" => {
            println!("\n>>> Running Read-Only Git Stash List (git stash list)...");
            let output = Command::new("git")
                .arg("stash")
                .arg("list")
                .output();

            match output {
                Ok(out) if out.status.success() => {
                    let text = String::from_utf8_lossy(&out.stdout);
                    if text.trim().is_empty() {
                        println!("No stashed changes found.");
                    } else {
                        println!("{}", text);
                    }
                    println!("[DONE] Read-only Git stash list check completed.");
                }
                Ok(out) => {
                    println!("[WARN] Git stash list returned error:");
                    println!("{}", String::from_utf8_lossy(&out.stderr));
                }
                Err(e) => {
                    println!("[ERROR] Failed to execute git: {}", e);
                }
            }
        }

        "git_diff" => {
            println!("\n>>> Running Read-Only Git Diff (git diff --stat)...");
            let output = Command::new("git")
                .arg("diff")
                .arg("--stat")
                .output();

            match output {
                Ok(out) if out.status.success() => {
                    let text = String::from_utf8_lossy(&out.stdout);
                    if text.trim().is_empty() {
                        println!("No changes in working tree against index.");
                    } else {
                        println!("{}", text);
                    }
                    println!("[DONE] Read-only Git diff completed.");
                }
                Ok(out) => {
                    println!("[WARN] Git diff returned error:");
                    println!("{}", String::from_utf8_lossy(&out.stderr));
                }
                Err(e) => {
                    println!("[ERROR] Failed to execute git: {}", e);
                }
            }
        }

        "git_pull_ff_only" => {
            println!("\n>>> Inspecting Git Working Tree Before Pull...");
            let status_out = Command::new("git")
                .arg("status")
                .arg("--short")
                .output();

            if let Ok(out) = status_out {
                let text = String::from_utf8_lossy(&out.stdout);
                if !text.trim().is_empty() {
                    println!("\n[BLOCKED] Working tree is dirty with uncommitted changes:\n{}", text);
                    println!("Refusing to pull changes while working tree has modifications.");
                    return Ok(());
                }
            }

            if execute_mode {
                println!("\n>>> Running Fast-Forward Git Pull (git pull --ff-only)...");
                let output = Command::new("git")
                    .arg("pull")
                    .arg("--ff-only")
                    .output();

                match output {
                    Ok(out) => {
                        println!("{}", String::from_utf8_lossy(&out.stdout));
                        if !out.status.success() {
                            println!("[WARN] Git pull error: {}", String::from_utf8_lossy(&out.stderr));
                        } else {
                            println!("[EXECUTED] Fast-forward git pull completed successfully.");
                        }
                    }
                    Err(e) => {
                        println!("[ERROR] Failed to execute git: {}", e);
                    }
                }
            } else {
                println!("[DRY-RUN DEFAULT] Working tree is clean. Pass '--execute' to perform 'git pull --ff-only'.");
            }
        }

        "laravel_test" => {
            println!("\n>>> Running Laravel Backend Tests (php artisan test)...");
            let output = Command::new("php")
                .arg("artisan")
                .arg("test")
                .output();

            match output {
                Ok(out) => {
                    println!("{}", String::from_utf8_lossy(&out.stdout));
                    if !out.status.success() {
                        let err = String::from_utf8_lossy(&out.stderr);
                        if !err.trim().is_empty() {
                            println!("[WARN] Test runner stderr: {}", err);
                        }
                    }
                    println!("[DONE] Laravel test suite execution finished.");
                }
                Err(e) => {
                    println!("[ERROR] Failed to execute 'php artisan test': {}", e);
                    println!("Ensure PHP is installed and available in PATH, or run from Laravel project root.");
                }
            }
        }

        "laravel_route_list" => {
            println!("\n>>> Inspecting Laravel Routes (php artisan route:list)...");
            let output = Command::new("php")
                .arg("artisan")
                .arg("route:list")
                .output();

            match output {
                Ok(out) => {
                    println!("{}", String::from_utf8_lossy(&out.stdout));
                    if !out.status.success() {
                        let err = String::from_utf8_lossy(&out.stderr);
                        if !err.trim().is_empty() {
                            println!("[WARN] Route list stderr: {}", err);
                        }
                    }
                    println!("[DONE] Route inspection finished.");
                }
                Err(e) => {
                    println!("[ERROR] Failed to execute 'php artisan route:list': {}", e);
                    println!("Ensure PHP is installed and available in PATH, or run from Laravel project root.");
                }
            }
        }

        "edge_engine_check" => {
            println!("\n>>> Compile-Checking Edge Engine (cargo check -j 2)...");
            let edge_dir = Path::new("clients/edge-engine-rs");
            let mut cmd = Command::new("cargo");
            cmd.arg("check").arg("-j").arg("2");
            if edge_dir.exists() {
                cmd.current_dir(edge_dir);
                println!("Working directory: {:?}", edge_dir);
            } else {
                println!("[INFO] 'clients/edge-engine-rs' not in current directory. Checking current workspace...");
            }

            match cmd.output() {
                Ok(out) => {
                    println!("{}", String::from_utf8_lossy(&out.stderr));
                    if out.status.success() {
                        println!("[DONE] Edge engine compilation check passed.");
                    } else {
                        println!("[WARN] Edge engine compilation returned errors.");
                    }
                }
                Err(e) => {
                    println!("[ERROR] Failed to execute cargo: {}", e);
                }
            }
        }

        "protocol_test" => {
            println!("\n>>> Testing Terminal Protocol (cargo test)...");
            let proto_dir = Path::new("packages/protocol");
            let mut cmd = Command::new("cargo");
            cmd.arg("test");
            if proto_dir.exists() {
                cmd.current_dir(proto_dir);
                println!("Working directory: {:?}", proto_dir);
            } else {
                println!("[INFO] 'packages/protocol' not in current directory. Running cargo test locally...");
            }

            match cmd.output() {
                Ok(out) => {
                    println!("{}", String::from_utf8_lossy(&out.stdout));
                    let err = String::from_utf8_lossy(&out.stderr);
                    if !err.trim().is_empty() {
                        println!("{}", err);
                    }
                    println!("[DONE] Protocol tests finished.");
                }
                Err(e) => {
                    println!("[ERROR] Failed to execute cargo test: {}", e);
                }
            }
        }

        "npm_build" => {
            if execute_mode {
                println!("\n>>> Building Frontend Assets (npm run build)...");
                let npm_bin = if cfg!(windows) { "npm.cmd" } else { "npm" };
                let output = Command::new(npm_bin)
                    .arg("run")
                    .arg("build")
                    .output();

                match output {
                    Ok(out) => {
                        println!("{}", String::from_utf8_lossy(&out.stdout));
                        if !out.status.success() {
                            println!("[WARN] Frontend build error: {}", String::from_utf8_lossy(&out.stderr));
                        } else {
                            println!("[EXECUTED] Frontend assets built successfully.");
                        }
                    }
                    Err(e) => {
                        println!("[ERROR] Failed to execute npm: {}", e);
                    }
                }
            } else {
                println!("\n[DRY-RUN DEFAULT] Target command: 'npm run build'. Pass '--execute' and confirm [y/N] to compile assets.");
            }
        }

        "laravel_clear_cache" => {
            if execute_mode {
                println!("\n>>> Clearing Laravel Caches (php artisan optimize:clear)...");
                let output = Command::new("php")
                    .arg("artisan")
                    .arg("optimize:clear")
                    .output();

                match output {
                    Ok(out) => {
                        println!("{}", String::from_utf8_lossy(&out.stdout));
                        if !out.status.success() {
                            println!("[WARN] Optimize clear error: {}", String::from_utf8_lossy(&out.stderr));
                        } else {
                            println!("[EXECUTED] Laravel application caches cleared.");
                        }
                    }
                    Err(e) => {
                        println!("[ERROR] Failed to execute 'php artisan optimize:clear': {}", e);
                    }
                }
            } else {
                println!("\n[DRY-RUN DEFAULT] Target command: 'php artisan optimize:clear'. Pass '--execute' and confirm [y/N] to clear caches.");
            }
        }

        "block_unsafe" => {
            println!("\n[GUARDRAIL TRIGGERED] Action blocked!");
            println!("The prompt '{}' was classified as unsafe or destructive.", prompt);
            println!("No filesystem changes or commands were executed.");
        }

        _ => {
            println!("[WARN] Unknown action '{}'. Ignoring.", action);
        }
    }

    Ok(())
}
