use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Array2};
use ort::{
    environment::Environment,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct RlConfig {
    #[serde(default = "default_temperature")]
    temperature_by_options: Vec<f32>,
}

fn default_temperature() -> Vec<f32> {
    vec![1.05]
}

/// Available Windows automation targets
const AUTOMATION_OPTIONS: &[&str] = &[
    "system_triage",
    "organize_downloads",
    "clean_temp",
    "block_unsafe",
];

const ROUTER_QUESTION: &str = "Which Windows automation action should be executed for this request?";

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_usage(&args[0]);
        return Ok(());
    }

    let input_prompt = args[1..].join(" ");
    println!("==================================================");
    println!(" [Laya-Siro] Windows Low-Memory Automation Engine ");
    println!("==================================================");
    println!("Prompt: \"{}\"\n", input_prompt);

    let models_dir = resolve_models_dir();
    let fast_model = models_dir.join("model_fast.onnx");
    let base_model = models_dir.join("model.onnx");
    let model_path = if fast_model.exists() {
        println!("[Fast Inference Engine] Using optimized sequence-length model: {:?}", fast_model);
        fast_model
    } else {
        base_model
    };
    let tokenizer_path = models_dir.join("tokenizer.json");
    let config_path = models_dir.join("rl_agent_config.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("[WARN] Model weights not found in {:?}.", models_dir);
        println!("Please download the INT8 weights first by running:");
        println!("  powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1\n");
        println!("[FALLBACK] Running rule-based dry-run dispatch test...");
        let (action, conf) = fallback_heuristic_routing(&input_prompt);
        println!("Predicted Action : {}", action);
        println!("Confidence Score : {:.2}%\n", conf * 100.0);
        execute_windows_action(action, &input_prompt)?;
        return Ok(());
    }

    // 1. Initialize ONNX Runtime environment
    let dylib_path = models_dir.join("onnxruntime.dll");
    let env = if dylib_path.exists() {
        println!("[ORT Engine] Loaded runtime from: {:?}", dylib_path);
        ort::init_from(&dylib_path)
            .map_err(|e| anyhow!("Failed to load onnxruntime.dll: {}", e))?
            .with_name("Laya-Siro")
            .build()
            .map_err(|e| anyhow!("Failed to build ORT environment: {}", e))?
    } else {
        println!("[ORT Engine] Initializing default system runtime...");
        ort::init()
            .with_name("Laya-Siro")
            .build()
            .map_err(|e| anyhow!("Failed to build ORT environment: {}", e))?
    };

    // 2. Build low-memory session optimized for 8 GB RAM / i5-1155G7
    let mut session = build_low_memory_session(&env, &model_path)?;

    println!(">>> ONNX Model Graph Inputs Inspection <<<");
    for (i, input) in session.inputs().iter().enumerate() {
        println!("  Input  [{}] '{}': {:?}", i, input.name(), input.dtype());
    }
    for (i, output) in session.outputs().iter().enumerate() {
        println!("  Output [{}] '{}': {:?}", i, output.name(), output.dtype());
    }
    println!();

    // 3. Load tokenizer and calibration config
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
    let temperature = load_temperature(&config_path, AUTOMATION_OPTIONS.len());

    // 4. Run 5-tensor graph inference
    let result = evaluate_laya_decision(
        &mut session,
        &tokenizer,
        &input_prompt,
        ROUTER_QUESTION,
        AUTOMATION_OPTIONS,
        temperature,
    )?;

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

    // 5. Safety check: Escalate or block if confidence is low or action flagged
    if result.should_escalate || result.confidence < 0.35 {
        println!("[ABORT] Confidence is below safety threshold or model triggered abstain gate.");
        println!("Request not executed on system for safety.");
        return Ok(());
    }

    // 6. Execute selected Windows action
    execute_windows_action(result.action, &input_prompt)?;

    Ok(())
}

fn print_usage(prog_name: &str) {
    println!("Usage: {} \"<request text>\"", prog_name);
    println!("\nExamples:");
    println!("  {} \"find which processes are hogging all my RAM\"", prog_name);
    println!("  {} \"tidy up my downloads folder by sorting files\"", prog_name);
    println!("  {} \"delete temporary files in Windows cache\"", prog_name);
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

/// Builds an ORT session strictly capped for minimal memory overhead
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

/// Evaluates Laya's exact 5-tensor input contract
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

    // Layout: [CLS] question [SEP] [MASK] opt_1 [SEP] [MASK] opt_2 [SEP] ... [SEP] [SEP] state [SEP]
    let mut input_ids: Vec<i64> = vec![cls_id as i64];

    let q_enc = tokenizer
        .encode(question, false)
        .map_err(|e| anyhow!("Tokenize question: {}", e))?;
    input_ids.extend(q_enc.get_ids().iter().map(|&id| id as i64));
    input_ids.push(sep_id as i64);

    let mut marker_positions: Vec<i64> = Vec::with_capacity(options.len());
    for opt in options {
        marker_positions.push(input_ids.len() as i64); // Index of [MASK]
        input_ids.push(mask_id as i64);

        let opt_enc = tokenizer
            .encode(*opt, false)
            .map_err(|e| anyhow!("Tokenize option: {}", e))?;
        input_ids.extend(opt_enc.get_ids().iter().map(|&id| id as i64));
        input_ids.push(sep_id as i64);
    }

    // Double [SEP] before state
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
    let qtype_arr = Array1::from_vec(vec![0_i64]); // 0 = choice

    let infer_start = std::time::Instant::now();
    let outputs = session.run(inputs![
        "input_ids" => TensorRef::from_array_view(&input_ids_arr)?,
        "attention_mask" => TensorRef::from_array_view(&attn_mask_arr)?,
        "marker_pos" => TensorRef::from_array_view(&marker_pos_arr)?,
        "marker_mask" => TensorRef::from_array_view(&marker_mask_arr)?,
        "qtype" => TensorRef::from_array_view(&qtype_arr)?,
    ])?;
    let latency_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    // Extract logits for options safely from outputs map
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

    // Extract act_logits (2-way gate: [0] = answer, [1] = escalate/abstain)
    let mut act_scores = (0.0_f32, 0.0_f32);
    let mut should_escalate = false;
    if let Some(act_tensor) = outputs.get("act_logits").or_else(|| outputs.get("act")) {
        if let Ok((_, act_vals)) = act_tensor.try_extract_tensor::<f32>() {
            if act_vals.len() >= 2 {
                act_scores = (act_vals[0], act_vals[1]);
                should_escalate = act_vals[1] > act_vals[0];
            }
        }
    }

    let mut best_idx = 0;
    let mut best_prob = 0.0;
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

fn calibrated_softmax(logits: &[f32], temp: f32) -> Vec<f32> {
    let t = if temp <= 0.01 { 1.0 } else { temp };
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| ((x - max_val) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.into_iter().map(|x| x / sum).collect()
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

/// Fallback heuristic routing when model file is not yet downloaded
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

/// Native Windows Automation Dispatcher
fn execute_windows_action(action: &str, prompt: &str) -> Result<()> {
    println!("[DISPATCHER] Executing native Windows action: '{}'", action);

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
                        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("none");
                        count += 1;
                        if count <= 5 {
                            println!("  [Found File] {:?} (ext: .{})", path.file_name().unwrap_or_default(), ext);
                        }
                    }
                }
            }

            println!("Total scanned files in Downloads: {}", count);
            println!("[NOTE] Dry-run audit completed. Set auto_move=true to categorize into Subfolders.");
        }

        "clean_temp" => {
            let temp_dir = env::var("TEMP").unwrap_or_else(|_| "C:\\Windows\\Temp".to_string());
            println!("\n>>> Inspecting Windows Temp Cache: {:?}", temp_dir);

            let mut total_bytes: u64 = 0;
            let mut file_count = 0;
            if let Ok(entries) = fs::read_dir(&temp_dir) {
                for entry in entries.flatten() {
                    if let Ok(meta) = entry.metadata() {
                        if meta.is_file() {
                            total_bytes += meta.len();
                            file_count += 1;
                        }
                    }
                }
            }

            let size_mb = (total_bytes as f64) / (1024.0 * 1024.0);
            println!("Temp files scanned: {} files ({:.2} MB)", file_count, size_mb);
            println!("[DONE] Cache inspection complete. Temp storage intact.");
        }

        "block_unsafe" => {
            println!("\n[GUARDRAIL TRIGGERED] Action blocked!");
            println!("The prompt '{}' was classified as unsafe or destructive.", prompt);
            println!("No filesystem changes were made.");
        }

        _ => {
            println!("[WARN] Unknown action '{}'. Ignoring.", action);
        }
    }

    Ok(())
}

