#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use tokio::process::Command;

#[derive(Serialize, Deserialize, Clone)]
struct CheckItem {
    label: String,
    detail: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct ScanResult {
    file: String,
    sha256: String,
    verdict: String,
    known_malware_layer: Vec<CheckItem>,
    weight_space_layer: Vec<CheckItem>,
}

fn find_sidecar_paths() -> (PathBuf, PathBuf) {
    let exe_dir = std::env::current_exe()
        .expect("Failed to get executable path")
        .parent()
        .expect("Failed to get executable directory")
        .to_path_buf();

    let is_dev_mode = exe_dir.ends_with("target/debug") || exe_dir.ends_with("target/release");

    if is_dev_mode {
        let mut project_root = exe_dir.clone();
        project_root.pop(); // target
        project_root.pop(); // ai-kavacha root

        let sidecar_dir = project_root.join("sidecar");
        let python_exe = sidecar_dir.join("kavacha/bin/python3");
        let script_path = sidecar_dir.join("scan_engine.py");

        (python_exe, script_path)
    } else {
        let sidecar_dir = exe_dir.join("sidecar");
        let python_exe = if cfg!(target_os = "windows") {
            sidecar_dir.join("kavacha/Scripts/python.exe")
        } else {
            sidecar_dir.join("kavacha/bin/python3")
        };
        let script_path = sidecar_dir.join("scan_engine.py");

        (python_exe, script_path)
    }
}

#[tauri::command]
async fn scan_file(_app: tauri::AppHandle, path: String) -> Result<ScanResult, String> {
    let (python_exe, script_path) = find_sidecar_paths();

    // Silently fall back to system python3 if the bundled venv isn't found
    let python_cmd = if python_exe.exists() {
        python_exe.to_string_lossy().to_string()
    } else {
        "python3".to_string()
    };

    let output = Command::new(&python_cmd)
        .arg(&script_path)
        .arg(&path)
        .output()
        .await
        .map_err(|e| format!("Failed to execute python sidecar: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    // Only log to terminal if there is an actual error
    if !output.status.success() {
        eprintln!("Python sidecar failed: {}", stderr);
        return Err(format!("Python script failed. Check terminal for details."));
    }

    let json_result: Value = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let checks = json_result["checks"]
        .as_array()
        .ok_or("Invalid checks format")?;

    // --- SMART ROUTING LOGIC (KEPT 100% INTACT) ---
    let mut known_malware_layer: Vec<CheckItem> = Vec::new();
    let mut weight_space_layer: Vec<CheckItem> = Vec::new();

    for check in checks {
        let label_str = check["label"].as_str().unwrap_or("unknown");
        let detail_str = check["detail"].as_str().unwrap_or("");
        let status_str = check["status"].as_str().unwrap_or("unknown").to_uppercase();

        let label_lower = label_str.to_lowercase();
        let is_lora_check = label_lower.contains("lora") || label_lower.contains("spectral");

        let item = CheckItem {
            label: label_str.to_string(),
            detail: format!("[{}] {}", status_str, detail_str)
        };

        // Route LoRA/Spectral checks to the second panel
        if is_lora_check {
            weight_space_layer.push(item);
        } else {
            // Route everything else to the first panel
            known_malware_layer.push(item);
        }
    }

    let overall_status = json_result["overall_status"].as_str().unwrap_or("unknown");
    let verdict = match overall_status {
        "clean" => "Clean - No threats detected",
        "malicious" => "Malicious - Threats detected!",
        "error" => "Error - Scan failed",
        _ => "Unknown status",
    };

    Ok(ScanResult {
        file: path,
        sha256: json_result["file_hash"].as_str().unwrap_or("").to_string(),
        verdict: verdict.to_string(),
        known_malware_layer,
        weight_space_layer,
    })
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![scan_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
