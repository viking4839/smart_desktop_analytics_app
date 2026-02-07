use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tauri::State;

struct BackendConnection {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

struct AppState {
    backend: Arc<Mutex<Option<BackendConnection>>>,
}

#[tauri::command]
async fn call_python_backend(
    command: String,
    payload: Value,
    state: State<'_, AppState>,
) -> Result<Value, String> {
    println!("📨 Sending command to Python: {}", command);
    
    let mut backend_lock = state.backend.lock().await;
    
    // If backend is not connected, try to reconnect
    if backend_lock.is_none() {
        println!("⚠️  Backend not connected, attempting to reconnect...");
        match connect_to_backend() {
            Ok(conn) => {
                *backend_lock = Some(conn);
                println!("✅ Reconnected to Python backend");
            }
            Err(e) => {
                return Err(format!("Failed to reconnect to backend: {}", e));
            }
        }
    }
    
    let backend = backend_lock.as_mut().ok_or("Python backend not connected")?;
    
    // Generate request ID
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("Time error: {}", e))?
        .as_nanos();
    let request_id = format!("req-{}", timestamp);
    
    // Create JSON-RPC 2.0 request (MATCHES Python's expected format)
    let request = json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": command,
        "params": payload
    });
    
    let request_str = request.to_string() + "\n";
    
    // Send to Python backend
    backend.stdin
        .write_all(request_str.as_bytes())
        .map_err(|e| format!("Failed to write to Python: {}", e))?;
    
    backend.stdin
        .flush()
        .map_err(|e| format!("Failed to flush stdin: {}", e))?;
    
    println!("📤 Sent request (id: {}): {}", request_id, command);
    
    // Read response from Python
    let mut response_line = String::new();
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(30);
    
    while start.elapsed() < timeout {
        match backend.stdout.read_line(&mut response_line) {
            Ok(0) => {
                // EOF - backend disconnected
                *backend_lock = None;
                return Err("Python backend disconnected".to_string());
            }
            Ok(_) => {
                let trimmed = response_line.trim();
                if !trimmed.is_empty() {
                    let preview = if trimmed.len() > 100 {
                        format!("{}...", &trimmed[0..100])
                    } else {
                        trimmed.to_string()
                    };
                    println!("📥 Received response: {}", preview);
                    break;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No data yet, wait a bit
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
            Err(e) => {
                *backend_lock = None;
                return Err(format!("Failed to read from Python: {}", e));
            }
        }
    }
    
    if response_line.trim().is_empty() {
        return Err("Timeout waiting for Python response".to_string());
    }
    
    // Parse JSON-RPC response
    let response: Value = serde_json::from_str(&response_line)
        .map_err(|e| format!("Invalid JSON from Python: {} (raw: {})", e, response_line))?;
    
    // Check for JSON-RPC error
    if let Some(error) = response.get("error") {
        if !error.is_null() {
            let error_msg = error.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            return Err(format!("Python backend error: {}", error_msg));
        }
    }
    
    // Extract result from JSON-RPC response
    response.get("result")
        .cloned()
        .ok_or_else(|| "No result in JSON-RPC response".to_string())
}

#[tauri::command]
async fn check_backend_health(state: State<'_, AppState>) -> Result<Value, String> {
    println!("🏥 Checking backend health...");
    
    let backend_lock = state.backend.lock().await;
    
    if let Some(_conn) = &*backend_lock {
        // Try to send a ping to check if backend is responsive
        drop(backend_lock); // Release lock to avoid deadlock
        
        // Send ping command
        let result = call_python_backend(
            "ping".to_string(),
            json!({}),
            state.clone()
        ).await;
        
        match result {
            Ok(response) => Ok(json!({
                "status": "connected",
                "message": "Python backend is responsive",
                "ping_response": response
            })),
            Err(e) => Ok(json!({
                "status": "disconnected",
                "message": format!("Backend error: {}", e)
            }))
        }
    } else {
        Ok(json!({
            "status": "disconnected", 
            "message": "Python backend is not connected"
        }))
    }
}

#[tauri::command]
async fn restart_backend(state: State<'_, AppState>) -> Result<Value, String> {
    println!("🔄 Restarting Python backend...");
    
    let mut backend_lock = state.backend.lock().await;
    
    // Kill existing backend if running
    if let Some(mut conn) = backend_lock.take() {
        let _ = conn.child.kill();
        let _ = conn.child.wait();
        println!("✅ Stopped existing Python backend");
    }
    
    // Start new backend
    match connect_to_backend() {
        Ok(conn) => {
            *backend_lock = Some(conn);
            println!("✅ Python backend restarted successfully");
            Ok(json!({"success": true, "message": "Backend restarted"}))
        }
        Err(e) => {
            println!("❌ Failed to restart backend: {}", e);
            Err(format!("Failed to restart backend: {}", e))
        }
    }
}

fn connect_to_backend() -> Result<BackendConnection, String> {
    println!("🚀 Connecting to Python backend...");
    
    let current_dir = std::env::current_dir()
        .map_err(|e| format!("Failed to get current dir: {}", e))?;
    
    // Find backend directory
    let possible_paths = [
        current_dir.join("../../backend"),
        current_dir.join("../backend"),
        current_dir.join("backend"),
    ];
    
    let backend_dir = possible_paths.iter()
        .find(|p| p.exists())
        .ok_or_else(|| "Backend directory not found".to_string())?;
    
    println!("🔍 Found backend at: {:?}", backend_dir);
    
    let main_py = backend_dir.join("main.py");
    if !main_py.exists() {
        return Err(format!("main.py not found at {:?}", main_py));
    }
    
    // Use virtual environment Python if available
    let python_exe = if cfg!(target_os = "windows") {
        let venv_python = backend_dir.join("venv").join("Scripts").join("python.exe");
        if venv_python.exists() {
            venv_python.to_string_lossy().to_string()
        } else {
            "python".to_string()
        }
    } else {
        let venv_python = backend_dir.join("venv").join("bin").join("python");
        if venv_python.exists() {
            venv_python.to_string_lossy().to_string()
        } else {
            "python3".to_string()
        }
    };
    
    println!("🐍 Using Python: {}", python_exe);
    
    // Start Python process with proper pipes
    let mut child = Command::new(&python_exe)
        .arg(main_py.to_str().unwrap())
        .current_dir(&backend_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python: {}", e))?;
    
    // Get handles to stdin/stdout
    let stdin = child.stdin.take()
        .ok_or_else(|| "Failed to get stdin".to_string())?;
    
    let stdout = child.stdout.take()
        .ok_or_else(|| "Failed to get stdout".to_string())?;
    
    let reader = BufReader::new(stdout);
    
    // Get stderr for monitoring
    let stderr = child.stderr.take()
        .ok_or_else(|| "Failed to get stderr".to_string())?;
    
    // Create a thread to monitor stderr for "READY" signal
    let ready_signal = Arc::new(std::sync::Mutex::new(false));
    let ready_signal_clone = ready_signal.clone();
    
    std::thread::spawn(move || {
        let stderr_reader = BufReader::new(stderr);
        for line in stderr_reader.lines() {
            match line {
                Ok(line) if line.contains("READY") => {
                    println!("✅ Python backend ready!");
                    *ready_signal_clone.lock().unwrap() = true;
                }
                Ok(line) => {
                    println!("🐍 [Python] {}", line);
                }
                Err(e) => {
                    eprintln!("❌ Error reading Python stderr: {}", e);
                }
            }
        }
    });
    
    // Wait for "READY" signal or timeout
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(10);
    
    while start.elapsed() < timeout {
        if *ready_signal.lock().unwrap() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    
    if !*ready_signal.lock().unwrap() {
        // Try to kill the process if it didn't become ready
        let _ = child.kill();
        let _ = child.wait();
        return Err("Python backend didn't become ready in time".to_string());
    }
    
    Ok(BackendConnection {
        child,
        stdin,
        stdout: reader,
    })
}

fn main() {
    println!("🚀 Starting Smart Desktop Analytics...");
    
    // Connect to Python backend
    let backend_connection = match connect_to_backend() {
        Ok(conn) => {
            println!("✅ Python backend connected successfully!");
            Some(conn)
        }
        Err(e) => {
            println!("⚠️  Failed to connect to Python backend: {}", e);
            println!("❌ Cannot start without backend. Make sure Python dependencies are installed:");
            println!("   cd backend && pip install pandas numpy openpyxl sqlalchemy aiosqlite");
            None
        }
    };
    
    // If backend failed to start, exit
    if backend_connection.is_none() {
        println!("💥 Exiting - Python backend required");
        return;
    }
    
    let app_state = AppState {
        backend: Arc::new(Mutex::new(backend_connection)),
    };
    
    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            call_python_backend,
            check_backend_health,
            restart_backend
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}