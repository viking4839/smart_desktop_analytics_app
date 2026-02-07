fn main() {
    // This tells Cargo to rerun this script if Tauri config changes
    println!("cargo:rerun-if-changed=tauri.conf.json");
    
    tauri_build::build()
}