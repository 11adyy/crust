mod app;
mod logger;
mod storage;
mod multiplayer;
mod ui;
use logger::{LogLevel, init_logger, log};
use storage::data;
use std::fs;

fn main() {
    if let Err(e) = setup_logger() {
        eprintln!("Could not initialize logger: {}", e);
        std::process::exit(1);
    }

    log(LogLevel::Info, "Starting crust...");

    if let Err(e) = app::run_game() {
        log(
            LogLevel::Error,
            &format!("Error occurred while starting game: {}", e),
        );
        std::process::exit(1);
    }
}

fn setup_logger() -> Result<(), Box<dyn std::error::Error>> {
    let proj_dirs = data::get_project_dirs()?;

    let log_dir = proj_dirs.data_dir().join("logs");
    fs::create_dir_all(&log_dir)?;
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let log_path = log_dir.join(format!("crust-{}.log", timestamp));
    let log_path_str = log_path
        .to_str()
        .ok_or("File path contains illegal chars")?;

    init_logger(log_path_str);
    Ok(())
}
