use std::io::IsTerminal;

use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

pub fn setup_logger(filter: EnvFilter) {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_file(true)
        .with_level(true)
        .with_line_number(true)
        .with_env_filter(filter)
        .with_ansi(std::io::stderr().is_terminal())
        .init();
}

pub fn default_env_filter(is_verbose: bool) -> EnvFilter {
    match EnvFilter::try_from_default_env() {
        Ok(filter) => filter,
        Err(_e) => {
            let default_level = if is_verbose {
                LevelFilter::TRACE
            } else {
                LevelFilter::INFO
            };

            EnvFilter::builder()
                .with_default_directive(default_level.into())
                .from_env_lossy()
        }
    }
}
