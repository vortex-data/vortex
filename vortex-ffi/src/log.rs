use log::LevelFilter;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

pub const LOG_LEVEL_OFF: u8 = 0;
pub const LOG_LEVEL_ERROR: u8 = 1;
pub const LOG_LEVEL_WARN: u8 = 2;
pub const LOG_LEVEL_INFO: u8 = 3;
pub const LOG_LEVEL_DEBUG: u8 = 4;
pub const LOG_LEVEL_TRACE: u8 = 5;

/// Initialize native logging with the specified level.
///
/// This function is optional, if it is not called then no runtime
/// logger will be installed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_init_logging(level: u8) {
    let filter = match level {
        LOG_LEVEL_OFF => LevelFilter::Off,
        LOG_LEVEL_ERROR => LevelFilter::Error,
        LOG_LEVEL_WARN => LevelFilter::Warn,
        LOG_LEVEL_INFO => LevelFilter::Info,
        LOG_LEVEL_DEBUG => LevelFilter::Debug,
        LOG_LEVEL_TRACE => LevelFilter::Trace,
        _ => {
            return;
        }
    };

    TermLogger::init(
        filter,
        Config::default(),
        TerminalMode::Stderr,
        ColorChoice::Auto,
    )
    .ok();
}
