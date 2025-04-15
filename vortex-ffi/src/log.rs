use log::LevelFilter;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

use crate::error::VXError;

#[repr(C)]
#[allow(non_camel_case_types)]
pub enum VXLogLevel {
    LOG_LEVEL_OFF = 0,
    LOG_LEVEL_ERROR = 1,
    LOG_LEVEL_WARN = 2,
    LOG_LEVEL_INFO = 3,
    LOG_LEVEL_DEBUG = 4,
    LOG_LEVEL_TRACE = 5,
}

/// Initialize native logging with the specified level.
///
/// This function is optional, if it is not called then no runtime
/// logger will be installed.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_init_logging(level: VXLogLevel) {
    let filter = match level {
        VXLogLevel::LOG_LEVEL_OFF => LevelFilter::Off,
        VXLogLevel::LOG_LEVEL_ERROR => LevelFilter::Error,
        VXLogLevel::LOG_LEVEL_WARN => LevelFilter::Warn,
        VXLogLevel::LOG_LEVEL_INFO => LevelFilter::Info,
        VXLogLevel::LOG_LEVEL_DEBUG => LevelFilter::Debug,
        VXLogLevel::LOG_LEVEL_TRACE => LevelFilter::Trace,
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
