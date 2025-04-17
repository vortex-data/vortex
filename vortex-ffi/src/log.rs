use log::LevelFilter;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

/// Log levels for the Vortex library.
#[repr(C)]
#[allow(non_camel_case_types)]
pub enum vx_log_level {
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
pub unsafe extern "C-unwind" fn vx_init_logging(level: vx_log_level) {
    let filter = match level {
        vx_log_level::LOG_LEVEL_OFF => LevelFilter::Off,
        vx_log_level::LOG_LEVEL_ERROR => LevelFilter::Error,
        vx_log_level::LOG_LEVEL_WARN => LevelFilter::Warn,
        vx_log_level::LOG_LEVEL_INFO => LevelFilter::Info,
        vx_log_level::LOG_LEVEL_DEBUG => LevelFilter::Debug,
        vx_log_level::LOG_LEVEL_TRACE => LevelFilter::Trace,
    };

    TermLogger::init(
        filter,
        Config::default(),
        TerminalMode::Stderr,
        ColorChoice::Auto,
    )
    .ok();
}
