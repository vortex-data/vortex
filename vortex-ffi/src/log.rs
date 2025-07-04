// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use log::LevelFilter;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

/// Log levels for the Vortex library.
#[repr(C)]
#[allow(non_camel_case_types)]
pub enum vx_log_level {
    /// No logging will be performed.
    LOG_LEVEL_OFF = 0,
    /// Only error messages will be logged.
    LOG_LEVEL_ERROR = 1,
    /// Warnings and error messages will be logged.
    LOG_LEVEL_WARN = 2,
    /// Informational messages, warnings, and error messages will be logged.
    LOG_LEVEL_INFO = 3,
    /// Debug messages, informational messages, warnings, and error messages will be logged.
    LOG_LEVEL_DEBUG = 4,
    /// All messages, including trace messages, will be logged.
    LOG_LEVEL_TRACE = 5,
}

/// Set the stderr logger to output at the specified level.
///
/// This function is optional, if it is not called then no logger will be installed.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_set_log_level(level: vx_log_level) {
    let level = match level {
        vx_log_level::LOG_LEVEL_OFF => LevelFilter::Off,
        vx_log_level::LOG_LEVEL_ERROR => LevelFilter::Error,
        vx_log_level::LOG_LEVEL_WARN => LevelFilter::Warn,
        vx_log_level::LOG_LEVEL_INFO => LevelFilter::Info,
        vx_log_level::LOG_LEVEL_DEBUG => LevelFilter::Debug,
        vx_log_level::LOG_LEVEL_TRACE => LevelFilter::Trace,
    };

    // Attempt to initialize the TermLogger if it is not already initialized.
    let _ = TermLogger::init(
        level,
        Config::default(),
        TerminalMode::Stderr,
        ColorChoice::Auto,
    );

    // In case the logger _was_ already initialized, we need to set the level again.
    log::set_max_level(level);
}
