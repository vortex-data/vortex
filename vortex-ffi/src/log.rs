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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_conversion() {
        unsafe {
            // Test all log levels
            vx_set_log_level(vx_log_level::LOG_LEVEL_OFF);
            assert_eq!(log::max_level(), LevelFilter::Off);

            vx_set_log_level(vx_log_level::LOG_LEVEL_ERROR);
            assert_eq!(log::max_level(), LevelFilter::Error);

            vx_set_log_level(vx_log_level::LOG_LEVEL_WARN);
            assert_eq!(log::max_level(), LevelFilter::Warn);

            vx_set_log_level(vx_log_level::LOG_LEVEL_INFO);
            assert_eq!(log::max_level(), LevelFilter::Info);

            vx_set_log_level(vx_log_level::LOG_LEVEL_DEBUG);
            assert_eq!(log::max_level(), LevelFilter::Debug);

            vx_set_log_level(vx_log_level::LOG_LEVEL_TRACE);
            assert_eq!(log::max_level(), LevelFilter::Trace);

            // Reset to off
            vx_set_log_level(vx_log_level::LOG_LEVEL_OFF);
        }
    }

    #[test]
    fn test_log_level_enum_values() {
        // Important: These values are part of the FFI ABI contract
        assert_eq!(vx_log_level::LOG_LEVEL_OFF as i32, 0);
        assert_eq!(vx_log_level::LOG_LEVEL_ERROR as i32, 1);
        assert_eq!(vx_log_level::LOG_LEVEL_WARN as i32, 2);
        assert_eq!(vx_log_level::LOG_LEVEL_INFO as i32, 3);
        assert_eq!(vx_log_level::LOG_LEVEL_DEBUG as i32, 4);
        assert_eq!(vx_log_level::LOG_LEVEL_TRACE as i32, 5);
    }
}
