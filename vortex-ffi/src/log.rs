// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

static LOGGER_INIT: AtomicBool = AtomicBool::new(false);

/// Convert a [`vx_log_level`] to a [`LevelFilter`].
fn to_level_filter(level: vx_log_level) -> LevelFilter {
    match level {
        vx_log_level::LOG_LEVEL_OFF => LevelFilter::OFF,
        vx_log_level::LOG_LEVEL_ERROR => LevelFilter::ERROR,
        vx_log_level::LOG_LEVEL_WARN => LevelFilter::WARN,
        vx_log_level::LOG_LEVEL_INFO => LevelFilter::INFO,
        vx_log_level::LOG_LEVEL_DEBUG => LevelFilter::DEBUG,
        vx_log_level::LOG_LEVEL_TRACE => LevelFilter::TRACE,
    }
}

/// Log levels for the Vortex library.
#[repr(C)]
#[expect(non_camel_case_types)]
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
/// The logger will only be installed on the first call.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_set_log_level(level: vx_log_level) {
    if !LOGGER_INIT.fetch_or(true, Ordering::SeqCst) {
        let filter = EnvFilter::builder()
            .with_default_directive(to_level_filter(level).into())
            .parse_lossy("");

        tracing_subscriber::fmt()
            .compact()
            .with_writer(std::io::stderr)
            .with_env_filter(filter)
            .init();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(
            to_level_filter(vx_log_level::LOG_LEVEL_OFF),
            LevelFilter::OFF
        );
        assert_eq!(
            to_level_filter(vx_log_level::LOG_LEVEL_ERROR),
            LevelFilter::ERROR
        );
        assert_eq!(
            to_level_filter(vx_log_level::LOG_LEVEL_WARN),
            LevelFilter::WARN
        );
        assert_eq!(
            to_level_filter(vx_log_level::LOG_LEVEL_INFO),
            LevelFilter::INFO
        );
        assert_eq!(
            to_level_filter(vx_log_level::LOG_LEVEL_DEBUG),
            LevelFilter::DEBUG
        );
        assert_eq!(
            to_level_filter(vx_log_level::LOG_LEVEL_TRACE),
            LevelFilter::TRACE
        );
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
