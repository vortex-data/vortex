// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for configuring Vortex logging.
//!
//! The hand-written C ABI exposed the `#[repr(C)]` enum `vx_log_level` and the free function
//! `vx_set_log_level`. The Diplomat port keeps the same six levels as a plain Diplomat enum and
//! exposes `set_log_level` as a static method. As in the original, the stderr logger is installed
//! only on the first call.

#[diplomat::bridge]
pub mod ffi {
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use tracing::level_filters::LevelFilter;
    use tracing_subscriber::EnvFilter;

    static LOGGER_INIT: AtomicBool = AtomicBool::new(false);

    /// Log levels for the Vortex library, in increasing verbosity.
    ///
    /// Equivalent to the C ABI `vx_log_level` enum. Diplomat assigns its own representation per
    /// target language; unlike the C ABI the discriminant values are not part of a stable wire
    /// contract.
    pub enum VxLogLevel {
        /// No logging will be performed.
        Off,
        /// Only error messages will be logged.
        Error,
        /// Warnings and error messages will be logged.
        Warn,
        /// Informational messages, warnings, and error messages will be logged.
        Info,
        /// Debug messages, informational messages, warnings, and error messages will be logged.
        Debug,
        /// All messages, including trace messages, will be logged.
        Trace,
    }

    impl VxLogLevel {
        /// Convert this level to a tracing [`LevelFilter`].
        fn to_level_filter(self) -> LevelFilter {
            match self {
                VxLogLevel::Off => LevelFilter::OFF,
                VxLogLevel::Error => LevelFilter::ERROR,
                VxLogLevel::Warn => LevelFilter::WARN,
                VxLogLevel::Info => LevelFilter::INFO,
                VxLogLevel::Debug => LevelFilter::DEBUG,
                VxLogLevel::Trace => LevelFilter::TRACE,
            }
        }

        /// Set the stderr logger to output at the specified level.
        ///
        /// Replaces the C ABI `vx_set_log_level`. The logger is installed only on the first call;
        /// subsequent calls are no-ops.
        pub fn set_log_level(level: VxLogLevel) {
            if !LOGGER_INIT.fetch_or(true, Ordering::SeqCst) {
                let filter = EnvFilter::builder()
                    .with_default_directive(level.to_level_filter().into())
                    .parse_lossy("");

                tracing_subscriber::fmt()
                    .compact()
                    .with_writer(std::io::stderr)
                    .with_env_filter(filter)
                    .init();
            }
        }
    }
}
