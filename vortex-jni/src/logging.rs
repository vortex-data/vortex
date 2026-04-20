// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use jni::EnvUnowned;
use jni::objects::JClass;
use jni::sys::jint;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

// Ensure the logger is initialized only once
static LOGGER_INIT: AtomicBool = AtomicBool::new(false);

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeLogging_initLogging(
    _env: EnvUnowned,
    _class: JClass,
    level: jint,
) {
    if !LOGGER_INIT.fetch_or(true, Ordering::SeqCst) {
        let level = match level {
            0 => LevelFilter::ERROR,
            1 => LevelFilter::WARN,
            2 => LevelFilter::INFO,
            3 => LevelFilter::DEBUG,
            4 => LevelFilter::TRACE,
            _ => LevelFilter::OFF,
        };

        let filter = EnvFilter::builder()
            .with_default_directive(level.into())
            .parse_lossy("");

        tracing_subscriber::fmt()
            .compact()
            .with_writer(std::io::stderr)
            .with_env_filter(filter)
            .init();
    }
}
