use std::sync::OnceLock;

use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::jint;
use log::LevelFilter;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

// Ensure the logger is initialized only once
static LOGGER_INIT: OnceLock<()> = OnceLock::new();

#[allow(clippy::expect_used)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeLogging_initLogging(
    _env: JNIEnv,
    _class: JClass,
    level: jint,
) {
    LOGGER_INIT.get_or_init(|| {
        let level = match level {
            0 => LevelFilter::Error,
            1 => LevelFilter::Warn,
            2 => LevelFilter::Info,
            3 => LevelFilter::Debug,
            4 => LevelFilter::Trace,
            _ => LevelFilter::Off,
        };

        TermLogger::init(
            level,
            Config::default(),
            TerminalMode::Stderr,
            ColorChoice::Auto,
        )
        .expect("Failed to initialize logger");
    });
}
