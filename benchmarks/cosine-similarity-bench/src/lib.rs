//! See `main.rs` for the binary entry points. This library crate re-exports
//! the modules so Criterion benches (and any future integrations) can link
//! directly against the kernel.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::missing_safety_doc,
    unused_unsafe
)]

pub mod generate;
pub mod kernel;
pub mod metrics;
pub mod scan_local;
pub mod scan_s3;
