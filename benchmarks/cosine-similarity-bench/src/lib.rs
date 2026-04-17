//! See `main.rs` for the binary entry points. This library crate re-exports
//! the modules so Criterion benches (and any future integrations) can link
//! directly against the kernel.
#![allow(clippy::too_many_arguments)]

pub mod generate;
pub mod kernel;
pub mod metrics;
pub mod scan_local;
pub mod scan_s3;
