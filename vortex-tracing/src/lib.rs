// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tracing wrappers for Vortex's I/O, segment, and layout trait boundaries.
//!
//! Each wrapper is opt-in: construct it at the point where the inner trait
//! object is built, and every call through the wrapped trait produces a
//! [`tracing`] span with structured fields (byte offsets, segment ids, row
//! ranges, expressions, durations).
//!
//! # Zero-cost opt-out
//!
//! If a wrapper is not constructed, there is no runtime cost: no span is
//! opened, no field is recorded, and the compiler can inline the original
//! trait call unchanged.
//!
//! When a wrapper *is* constructed, the cost per call is one atomic load and a
//! branch per span (the callsite cache inside [`tracing`]). For truly zero
//! overhead on release builds, enable one of `tracing`'s compile-time level
//! filters at the binary crate, for example:
//!
//! ```toml
//! tracing = { version = "0.1", features = ["release_max_level_info"] }
//! ```
//!
//! # Typical use
//!
//! ```ignore
//! use std::sync::Arc;
//! use vortex_tracing::TracingReadAt;
//!
//! let io = Arc::new(TracingReadAt::new(raw_io));
//! let file = session.open_options().open(io).await?;
//! ```
//!
//! The [`TracingSegmentSource`] wrapper is applied at the point where the
//! shared segment source is built inside `vortex-file`, and
//! [`TracingLayoutReader`] wraps the root layout reader inside the scan
//! builder.
//!
//! # Targets
//!
//! The wrappers emit spans under the following [`tracing`] targets so that
//! categories can be enabled and disabled independently via
//! `RUST_LOG=vortex_tracing::io=trace,vortex_tracing::layout=info`:
//!
//! - `vortex_tracing::io` — physical reads ([`TracingReadAt`])
//! - `vortex_tracing::segment` — logical segment requests ([`TracingSegmentSource`])
//! - `vortex_tracing::layout` — layout evaluation ([`TracingLayoutReader`])

mod io;
mod layout;
mod segment;
#[cfg(feature = "perfetto")]
pub mod subscriber;

pub use io::TracingReadAt;
pub use layout::TracingLayoutReader;
pub use segment::TracingSegmentSource;

/// Tracing target for physical reads.
pub const TARGET_IO: &str = "vortex_tracing::io";
/// Tracing target for logical segment requests.
pub const TARGET_SEGMENT: &str = "vortex_tracing::segment";
/// Tracing target for layout evaluation.
pub const TARGET_LAYOUT: &str = "vortex_tracing::layout";
