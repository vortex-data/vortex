// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cloud object store integration for Vortex.
//!
//! This crate owns everything Vortex needs to turn a URL into a credentialed
//! [`object_store::ObjectStore`]:
//!
//! * `Registry` resolves a URL to a store, caching one client per bucket. It reads configuration
//!   out of environment variables case-insensitively, matching how the `object_store` `from_env`
//!   builders behave.
//! * `opendal` supplies stores for cloud services the `object_store` crate does not implement
//!   natively — Tencent Cloud COS, Alibaba Cloud OSS, and Tencent Cloud GooseFS — bridged
//!   through `object_store_opendal`.
//! * `hf` serves Hugging Face Hub repositories over `object_store`'s HTTP store, adding no cloud
//!   SDK of its own.
//!
//! Every Vortex language binding resolves URLs through this one crate, so a scheme added here is
//! reachable from Python, Java and DuckDB alike.
//!
//! Both are feature-gated, so this overview refers to them by name rather than by link: an
//! intra-doc link would dangle in a build that leaves the corresponding feature off.
//!
//! # Cargo features
//!
//! * `registry` — the `Registry`, plus the natively-supported cloud backends (S3, Azure, GCS,
//!   HTTP) it resolves URLs to.
//! * `hf` — the Hugging Face Hub, the `hf://` scheme.
//! * `cos` — Tencent Cloud COS, the `cos://` scheme.
//! * `oss` — Alibaba Cloud OSS, the `oss://` scheme.
//! * `goosefs` — Tencent Cloud GooseFS, the `goosefs://` scheme.
//! * `opendal` — every OpenDAL-backed service above.
//!
//! The `registry` feature picks up whichever services are enabled, so a consumer that turns on
//! `oss` gets `oss://` resolution without touching its own scheme matching.

#[cfg(feature = "hf")]
pub mod hf;
#[cfg(any(feature = "cos", feature = "goosefs", feature = "oss"))]
pub mod opendal;
#[cfg(feature = "registry")]
mod registry;

#[cfg(feature = "registry")]
pub use registry::Registry;

/// Parse a boolean the way `object_store` does: `1`/`true`/`on`/`yes`/`y` and their negatives,
/// case-insensitively. `None` for anything else.
#[cfg(any(feature = "cos", feature = "oss", feature = "hf"))]
pub(crate) fn parse_object_store_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" | "y" => Some(true),
        "0" | "false" | "off" | "no" | "n" => Some(false),
        _ => None,
    }
}
