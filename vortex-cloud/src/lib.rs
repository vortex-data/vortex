// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cloud object store integration for Vortex.
//!
//! This crate owns everything Vortex needs to turn a URL into a credentialed
//! [`object_store::ObjectStore`]:
//!
//! * [`Registry`] resolves a URL to a store, caching one client per bucket. It reads configuration
//!   out of environment variables case-insensitively, matching how the `object_store` `from_env`
//!   builders behave.
//! * [`opendal`] supplies stores for cloud services the `object_store` crate does not implement
//!   natively — Tencent Cloud COS and Alibaba Cloud OSS — bridged through
//!   `object_store_opendal`.
//!
//! Every Vortex language binding resolves URLs through this one crate, so a scheme added here is
//! reachable from Python, Java and DuckDB alike.
//!
//! # Cargo features
//!
//! * `registry` — the [`Registry`], plus the natively-supported cloud backends (S3, Azure, GCS,
//!   HTTP) it resolves URLs to.
//! * `cos` — Tencent Cloud COS, the `cos://` scheme.
//! * `oss` — Alibaba Cloud OSS, the `oss://` scheme.
//! * `opendal` — every OpenDAL-backed service above.
//!
//! The `registry` feature picks up whichever OpenDAL services are enabled, so a consumer that
//! turns on `oss` gets `oss://` resolution without touching its own scheme matching.

#[cfg(any(feature = "cos", feature = "oss"))]
pub mod opendal;
#[cfg(feature = "registry")]
mod registry;

#[cfg(feature = "registry")]
pub use registry::Registry;
