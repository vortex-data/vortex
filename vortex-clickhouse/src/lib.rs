// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex ClickHouse Extension
//!
//! This crate provides ClickHouse integration for the Vortex columnar format.
//! It enables ClickHouse to read and write Vortex files with support for
//! predicate and projection pushdown.
//!
//! # Architecture
//!
//! The crate is organized into three layers:
//!
//! 1. **C++ Layer** (`cpp/`): Implements ClickHouse's `IInputFormat` and `IOutputFormat`
//!    interfaces, which are the entry points for the ClickHouse format system.
//!
//! 2. **FFI Boundary**: Uses `bindgen` to import ClickHouse types and `cbindgen` to
//!    export Rust functions to C++.
//!
//! 3. **Rust Core**: Implements the actual Vortex file reading/writing logic,
//!    type conversion, and query optimization features.
//!
//! # Usage
//!
//! ```sql
//! -- Read from Vortex file
//! SELECT * FROM file('data.vortex', 'Vortex');
//!
//! -- With predicate pushdown
//! SELECT * FROM file('data.vortex', 'Vortex') WHERE x > 100;
//! ```

#![allow(clippy::missing_safety_doc)]

use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex::dtype::session::DTypeSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::ext_types::{
    BigInt, ClickHouseDate, ClickHouseDateTime, ClickHouseEnum, ClickHouseFixedString,
    ClickHouseLowCardinality, Geo, IPAddress, UUID,
};

pub mod clickhouse;
pub mod convert;
mod copy;
pub mod error;
pub mod exporter;
pub mod ext_types;
mod scan;
mod utils;

#[rustfmt::skip]
#[path = "./cpp.rs"]
/// This module provides the FFI interface to our C++ code exposing additional functionality
/// for ClickHouse, such as custom data types and functions.
/// cbindgen:ignore
mod cpp;

#[cfg(test)]
mod e2e_test;

#[cfg(test)]
mod ffi_tests;

// A global runtime for Vortex operations within ClickHouse.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default().with_handle(RUNTIME.handle());
    // Register ClickHouse-specific extension types
    session.dtypes().register(BigInt);
    session.dtypes().register(Geo);
    session.dtypes().register(ClickHouseEnum);
    session.dtypes().register(ClickHouseDateTime);
    session.dtypes().register(ClickHouseDate);
    session.dtypes().register(ClickHouseLowCardinality);
    session.dtypes().register(ClickHouseFixedString);
    session.dtypes().register(UUID);
    session.dtypes().register(IPAddress);
    session
});

/// Get the global Vortex session used for ClickHouse operations.
pub fn session() -> &'static VortexSession {
    &SESSION
}

/// Get the global runtime handle for async operations.
pub fn runtime() -> &'static CurrentThreadRuntime {
    &RUNTIME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_initialization() {
        // Just verify that the lazy static initializes without panicking
        let _session = session();
        let _runtime = runtime();
    }
}
