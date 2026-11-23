// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Independent Arrow conformance checks and nested transport assertions.
//!
//! Owning crates supply independently built Arrow references to [`check_arrow`] and
//! [`check_canonical_field`]. [`check_containers`] checks top-level arrays and struct, list,
//! fixed-size-list, map value, dictionary, and run-end positions.
//! Plugin crates enable `_test-harness-hegel` to add shrinkable storage and nesting generation,
//! keeping their own extension metadata generators beside the types they own. Fuzz targets only
//! need `_test-harness`, so they do not link Hegel's native backend under libFuzzer coverage flags.
//!
//! Hegel suites run 100 cases with at most 32 rows and three extra nesting levels. Failures print
//! a replay blob for `#[hegel::reproduce_failure("...")]`. Variant transport needs the Parquet
//! Variant plugin, so its suite lives in `vortex-parquet-variant`.

#[cfg(feature = "_test-harness-hegel")]
mod generate;
#[cfg(feature = "_test-harness-hegel")]
pub use generate::StorageCase;
#[cfg(feature = "_test-harness-hegel")]
pub use generate::check_generated;
#[cfg(feature = "_test-harness-hegel")]
pub use generate::draw_storage_case;

mod round_trip;
pub use round_trip::check_arrow;
pub use round_trip::check_canonical_field;
pub use round_trip::check_containers;

#[cfg(all(test, feature = "_test-harness-hegel"))]
mod temporal;
