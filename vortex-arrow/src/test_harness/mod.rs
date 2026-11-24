// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Independent Arrow conformance checks and nested transport assertions.
//!
//! Plugin crates enable `_test-harness-hegel` and supply an independently built canonical Arrow
//! field and array to `check_generated`. This checks both conversion directions, field inference,
//! default export, and round trips through nested arrays. Use [`check_arrow`] for additional
//! explicit targets. Fuzz targets use [`check_containers`] with `_test-harness`, so they do not link
//! Hegel's native backend under libFuzzer coverage flags.
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
pub use generate::check_generated_containers;
#[cfg(feature = "_test-harness-hegel")]
pub use generate::draw_storage_case;

mod round_trip;
pub use round_trip::check_arrow;
pub use round_trip::check_containers;

#[cfg(all(test, feature = "_test-harness-hegel"))]
mod temporal;
