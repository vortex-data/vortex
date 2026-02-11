// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar value conversion between Vortex and ClickHouse.

use vortex::error::{VortexResult, vortex_bail};
use vortex::scalar::Scalar;

/// Convert a ClickHouse value to a Vortex Scalar.
pub fn clickhouse_value_to_vortex(_value_ptr: *const std::ffi::c_void) -> VortexResult<Scalar> {
    vortex_bail!("ClickHouse value to Vortex scalar conversion not yet implemented")
}

/// Convert a Vortex Scalar to a ClickHouse value.
pub fn vortex_to_clickhouse_value(_scalar: &Scalar) -> VortexResult<*mut std::ffi::c_void> {
    vortex_bail!("Vortex scalar to ClickHouse value conversion not yet implemented")
}
