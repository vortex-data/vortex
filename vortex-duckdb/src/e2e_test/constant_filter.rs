// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Owns native equality filters for static table-filter tests.
//!
//! The native constructor copies a non-null DuckDB literal. The owner releases that copy and the
//! filter together, and only lends the filter to the converter while both remain alive.

use std::ptr::NonNull;

use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::TableFilter;
use crate::duckdb::TableFilterRef;
use crate::duckdb::ValueRef;

unsafe extern "C" {
    fn duckdb_vx_test_constant_equality_filter(
        value: cpp::duckdb_value,
    ) -> cpp::duckdb_vx_table_filter;
    fn duckdb_vx_test_destroy_constant_equality_filter(filter: cpp::duckdb_vx_table_filter);
}

/// Owns a native equality filter and the literal copied into it.
pub(super) struct ConstantEqualityFilter {
    /// A live filter allocated by the native constructor and owned only by this guard.
    filter: NonNull<cpp::duckdb_vx_table_filter_>,
}

impl ConstantEqualityFilter {
    /// Copies a non-null literal, returning an error when the native constructor rejects it.
    pub(super) fn try_new(value: &ValueRef) -> VortexResult<Self> {
        // SAFETY: The constructor copies this live ValueRef and catches all native exceptions.
        let filter = unsafe { duckdb_vx_test_constant_equality_filter(value.as_ptr()) };
        let filter = NonNull::new(filter)
            .ok_or_else(|| vortex_err!("DuckDB constant equality filter construction failed"))?;

        Ok(Self { filter })
    }

    /// Borrows the filter for conversion without transferring ownership.
    pub(super) fn as_filter(&self) -> &TableFilterRef {
        // SAFETY: The constructor allocated this filter, and the returned borrow cannot outlive it.
        unsafe { TableFilter::borrow(self.filter.as_ptr()) }
    }
}

impl Drop for ConstantEqualityFilter {
    fn drop(&mut self) {
        // SAFETY: This guard uniquely owns the filter allocated by the matching constructor.
        unsafe { duckdb_vx_test_destroy_constant_equality_filter(self.filter.as_ptr()) }
    }
}
