// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 layout plan machinery.
//!
//! This module contains the layout-tree expansion vtables and executable
//! [`ScanPlan`](crate::scan::plan::ScanPlan) plans used by the alternate scan implementation.

pub mod session;

pub(crate) mod layouts;
mod row_idx;
pub use row_idx::with_row_idx;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::immediate_access::immediate_scope_access;
use vortex_array::extension::datetime::AnyTemporal;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

pub use crate::scan::plan::evidence;
pub use crate::scan::plan::request;

/// Environment variable selecting the file scan implementation.
///
/// Accepted values:
///
/// - `v1`, `scan`, `scan_builder`, `scan-builder`, `layout-reader`, or unset: use the
///   existing LayoutReader-based scan.
/// - `v2` or `scan2`: use the scan2
///   [`ScanPlan`](crate::scan::plan::ScanPlan) implementation.
pub const SCAN_IMPL_ENV: &str = "VORTEX_SCAN_IMPL";

/// Returns whether the scan2 implementation should be used by scan data sources.
pub fn scan2_enabled() -> VortexResult<bool> {
    match std::env::var(SCAN_IMPL_ENV) {
        Ok(value) if value.is_empty() => Ok(false),
        Ok(value) => parse_scan_impl(&value),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(value)) => {
            vortex_bail!("{SCAN_IMPL_ENV} must be valid unicode, got {value:?}")
        }
    }
}

fn parse_scan_impl(value: &str) -> VortexResult<bool> {
    match value {
        "v1" | "scan" | "scan_builder" | "scan-builder" | "layout-reader" => Ok(false),
        "v2" | "scan2" => Ok(true),
        other => vortex_bail!(
            "{SCAN_IMPL_ENV} must be one of v1, scan, scan_builder, scan-builder, layout-reader, v2, or scan2, got {other:?}"
        ),
    }
}

pub(crate) fn referenced_fields(expr: &Expression, scope: &StructFields) -> Vec<FieldName> {
    let mut fields: Vec<FieldName> = immediate_scope_access(expr, scope).into_iter().collect();
    fields.sort();
    fields
}

pub(crate) fn struct_fields(dtype: &DType) -> VortexResult<StructFields> {
    dtype
        .as_struct_fields_opt()
        .cloned()
        .ok_or_else(|| vortex_err!("scan2 expected struct dtype, got {dtype}"))
}

/// Validates temporal comparisons before scan2 pushdown.
pub fn validate_temporal_comparisons(expr: &Expression, scope: &DType) -> VortexResult<()> {
    for child in expr.children().iter() {
        validate_temporal_comparisons(child, scope)?;
    }

    let Some(operator) = expr.as_opt::<Binary>() else {
        return Ok(());
    };
    if !operator.is_comparison() {
        return Ok(());
    }

    let lhs = expr.child(0).return_dtype(scope)?;
    let rhs = expr.child(1).return_dtype(scope)?;
    if is_temporal(&lhs) && is_temporal(&rhs) && !lhs.eq_ignore_nullability(&rhs) {
        vortex_bail!("Cannot compare temporal DTypes with different metadata: {lhs} and {rhs}");
    }

    Ok(())
}

fn is_temporal(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(ext) => ext.metadata_opt::<AnyTemporal>().is_some(),
        _ => false,
    }
}
