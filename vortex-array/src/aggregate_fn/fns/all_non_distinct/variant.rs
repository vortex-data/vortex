// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::VariantArray;

/// Checks whether two canonical variant arrays are element-wise non-distinct.
///
/// Variant values cannot be routed back through [`all_non_distinct`]: canonicalizing a variant
/// value array yields another canonical variant (with no shredded tree), which would recurse
/// forever. The generic fallback therefore compares logical variant scalars row-by-row. Encodings
/// that can compare their typed/value children more cheaply (e.g. `ParquetVariant`) register an
/// aggregate kernel that intercepts the comparison before it reaches this fallback.
///
/// [`all_non_distinct`]: super::all_non_distinct
pub(super) fn check_variant_identical(
    lhs: &VariantArray,
    rhs: &VariantArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if lhs.len() != rhs.len() {
        return Ok(false);
    }
    for idx in 0..lhs.len() {
        if lhs.execute_scalar(idx, ctx)? != rhs.execute_scalar(idx, ctx)? {
            return Ok(false);
        }
    }
    Ok(true)
}
