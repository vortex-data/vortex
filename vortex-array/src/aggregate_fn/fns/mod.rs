// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::match_each_unsigned_integer_ptype;

/// Resolve a list `offsets`/`sizes` child to a `Vec<usize>` in a single pass, avoiding the
/// per-index `execute_scalar` fallback that `offset_at`/`size_at` take for non-primitive children.
pub(crate) fn list_offsets_to_usize(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<usize>> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    let primitive = primitive.reinterpret_cast(primitive.ptype().to_unsigned());
    Ok(match_each_unsigned_integer_ptype!(primitive.ptype(), |P| {
        primitive
            .as_slice::<P>()
            .iter()
            .map(|v| (*v).as_())
            .collect()
    }))
}

pub mod all_nan;
pub mod all_non_distinct;
pub mod all_non_nan;
pub mod all_non_null;
pub mod all_null;
pub mod bounded_max;
pub mod bounded_min;
pub mod count;
pub mod first;
pub mod is_constant;
pub mod is_sorted;
pub mod last;
pub mod max;
pub mod mean;
pub mod min;
pub mod min_max;
pub mod nan_count;
pub mod null_count;
pub mod sum;
pub mod uncompressed_size_in_bytes;
