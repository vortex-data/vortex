// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Keep dictionary lookups over sparse codes proportional to the number of patches.
//!
//! ```text
//! Dict(Sparse(positions, codes, NULL), values)
//!     -> Sparse(positions, Dict(codes, values), NULL)
//! ```

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::slice::SliceArraySlotsExt;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::dtype::DType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt;

/// Moves a dictionary lookup into the patches of null-filled sparse codes.
#[derive(Debug)]
pub(crate) struct SparseCodes;

impl ArrayParentReduceRule<Sparse> for SparseCodes {
    type Parent = Dict;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Sparse>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Dictionary child 0 holds codes; child 1 holds values. A null code produces a null
        // output, so a null fill survives the rewrite without a dictionary lookup of its own.
        if child_idx != 0 || !array.fill_scalar().is_null() || !supports_sparse(parent.dtype()) {
            return Ok(None);
        }

        // Positions identify output rows; codes select dictionary entries and may repeat.
        // Look up only the patch codes, preserving the output positions and slice offset.
        let patches = array
            .patches()
            .map_values(|codes| parent.values().take(codes))?;

        Ok(Some(
            Sparse::try_new_from_patches(patches, Scalar::null(parent.dtype().clone()))?
                .into_array(),
        ))
    }
}

impl ExecuteParentKernel<Sparse> for SparseCodes {
    type Parent = Dict;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Sparse>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Apply the rewrite when sparse storage is the codes child. Sparse dictionary values
        // still need the existing take kernel, which gathers from those values using the codes.
        if child_idx == 0 {
            return self.reduce_parent(array, parent, child_idx);
        }

        TakeExecuteAdaptor(Sparse).execute_parent(array, parent, child_idx, ctx)
    }
}

/// Preserves the rewrite when a lazy slice sits between the dictionary and its sparse codes.
#[derive(Debug)]
pub(crate) struct SlicedSparseCodes;

impl ExecuteParentKernel<Slice> for SlicedSparseCodes {
    type Parent = Dict;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Slice>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 || array.is_empty() || !supports_sparse(parent.dtype()) {
            return Ok(None);
        }

        let Some(codes) = array.child().as_opt::<Sparse>() else {
            return Ok(None);
        };
        if !codes.fill_scalar().is_null() {
            return Ok(None);
        }

        // Executing the slice as primitive codes would erase the sparse representation before
        // the dictionary can see it. Slice the sparse positions first, then reuse the rewrite.
        let Some(sliced) = <Sparse as SliceKernel>::slice(codes, array.slice_range().clone(), ctx)?
        else {
            return Ok(None);
        };

        if let Some(codes) = sliced.as_opt::<Sparse>() {
            return SparseCodes.reduce_parent(codes, parent, child_idx);
        }

        // Slicing may collapse the codes to another encoding, such as a constant for a range
        // with no patches. Let an ordinary dictionary lookup handle that result.
        Ok(Some(parent.values().take(sliced)?))
    }
}

fn supports_sparse(dtype: &DType) -> bool {
    // Dictionary execution supports types that sparse canonicalization cannot decode. Only
    // rewrite when the resulting sparse array is supported all the way down: decoding a struct
    // distributes sparsity into its fields, and list elements can themselves contain structs.
    match dtype {
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        DType::List(element, _) | DType::FixedSizeList(element, ..) => supports_sparse(element),
        DType::Struct(fields, _) => fields.fields().all(|field| supports_sparse(&field)),
        _ => false,
    }
}

#[cfg(test)]
mod tests;
