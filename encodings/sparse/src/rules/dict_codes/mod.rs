// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
        if child_idx != 0 || !array.fill_scalar().is_null() || !supports_sparse(parent.dtype()) {
            return Ok(None);
        }
        // Keep the output positions and slice offset intact. Repeated patch codes remain a
        // dictionary over only the present rows, rather than expanding codes to the full length.
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
        if child_idx == 0 {
            return self.reduce_parent(array, parent, child_idx);
        }
        TakeExecuteAdaptor(Sparse).execute_parent(array, parent, child_idx, ctx)
    }
}

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
        // Slice the sparse positions before the dictionary requests dense primitive codes.
        let Some(sliced) = <Sparse as SliceKernel>::slice(codes, array.slice_range().clone(), ctx)?
        else {
            return Ok(None);
        };
        if let Some(codes) = sliced.as_opt::<Sparse>() {
            return SparseCodes.reduce_parent(codes, parent, child_idx);
        }
        Ok(Some(parent.values().take(sliced)?))
    }
}

fn supports_sparse(dtype: &DType) -> bool {
    // Do not turn a supported dictionary into an unsupported sparse canonicalization. Struct
    // canonicalization pushes sparse arrays into its fields, so check descendants as well.
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
