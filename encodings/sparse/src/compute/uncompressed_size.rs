// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes_u64;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::validity_uncompressed_size_in_bytes;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::Sparse;
use crate::SparseExt;

#[derive(Debug)]
pub(crate) struct SparseUncompressedSizeInBytesKernel;

impl DynAggregateKernel for SparseUncompressedSizeInBytesKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<UncompressedSizeInBytes>() {
            return Ok(None);
        }

        // We only want to use this kernel for variable length types
        if batch.dtype().element_size().is_some() {
            return Ok(None);
        }

        let Some(sparse) = batch.as_opt::<Sparse>() else {
            return Ok(None);
        };

        let patches = sparse.patches();
        let n_fill = sparse.len() - patches.num_patches();

        let base = u64::try_from(sparse.fill_scalar().approx_nbytes() * (n_fill))
            .map_err(|_| vortex_err!("sparse fill size overflow"))?;
        let patches = uncompressed_size_in_bytes_u64(patches.values(), ctx)?;
        let validity = validity_uncompressed_size_in_bytes(
            sparse
                .as_ref()
                .validity()?
                .execute_mask(sparse.len(), ctx)?,
        )?;

        let total = base
            .checked_add(patches)
            .and_then(|v| v.checked_add(validity))
            .ok_or_else(|| vortex_err!("total size overflow"))?;

        Ok(Some(Scalar::primitive(total, Nullability::Nullable)))
    }
}
