// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;

use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;
use crate::types::vector::AnyVector;

#[derive(Debug)]
pub struct L2DenormScheme;

impl Scheme for L2DenormScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.tensor.l2_denorm"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let Canonical::Extension(ext) = canonical else {
            return false;
        };

        // `AnyVector` is the strict matcher for plain `Vector` only, so a `NormalizedVector`
        // input is naturally excluded here (it would already carry an authoritative unit-norm
        // representation and does not need re-normalization).
        ext.ext_dtype().is::<AnyVector>()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // We almost always want to pre-normalize our data if the vector is not already normalized.
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let l2_denorm = normalize_as_l2_denorm(data.array().clone(), exec_ctx)?;
        Ok(l2_denorm.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::EmptyMetadata;
    use vortex_array::validity::Validity;
    use vortex_compressor::scheme::Scheme;
    use vortex_error::VortexResult;

    use super::L2DenormScheme;
    use crate::types::fixed_shape::FixedShapeTensor;
    use crate::types::fixed_shape::FixedShapeTensorMetadata;
    use crate::types::vector::Vector;

    fn fsl_storage(elements: &[f32], list_size: u32) -> VortexResult<FixedSizeListArray> {
        let len = elements.len() / list_size as usize;
        let elements = PrimitiveArray::from_iter(elements.iter().copied()).into_array();
        FixedSizeListArray::try_new(elements, list_size, Validity::NonNullable, len)
    }

    #[test]
    fn matches_vector() -> VortexResult<()> {
        let fsl = fsl_storage(&[1.0, 0.0], 2)?;
        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
        let canonical = Canonical::Extension(ExtensionArray::new(ext_dtype, fsl.into_array()));

        assert!(L2DenormScheme.matches(&canonical));
        Ok(())
    }

    #[test]
    fn rejects_fixed_shape_tensor() -> VortexResult<()> {
        let fsl = fsl_storage(&[1.0, 0.0, 0.0, 1.0], 4)?;
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            4,
            Nullability::NonNullable,
        );
        let ext_dtype = ExtDType::<FixedShapeTensor>::try_new(
            FixedShapeTensorMetadata::new(vec![2, 2]),
            storage_dtype,
        )?
        .erased();
        let canonical = Canonical::Extension(ExtensionArray::new(ext_dtype, fsl.into_array()));

        assert!(!L2DenormScheme.matches(&canonical));
        Ok(())
    }
}
