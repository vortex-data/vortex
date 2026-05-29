// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST binary compression.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// FSST compression for binary values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BinaryFSSTScheme;

impl Scheme for BinaryFSSTScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.binary.fsst"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_binary()
    }

    /// Children: lengths=0, code_offsets=1.
    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let binary = data.array_as_varbinview().into_owned();
        let compressor_fsst = fsst_train_compressor(&binary);
        let fsst = fsst_compress(
            &binary,
            binary.len(),
            binary.dtype(),
            &compressor_fsst,
            exec_ctx,
        );

        let uncompressed_lengths_primitive = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_original_lengths = compressor.compress_child(
            &uncompressed_lengths_primitive.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        let codes_offsets_primitive = fsst
            .codes()
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_codes_offsets = compressor.compress_child(
            &codes_offsets_primitive.into_array(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;
        let compressed_codes = VarBinArray::try_new(
            compressed_codes_offsets,
            fsst.codes().bytes().clone(),
            fsst.codes().dtype().clone(),
            fsst.codes().validity()?,
        )?;

        let fsst = FSST::try_new(
            fsst.dtype().clone(),
            fsst.symbols().clone(),
            fsst.symbol_lengths().clone(),
            compressed_codes,
            compressed_original_lengths,
            exec_ctx,
        )?;

        Ok(fsst.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_fsst::FSST;
    use vortex_session::VortexSession;

    use crate::BtrBlocksCompressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn binary_fsst_data() -> VarBinViewArray {
        VarBinViewArray::from_iter(
            (0..1024).map(|idx| {
                Some(format!("variant-key-{idx:04}-invoice-total-line-items").into_bytes())
            }),
            DType::Binary(Nullability::NonNullable),
        )
    }

    #[test]
    fn default_compressor_uses_fsst_for_binary_data() -> VortexResult<()> {
        let array = binary_fsst_data().into_array();
        let compressed =
            BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;

        assert!(
            compressed.is::<FSST>(),
            "expected binary data to be FSST-compressed, got {}",
            compressed.encoding_id(),
        );
        assert!(compressed.nbytes() < array.nbytes());

        let decompressed =
            compressed.execute::<vortex_array::ArrayRef>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(array, decompressed);

        Ok(())
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn compact_compressor_uses_zstd_for_binary_data() -> VortexResult<()> {
        let array = binary_fsst_data().into_array();
        let compressed = crate::BtrBlocksCompressorBuilder::default()
            .with_compact()
            .build()
            .compress(&array, &mut SESSION.create_execution_ctx())?;

        assert!(
            compressed.is::<vortex_zstd::Zstd>(),
            "expected compact binary data to be Zstd-compressed, got {}",
            compressed.encoding_id(),
        );
        assert!(compressed.nbytes() < array.nbytes());

        let decompressed =
            compressed.execute::<vortex_array::ArrayRef>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(array, decompressed);

        Ok(())
    }
}
