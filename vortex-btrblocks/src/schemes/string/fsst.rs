// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST (Fast Static Symbol Table) string compression.

use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArraySlotsExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::FSSTArraySlotsExt;
use vortex_fsst::FSSTSymbolTable;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;

/// FSST (Fast Static Symbol Table) compression.
///
/// One of the two string-fragmentation schemes in the default
/// [`crate::CompressionSession`] (alongside `OnPairScheme`); the sample-based selector
/// keeps whichever is smaller per column. FSST compresses faster, OnPair
/// usually wins on ratio.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FSSTScheme;

impl Scheme for FSSTScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.fsst"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8() || canonical.dtype().is_binary()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![FSST.id(), VarBin.id()]
    }

    /// Children: lengths=0, code_offsets=1.
    fn num_children(&self) -> usize {
        2
    }

    /// The symbol table: at most 255 symbols, however many rows it serves.
    fn fixed_nbytes(&self, compressed: &ArrayRef) -> u64 {
        compressed.as_opt::<FSST>().map_or(0, |fsst| {
            (size_of_val(fsst.symbols()) + size_of_val(fsst.symbol_lengths())) as u64
        })
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    /// Children: uncompressed lengths=0, codes offsets=1. The codes store the input's validity,
    /// which is registered.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let utf8 = data.array();
        let compressor_fsst = fsst_train_compressor(utf8, exec_ctx)?;
        let fsst = fsst_compress(utf8, &compressor_fsst, exec_ctx)?;

        let uncompressed_lengths = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let codes_offsets = fsst
            .codes()
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;

        // Reuse the padded symbol table as-is; only the codes and lengths change here.
        let symbol_table = Arc::new(FSSTSymbolTable::new_padded(
            fsst.padded_symbols().clone(),
            fsst.padded_symbol_lengths().clone(),
            fsst.n_symbols(),
        )?);
        let dtype = fsst.dtype().clone();
        let codes_bytes = fsst.codes().bytes().clone();
        let codes_dtype = fsst.codes().dtype().clone();
        Ok(Encoded::new(
            vec![
                Child::new(self, 0, uncompressed_lengths.into_array()),
                Child::new(self, 1, codes_offsets.into_array()),
            ],
            Vec::new(),
            move |children, exec_ctx| {
                let [uncompressed_lengths, codes_offsets] = children.compressed()?;
                let codes = VarBinArray::try_new(
                    codes_offsets,
                    codes_bytes,
                    codes_dtype,
                    children.validity()?,
                    exec_ctx,
                )?;
                Ok(FSST::try_new_with_symbol_table(
                    dtype,
                    symbol_table,
                    codes,
                    uncompressed_lengths,
                    exec_ctx,
                )?
                .into_array())
            },
        )
        .with_validity(fsst.codes().validity()?))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}
