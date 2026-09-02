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
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::FSSTArraySlotsExt;
use vortex_fsst::FSSTSymbolTable;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// FSST (Fast Static Symbol Table) compression.
///
/// One of the two string-fragmentation schemes in the default
/// [`crate::ALL_SCHEMES`] (alongside `OnPairScheme`); the sample-based selector
/// keeps whichever is smaller per column. FSST compresses faster, OnPair
/// usually wins on ratio.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FSSTScheme;

/// Number of values in the tiny sample used to gate FSST on binary columns.
const BINARY_SAMPLE_VALUES: usize = 64;

/// Minimum fraction of value bytes FSST must save over plain VarBin storage on the sample
/// before it is allowed into scheme selection for a binary column.
const BINARY_MIN_SAVINGS: f64 = 0.15;

/// Gates FSST on binary columns with a tiny trial compression.
///
/// Strings almost always have intra-value structure, but arbitrary binary payloads often do
/// not (hashes, ciphertexts, compressed blobs), and a symbol table that buys nothing still
/// costs training on write and decoding on read. Trial-compress up to
/// [`BINARY_SAMPLE_VALUES`] values strided across the column and admit FSST only when its
/// code bytes undercut the raw value bytes (the VarBin baseline, views excluded) by more than
/// [`BINARY_MIN_SAVINGS`]. The symbol table is excluded from the measurement: its size is
/// fixed per column and amortizes away at real column lengths, while against a 64-value
/// sample it would drown out the signal.
///
/// Once past the gate, the ratio entered into selection comes from the compressor's standard
/// sampling estimator, so FSST competes with other schemes on the same measurement basis.
fn estimate_binary_fsst(
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateVerdict> {
    let view = data.array_as_varbinview();
    let len = view.len();
    if len == 0 {
        return Ok(EstimateVerdict::Skip);
    }
    let mask = view.validity()?.execute_mask(len, exec_ctx)?;

    let step = len.div_ceil(BINARY_SAMPLE_VALUES);
    let sample = VarBinArray::from_iter(
        (0..len)
            .step_by(step)
            .map(|i| mask.value(i).then(|| view.bytes_at(i).as_slice().to_vec())),
        view.dtype().clone(),
    );
    let raw_nbytes = sample.bytes().len();
    if raw_nbytes == 0 {
        return Ok(EstimateVerdict::Skip);
    }

    let sample = sample.into_array();
    let trained = fsst_train_compressor(&sample, exec_ctx)?;
    let fsst = fsst_compress(&sample, &trained, exec_ctx)?;
    let compressed_nbytes = fsst.codes().bytes().len();

    if (compressed_nbytes as f64) >= (1.0 - BINARY_MIN_SAVINGS) * raw_nbytes as f64 {
        return Ok(EstimateVerdict::Skip);
    }

    let score =
        compressor.estimate_by_sampling(&FSSTScheme, data.array(), compress_ctx, exec_ctx)?;
    Ok(match score.finite_ratio() {
        Some(ratio) => EstimateVerdict::Ratio(ratio),
        None => EstimateVerdict::Skip,
    })
}

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

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if data.array().dtype().is_binary() {
            return CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |compressor, data, _best_so_far, compress_ctx, exec_ctx| {
                    estimate_binary_fsst(compressor, data, compress_ctx, exec_ctx)
                },
            )));
        }
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let utf8 = data.array_as_varbinview().into_owned().into_array();
        let compressor_fsst = fsst_train_compressor(&utf8, exec_ctx)?;
        let fsst = fsst_compress(&utf8, &compressor_fsst, exec_ctx)?;

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

        // Reuse the padded symbol table as-is; only the codes and lengths change here.
        let fsst = FSST::try_new_with_symbol_table(
            fsst.dtype().clone(),
            Arc::new(FSSTSymbolTable::new_padded(
                fsst.padded_symbols().clone(),
                fsst.padded_symbol_lengths().clone(),
                fsst.n_symbols(),
            )?),
            compressed_codes,
            compressed_original_lengths,
            exec_ctx,
        )?;

        Ok(fsst.into_array())
    }
}
