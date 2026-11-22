// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Offset-based storage for binary arrays.
//!
//! Canonical binary arrays are [`VarBinViewArray`](vortex_array::arrays::VarBinViewArray), which
//! spends a fixed 16 bytes per element on an opaque views buffer that no scheme can compress.
//! Re-encoding as [`VarBinArray`] replaces that buffer with an offsets child array, which the
//! cascading compressor can then compress with the ordinary integer schemes. For fixed-width
//! values the offsets are a constant-stride sequence and collapse to nothing.

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
use vortex_array::builders::VarBinBuilder;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;

/// Offset-based (rather than view-based) storage for binary arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VarBinScheme;

impl Scheme for VarBinScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.binary.varbin"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_binary()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![VarBin.id()]
    }

    fn top_level_encodings(&self) -> Vec<ArrayId> {
        vec![VarBin.id()]
    }

    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    /// Children: offsets=0. The builder rebuilds the input's validity, which is registered.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        // `append_to_builder` resolves the views slice and data buffers once and appends
        // borrowed slices into a single pre-sized allocation. Iterating the array per element
        // instead would clone a buffer handle and allocate for every value.
        let array = data.array();
        let mut builder = VarBinBuilder::<u64>::with_capacity_in(
            array.dtype().clone(),
            array.len(),
            exec_ctx.allocator(),
        );
        array.append_to_builder(&mut builder, exec_ctx)?;
        let varbin = builder.finish_into_varbin();

        let offsets = varbin
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?
            .into_array();
        let bytes = varbin.bytes().clone();
        let dtype = varbin.dtype().clone();
        Ok(Encoded::new(
            vec![Child::new(self, 0, offsets)],
            Vec::new(),
            move |children, exec_ctx| {
                let [offsets] = children.compressed()?;
                Ok(
                    VarBinArray::try_new(offsets, bytes, dtype, children.validity()?, exec_ctx)?
                        .into_array(),
                )
            },
        )
        .with_validity(varbin.validity()?))
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
