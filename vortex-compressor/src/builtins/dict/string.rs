// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF8-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::builtins::dict::string_candidate::CachedStringDictionary;
use crate::builtins::dict::string_candidate::string_dictionary_estimate;
use crate::scheme::ChildSelection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;

/// Dictionary encoding for low-cardinality string values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StringDictScheme;

impl Scheme for StringDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    /// Children: values=0, codes=1.
    fn num_children(&self) -> usize {
        2
    }

    /// String dict codes (child 1) are compact unsigned integers that should not be dict-encoded
    /// again.
    ///
    /// Additional exclusions for codes (IntSequenceScheme, FoRScheme, ZigZagScheme, SparseScheme,
    /// RunEndScheme, RLE, etc.) are expressed as pull rules on those schemes in `vortex-btrblocks`.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(1),
        }]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        string_dictionary_estimate()
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if let Some(candidate) = data.get::<CachedStringDictionary>() {
            return Ok(candidate.array().clone());
        }

        let dict = dict_encode(data.array(), exec_ctx)?;
        compress_dictionary(compressor, &dict, compress_ctx, exec_ctx)
    }
}

/// Compresses the value and code children of a dictionary candidate.
pub(super) fn compress_dictionary(
    compressor: &CascadingCompressor,
    dict: &DictArray,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Values = child 0.
    let compressed_values = compressor.compress_child(
        dict.values(),
        &compress_ctx,
        StringDictScheme.id(),
        0,
        exec_ctx,
    )?;

    // Codes = child 1.
    let narrowed_codes = dict
        .codes()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?
        .into_array();
    let compressed_codes = compressor.compress_child(
        &narrowed_codes,
        &compress_ctx,
        StringDictScheme.id(),
        1,
        exec_ctx,
    )?;

    // SAFETY: compressing codes or values does not alter the invariants.
    unsafe {
        Ok(
            DictArray::new_unchecked(compressed_codes, compressed_values)
                .set_all_values_referenced(dict.has_all_values_referenced())
                .into_array(),
        )
    }
}
