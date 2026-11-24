// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use vortex_array::ArrayId;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::builders::dict::dict_encode_by_frequency;
use vortex_error::VortexResult;

use super::bytes_candidate::bytes_dictionary_estimate;
use super::dictionary_encoded;
use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DescendantExclusion;
use crate::scheme::Encoded;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;

/// Dictionary encoding for low-cardinality binary values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BinaryDictScheme;

impl Scheme for BinaryDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.binary.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_binary()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    /// Children: values=0, codes=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Binary dict codes (child 1) are compact unsigned integers that should not be dict-encoded
    /// again.
    ///
    /// Pull rules in `vortex-btrblocks` also keep `SequenceScheme`, `FoRScheme`, `ZigZagScheme` and
    /// `DeltaScheme` off the codes (child 1), and `RunEndScheme` and the RLE schemes off the values
    /// (child 0).
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
        bytes_dictionary_estimate(&BinaryDictScheme)
    }

    /// Children: values=0, codes=1. The codes carry the validity.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let dict = dict_encode_by_frequency(data.array(), exec_ctx)?;
        Ok(dictionary_encoded(self, &dict))
    }
}
