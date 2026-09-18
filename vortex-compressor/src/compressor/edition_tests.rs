// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use super::*;
use crate::scheme::CompressionEstimate;
use crate::scheme::EstimateVerdict;
use crate::stats::ArrayAndStats;

static V1_ID: CachedId = CachedId::new("test.format_v1");
static V2_ID: CachedId = CachedId::new("test.format_v2");

/// A scheme that always writes `test.format_v1` and, when permitted, also `test.format_v2`.
#[derive(Debug)]
struct ModeScheme;

impl Scheme for ModeScheme {
    fn scheme_name(&self) -> &'static str {
        "test.mode"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![*V1_ID]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(data.array().clone())
    }
}

fn only(ids: &[&CachedId]) -> AllowedSerializedIds {
    AllowedSerializedIds::Only(ids.iter().map(|id| ***id).collect())
}

#[test]
fn unrestricted_contexts_permit_everything() {
    let compressor = CascadingCompressor::new(vec![&ModeScheme]);
    assert_eq!(
        compressor.allowed_serialized_ids(),
        &AllowedSerializedIds::All
    );
    let ctx = compressor.root_context();
    assert!(ctx.allows_serialized_id(&V1_ID));
    assert!(ctx.allows_serialized_id(&V2_ID));
}

#[test]
fn the_permitted_set_reaches_descendant_contexts() {
    let compressor =
        CascadingCompressor::new(vec![&ModeScheme]).with_allowed_serialized_ids(&only(&[&V1_ID]));
    let root = compressor.root_context();
    assert!(root.allows_serialized_id(&V1_ID));
    assert!(!root.allows_serialized_id(&V2_ID));

    let child = root.descend_with_scheme(ModeScheme.id(), 0);
    assert!(child.allows_serialized_id(&V1_ID));
    assert!(!child.allows_serialized_id(&V2_ID));
    assert_eq!(child.allowed_serialized_ids(), &only(&[&V1_ID]));
}

#[test]
fn repeated_restrictions_intersect() {
    let compressor = CascadingCompressor::new(vec![&ModeScheme])
        .with_allowed_serialized_ids(&only(&[&V1_ID, &V2_ID]))
        .with_allowed_serialized_ids(&only(&[&V1_ID]));
    assert_eq!(compressor.allowed_serialized_ids(), &only(&[&V1_ID]));
    assert!(!compressor.root_context().allows_serialized_id(&V2_ID));

    // Intersecting with `All` changes nothing.
    let compressor = compressor.with_allowed_serialized_ids(&AllowedSerializedIds::All);
    assert_eq!(compressor.allowed_serialized_ids(), &only(&[&V1_ID]));
}
