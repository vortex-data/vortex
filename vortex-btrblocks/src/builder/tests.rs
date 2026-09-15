// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::arrays::VarBin;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fsst::FSST;
use vortex_session::registry::CachedId;

use super::*;
use crate::ArrayAndStats;
use crate::CompressorContext;

#[test]
fn empty_starts_with_no_schemes() {
    assert!(BtrBlocksCompressorBuilder::empty().schemes.is_empty());
}

#[test]
fn default_includes_all_schemes() {
    assert_eq!(
        BtrBlocksCompressorBuilder::default().schemes.len(),
        ALL_SCHEMES.len()
    );
}

#[test]
fn allowed_serialized_ids_filter_schemes_at_build() {
    let compressor = BtrBlocksCompressorBuilder::default()
        .allow_serialized_ids(&HashSet::from([FoR.id()]))
        .build();
    for scheme in ALL_SCHEMES {
        assert_eq!(
            compressor.has_scheme(scheme.id()),
            scheme.id() == integer::FoRScheme.id()
        );
    }
}

#[test]
fn allowing_all_declared_outputs_keeps_every_scheme() {
    let allowed = ALL_SCHEMES
        .iter()
        .flat_map(|s| s.produced_encodings())
        .collect();
    let compressor = BtrBlocksCompressorBuilder::default()
        .allow_serialized_ids(&allowed)
        .build();
    for scheme in ALL_SCHEMES {
        assert!(compressor.has_scheme(scheme.id()));
    }
}

#[rstest]
#[case::neither(vec![], false)]
#[case::fsst_only(vec![FSST.id()], false)]
#[case::varbin_only(vec![VarBin.id()], false)]
#[case::both(vec![FSST.id(), VarBin.id()], true)]
fn all_required_outputs_must_be_allowed(#[case] allowed: Vec<ArrayId>, #[case] expected: bool) {
    let compressor = BtrBlocksCompressorBuilder::default()
        .allow_serialized_ids(&allowed.into_iter().collect())
        .build();
    assert_eq!(compressor.has_scheme(string::FSSTScheme.id()), expected);
}

#[rstest]
#[case::forbidden(HashSet::new(), false)]
#[case::permitted(HashSet::from([FoR.id()]), true)]
fn restriction_applies_to_schemes_added_later(
    #[case] allowed: HashSet<ArrayId>,
    #[case] expected: bool,
) {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .allow_serialized_ids(&allowed)
        .with_new_scheme(&integer::FoRScheme)
        .build();
    assert_eq!(compressor.has_scheme(integer::FoRScheme.id()), expected);
}

#[test]
fn repeated_restrictions_intersect() {
    let compressor = BtrBlocksCompressorBuilder::default()
        .allow_serialized_ids(&HashSet::from([FoR.id(), FSST.id()]))
        .allow_serialized_ids(&HashSet::from([FSST.id(), VarBin.id()]))
        .build();
    assert!(!compressor.has_scheme(integer::FoRScheme.id()));
    assert!(!compressor.has_scheme(string::FSSTScheme.id()));
}

#[test]
fn cuda_compatible_excludes_alprd() {
    let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
    assert!(
        !builder
            .schemes
            .iter()
            .any(|s| s.id() == float::ALPRDScheme.id())
    );
}

/// `vortex.sparse` has no CUDA decode kernel, so no sparse scheme may survive this preset.
#[test]
fn cuda_compatible_excludes_every_sparse_scheme() {
    let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
    for excluded in [
        integer::SparseScheme.id(),
        float::NullDominatedSparseScheme.id(),
        string::NullDominatedSparseScheme.id(),
    ] {
        assert!(
            !builder.schemes.iter().any(|s| s.id() == excluded),
            "{excluded} should be excluded"
        );
    }
}

#[test]
fn cuda_compatible_uses_fsst_for_strings() {
    let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
    assert!(
        builder
            .schemes
            .iter()
            .any(|scheme| scheme.id() == string::FSSTScheme.id())
    );
    #[cfg(feature = "zstd")]
    assert!(
        !builder
            .schemes
            .iter()
            .any(|scheme| scheme.id() == string::ZstdScheme.id())
    );
}

#[test]
#[cfg(feature = "pco")]
fn cuda_compatible_excludes_pco() {
    let builder = BtrBlocksCompressorBuilder::default()
        .with_new_scheme(&integer::PcoScheme)
        .with_new_scheme(&float::PcoScheme)
        .only_cuda_compatible();
    for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
        assert!(!builder.schemes.iter().any(|s| s.id() == scheme));
    }
}

static FOR_V2_ID: CachedId = CachedId::new("test.for_v2");

#[derive(Debug)]
struct NewFoRScheme;

impl Scheme for NewFoRScheme {
    fn scheme_name(&self) -> &'static str {
        "test.for_v2"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        integer::FoRScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![*FOR_V2_ID]
    }

    fn predecessor(&self) -> Option<&'static dyn Scheme> {
        Some(&integer::FoRScheme)
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

#[test]
fn restrictions_select_predecessors_of_schemes_added_later() {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .allow_serialized_ids(&HashSet::from([FoR.id()]))
        .with_new_scheme(&NewFoRScheme)
        .build();
    assert!(compressor.has_scheme(integer::FoRScheme.id()));
    assert!(compressor.has_scheme_family(NewFoRScheme.id()));
    assert!(!compressor.has_scheme(NewFoRScheme.id()));
}

#[rstest]
#[case::old(integer::FoRScheme.id())]
#[case::new(NewFoRScheme.id())]
fn excluding_any_version_removes_the_chain(#[case] excluded: SchemeId) {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&NewFoRScheme)
        .exclude_schemes([excluded])
        .build();
    assert!(!compressor.has_scheme_family(integer::FoRScheme.id()));
    assert!(!compressor.has_scheme_family(NewFoRScheme.id()));
}
