// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use super::*;
use crate::scheme::AncestorExclusion;
use crate::scheme::CompressionEstimate;
use crate::scheme::EstimateVerdict;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

static V1_ID: CachedId = CachedId::new("test.version_1");
static V2_ID: CachedId = CachedId::new("test.version_2");
static V3_ID: CachedId = CachedId::new("test.version_3");
static AUX_ID: CachedId = CachedId::new("test.auxiliary");

#[derive(Debug)]
struct TestScheme {
    name: &'static str,
    version: u8,
    predecessor: Option<&'static dyn Scheme>,
    push: Option<&'static dyn Scheme>,
    pull: Option<&'static dyn Scheme>,
}

impl TestScheme {
    const fn new(
        name: &'static str,
        version: u8,
        predecessor: Option<&'static dyn Scheme>,
    ) -> Self {
        Self {
            name,
            version,
            predecessor,
            push: None,
            pull: None,
        }
    }
}

impl Scheme for TestScheme {
    fn scheme_name(&self) -> &'static str {
        self.name
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        match self.version {
            1 => vec![*V1_ID],
            2 => vec![*V2_ID, *AUX_ID],
            3 => vec![*V3_ID],
            _ => vec![],
        }
    }

    fn predecessor(&self) -> Option<&'static dyn Scheme> {
        self.predecessor
    }

    fn num_children(&self) -> usize {
        2
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        self.push
            .map(|scheme| DescendantExclusion {
                excluded: scheme.id(),
                children: ChildSelection::One(1),
            })
            .into_iter()
            .collect()
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        self.pull
            .map(|scheme| AncestorExclusion {
                ancestor: scheme.id(),
                children: ChildSelection::One(1),
            })
            .into_iter()
            .collect()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Older versions would beat newer versions if they reached estimation together.
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(5.0 - f64::from(self.version)))
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

static V1: TestScheme = TestScheme::new("test.scheme_v1", 1, None);
static V2: TestScheme = TestScheme::new("test.scheme_v2", 2, Some(&V1));
static V3: TestScheme = TestScheme::new("test.scheme_v3", 3, Some(&V2));
static OTHER: TestScheme = TestScheme::new("test.other", 0, None);

#[test]
fn newest_eligible_version_is_selected_before_estimation() -> VortexResult<()> {
    let session = vortex_array::array_session();
    let mut exec_ctx = session.create_execution_ctx();
    let data = ArrayAndStats::new(
        PrimitiveArray::from_iter(0..128i32).into_array(),
        GenerateStatsOptions::default(),
    );
    for (allowed, expected) in [
        (None, V3.id()),
        (
            Some(HashSet::from([*V1_ID, *V2_ID, *AUX_ID, *V3_ID])),
            V3.id(),
        ),
        (Some(HashSet::from([*V1_ID, *V2_ID, *AUX_ID])), V2.id()),
        (Some(HashSet::from([*V2_ID, *AUX_ID])), V2.id()),
        (Some(HashSet::from([*V1_ID, *V2_ID])), V1.id()),
        (Some(HashSet::from([*V1_ID])), V1.id()),
    ] {
        let mut compressor = CascadingCompressor::new(vec![&V3]);
        if let Some(allowed) = allowed {
            compressor = compressor.with_allowed_serialized_ids(allowed);
        }
        assert_eq!(compressor.schemes.len(), 1);
        let winner = compressor.choose_best_scheme(
            &compressor.schemes,
            &data,
            compressor.root_context(),
            &mut exec_ctx,
        )?;
        assert_eq!(winner.map(|(scheme, _)| scheme.id()), Some(expected));
        for version in [&V1, &V2, &V3] {
            assert!(compressor.has_scheme(version.id()));
        }
    }
    Ok(())
}

#[test]
fn no_eligible_version_removes_the_entire_chain() {
    for allowed in [HashSet::new(), HashSet::from([*V2_ID])] {
        let compressor = CascadingCompressor::new(vec![&V3]).with_allowed_serialized_ids(allowed);
        assert!(compressor.schemes.is_empty());
        for version in [&V1, &V2, &V3] {
            assert!(!compressor.has_scheme(version.id()));
        }
    }
}

#[test]
fn fallback_preserves_registration_order() {
    let compressor = CascadingCompressor::new(vec![&V3, &OTHER])
        .with_allowed_serialized_ids(HashSet::from([*V1_ID]));
    assert_eq!(
        compressor
            .schemes
            .iter()
            .map(|s| s.id())
            .collect::<Vec<_>>(),
        vec![V1.id(), OTHER.id()]
    );
}

#[test]
fn successive_restrictions_keep_aliases_and_intersect_wire_ids() {
    let compressor = CascadingCompressor::new(vec![&V3])
        .with_allowed_serialized_ids(HashSet::from([*V1_ID, *V2_ID, *AUX_ID]))
        .with_allowed_serialized_ids(HashSet::from([*V1_ID]));
    assert_eq!(compressor.schemes[0].id(), V1.id());
    assert_eq!(compressor.resolve_scheme_id(V3.id()), V1.id());

    let compressor = compressor.with_allowed_serialized_ids(HashSet::from([*V2_ID, *AUX_ID]));
    assert!(compressor.schemes.is_empty());
    assert!(!compressor.has_scheme(V3.id()));
}

static PUSH_OLD: TestScheme = TestScheme {
    push: Some(&V1),
    ..TestScheme::new("test.push_old", 0, None)
};
static PUSH_NEW: TestScheme = TestScheme {
    push: Some(&V3),
    ..TestScheme::new("test.push_new", 0, None)
};
static PULL_OLD: TestScheme = TestScheme {
    pull: Some(&V1),
    ..TestScheme::new("test.pull_old", 0, None)
};
static PULL_NEW: TestScheme = TestScheme {
    pull: Some(&V3),
    ..TestScheme::new("test.pull_new", 0, None)
};

#[test]
fn exclusions_follow_upgrades_and_fallbacks() {
    for allowed in [HashSet::from([*V1_ID]), HashSet::from([*V3_ID])] {
        let compressor =
            CascadingCompressor::new(vec![&V3, &PUSH_OLD, &PUSH_NEW, &PULL_OLD, &PULL_NEW])
                .with_allowed_serialized_ids(allowed);
        let selected = compressor.schemes[0];
        for child in [0, 1] {
            for pusher in [&PUSH_OLD, &PUSH_NEW] {
                let ctx = compressor
                    .root_context()
                    .descend_with_scheme(pusher.id(), child);
                assert_eq!(compressor.is_excluded(selected, &ctx), child == 1);
            }
            let ctx = compressor
                .root_context()
                .descend_with_scheme(selected.id(), child);
            for puller in [&PULL_OLD, &PULL_NEW] {
                assert_eq!(compressor.is_excluded(puller, &ctx), child == 1);
            }
            assert!(compressor.is_excluded(selected, &ctx));
        }
    }
}

#[test]
fn root_exclusions_follow_new_versions() {
    static DICT_V2: TestScheme = TestScheme::new("test.dict_v2", 3, Some(&IntDictScheme));
    let compressor = CascadingCompressor::new(vec![&DICT_V2]);
    let ctx = compressor
        .root_context()
        .descend_with_scheme(ROOT_SCHEME_ID, structural::root_list_children::OFFSETS);
    assert!(compressor.is_excluded(&DICT_V2, &ctx));
    let ctx = compressor
        .root_context()
        .descend_with_scheme(ROOT_SCHEME_ID, structural::root_list_children::SIZES);
    assert!(!compressor.is_excluded(&DICT_V2, &ctx));
}

#[test]
#[should_panic(expected = "appears more than once")]
fn predecessor_cycles_are_rejected() {
    static CYCLE: TestScheme = TestScheme::new("test.cycle", 1, Some(&CYCLE));
    CascadingCompressor::new(vec![&CYCLE]);
}

#[test]
#[should_panic(expected = "appears more than once")]
fn registering_multiple_versions_is_rejected() {
    CascadingCompressor::new(vec![&V3, &V1]);
}
