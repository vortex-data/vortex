// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::VTable;
use vortex_fastlanes::FoR;

use super::*;

static DELTA_2: integer::DeltaScheme = integer::DeltaScheme::new(2.0);
static DELTA_3: integer::DeltaScheme = integer::DeltaScheme::new(3.0);

#[test]
fn empty_starts_with_no_schemes() {
    let builder = BtrBlocksCompressorBuilder::empty();
    assert!(builder.schemes.is_empty());
}

#[test]
fn default_includes_all_schemes() {
    let builder = BtrBlocksCompressorBuilder::default();
    assert_eq!(builder.schemes.len(), ALL_SCHEMES.len());
    let compressor = builder.build();
    for scheme in ALL_SCHEMES {
        assert!(compressor.has_scheme(scheme.id()));
    }
}

#[test]
fn retain_allowed_encodings_filters_schemes() {
    let allowed: HashSet<ArrayId> = [FoR.id()].into_iter().collect();
    let compressor = BtrBlocksCompressorBuilder::default()
        .retain_allowed_encodings(&allowed)
        .build();
    assert!(compressor.has_scheme(integer::FoRScheme.id()));
    assert!(!compressor.has_scheme(integer::BitPackingScheme.id()));

    let none = BtrBlocksCompressorBuilder::default()
        .retain_allowed_encodings(&HashSet::new())
        .build();
    for scheme in ALL_SCHEMES {
        assert!(!none.has_scheme(scheme.id()));
    }
}

#[test]
fn all_produced_encodings_retain_every_default_scheme() {
    let allowed: HashSet<_> = ALL_SCHEMES
        .iter()
        .flat_map(|scheme| scheme.produced_encodings())
        .chain([decimal_byte_parts_v2_id()])
        .collect();
    let compressor = BtrBlocksCompressorBuilder::new(&allowed).build();
    for scheme in ALL_SCHEMES {
        assert!(compressor.has_scheme(scheme.id()));
    }
}

#[test]
fn filtering_leaves_later_registrations_unchanged() {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .retain_allowed_encodings(&HashSet::new())
        .with_new_scheme(&integer::FoRScheme)
        .build();
    assert!(compressor.has_scheme(integer::FoRScheme.id()));
}

#[test]
#[should_panic(expected = "already present")]
fn duplicate_schemes_are_rejected_at_registration() {
    BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&integer::FoRScheme)
        .with_new_scheme(&integer::FoRScheme);
}

#[test]
fn excluded_scheme_can_be_replaced() {
    let id = DELTA_2.id();
    let builder = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([id])
        .with_new_scheme(&DELTA_2);
    assert!(builder.clone().build().has_scheme(id));
    let builder = builder.exclude_schemes([id]);
    assert!(!builder.clone().build().has_scheme(id));
    assert!(builder.with_new_scheme(&DELTA_3).build().has_scheme(id));
}

#[test]
fn default_schemes_can_be_registered() {
    let compressor = ALL_SCHEMES
        .iter()
        .fold(BtrBlocksCompressorBuilder::empty(), |builder, scheme| {
            builder.with_new_scheme(*scheme)
        })
        .build();
    for scheme in ALL_SCHEMES {
        assert!(compressor.has_scheme(scheme.id()));
    }
}

#[rstest]
#[case::empty(BtrBlocksCompressorBuilder::empty())]
#[case::excluded(BtrBlocksCompressorBuilder::default().exclude_schemes([DECIMAL_V1.id()]))]
fn allowed_formats_do_not_restore_decimal(#[case] builder: BtrBlocksCompressorBuilder) {
    let compressor = builder
        .retain_allowed_encodings(&HashSet::from([
            vortex_decimal_byte_parts::decimal_byte_parts_v1_id(),
            decimal_byte_parts_v2_id(),
        ]))
        .build();
    assert!(!compressor.has_scheme(DECIMAL_V1.id()));
}

#[test]
fn cuda_compatible_does_not_add_decimal() {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .only_cuda_compatible()
        .build();
    assert!(!compressor.has_scheme(DECIMAL_V1.id()));
}

#[test]
fn cuda_compatible_does_not_restore_excluded_decimal() {
    let id = DECIMAL_V1.id();
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([id])
        .only_cuda_compatible()
        .build();
    assert!(!compressor.has_scheme(id));
}

#[test]
fn cuda_compatible_excludes_alprd() {
    let compressor = BtrBlocksCompressorBuilder::default()
        .only_cuda_compatible()
        .build();
    assert!(!compressor.has_scheme(float::ALPRDScheme.id()));
}

/// `vortex.sparse` has no CUDA decode kernel, so no sparse scheme may survive this preset.
#[test]
fn cuda_compatible_excludes_every_sparse_scheme() {
    let compressor = BtrBlocksCompressorBuilder::default()
        .only_cuda_compatible()
        .build();
    for excluded in [
        integer::SparseScheme.id(),
        float::NullDominatedSparseScheme.id(),
        string::NullDominatedSparseScheme.id(),
    ] {
        assert!(
            !compressor.has_scheme(excluded),
            "{excluded} should be excluded"
        );
    }
}

#[test]
fn cuda_compatible_uses_fsst_for_strings() {
    let compressor = BtrBlocksCompressorBuilder::default()
        .only_cuda_compatible()
        .build();
    assert!(compressor.has_scheme(string::FSSTScheme.id()));
    #[cfg(feature = "zstd")]
    assert!(!compressor.has_scheme(string::ZstdScheme.id()));
}

#[test]
#[cfg(feature = "pco")]
fn cuda_compatible_excludes_pco() {
    let compressor = BtrBlocksCompressorBuilder::default()
        .with_new_scheme(&integer::PcoScheme)
        .with_new_scheme(&float::PcoScheme)
        .only_cuda_compatible()
        .build();
    for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
        assert!(!compressor.has_scheme(scheme));
    }
}
