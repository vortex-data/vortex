// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme configuration checks every required serialized ID.

#![cfg(test)]

use rstest::rstest;
use vortex_alp::ALP;
use vortex_array::ArrayId;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Patched;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_btrblocks::AllowedSerializedIds;
use vortex_btrblocks::Scheme;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::float::ALPScheme;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_btrblocks::schemes::integer::SparseScheme;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_fastlanes::BitPacked;
use vortex_fsst::FSST;
use vortex_sparse::Sparse;
use vortex_utils::aliases::hash_set::HashSet;

#[rstest]
#[case::fsst(&FSSTScheme, vec![FSST.id(), VarBin.id()])]
#[case::sparse(&SparseScheme, vec![Sparse.id(), Constant.id()])]
fn every_required_id_must_be_permitted(
    #[case] scheme: &dyn Scheme,
    #[case] ids: Vec<ArrayId>,
) {
    let all: HashSet<_> = ids.iter().copied().collect();
    assert_eq!(
        scheme
            .configure(&AllowedSerializedIds::Only(all.clone()))
            .map(|configured| configured.id()),
        Some(scheme.id())
    );
    for id in ids {
        let mut allowed = all.clone();
        allowed.remove(&id);
        assert!(
            scheme
                .configure(&AllowedSerializedIds::Only(allowed))
                .is_none()
        );
        let forbidden = HashSet::from([id]);
        assert!(
            scheme
                .configure(&AllowedSerializedIds::AllExcept(forbidden))
                .is_none()
        );
    }
}

#[rstest]
#[case::bitpacking(&BitPackingScheme, BitPacked.id())]
#[case::alp(&ALPScheme, ALP.id())]
fn patched_permission_is_required_when_enabled(
    #[case] scheme: &dyn Scheme,
    #[case] id: ArrayId,
) {
    let mut allowed = HashSet::from([id]);
    assert_eq!(
        scheme
            .configure(&AllowedSerializedIds::Only(allowed.clone()))
            .is_some(),
        !use_experimental_patches()
    );
    allowed.insert(Patched.id());
    assert!(
        scheme
            .configure(&AllowedSerializedIds::Only(allowed.clone()))
            .is_some()
    );
    allowed.remove(&id);
    assert!(
        scheme
            .configure(&AllowedSerializedIds::Only(allowed))
            .is_none()
    );
}
