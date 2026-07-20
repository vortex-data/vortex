// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::CORE_2026_07_0;
use crate::CORE_2026_10_0;
use crate::EditionError;
use crate::manifest::EditionManifest;
use crate::manifest::EditionStatus;
use crate::manifest::compute_manifests;

/// The frozen encoding set of `core2026.07.0`.
///
/// This is the freeze: published editions must never change. If this test fails, you have
/// changed a published edition's computed set — revert the change, or stage it into the
/// current draft edition instead.
const FROZEN_CORE_2026_07_0: &[&str] = &[
    "fastlanes.bitpacked",
    "fastlanes.delta",
    "fastlanes.for",
    "fastlanes.rle",
    "vortex.alp",
    "vortex.alprd",
    "vortex.bool",
    "vortex.bytebool",
    "vortex.chunked",
    "vortex.constant",
    "vortex.datetimeparts",
    "vortex.decimal",
    "vortex.decimal_byte_parts",
    "vortex.dict",
    "vortex.ext",
    "vortex.fixed_size_list",
    "vortex.fsst",
    "vortex.list",
    "vortex.listview",
    "vortex.masked",
    "vortex.null",
    "vortex.pco",
    "vortex.primitive",
    "vortex.runend",
    "vortex.sequence",
    "vortex.sparse",
    "vortex.struct",
    "vortex.varbin",
    "vortex.varbinview",
    "vortex.variant",
    "vortex.zigzag",
    "vortex.zstd",
];

fn manifest_of(id: &str) -> Result<EditionManifest, EditionError> {
    compute_manifests()?
        .into_iter()
        .find(|m| m.id == id)
        .ok_or_else(|| EditionError::new(format!("edition {id} not found")))
}

#[test]
fn frozen_core_2026_07_0() -> Result<(), EditionError> {
    let manifest = manifest_of("core2026.07.0")?;
    let ids: Vec<&str> = manifest.encodings.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, FROZEN_CORE_2026_07_0);
    assert_eq!(manifest.frozen.as_deref(), Some("2026-07"));
    Ok(())
}

#[test]
fn required_release_derives_from_membership_edges() -> Result<(), EditionError> {
    // No membership edge records a required release yet, so the edition-level requirement
    // stays unrecorded and falls back to "first release containing the edition".
    let manifest = manifest_of("core2026.07.0")?;
    assert!(manifest.required_vortex_release.is_none());
    assert!(
        manifest
            .encodings
            .iter()
            .all(|e| e.required_vortex_release.is_none())
    );
    Ok(())
}

#[test]
fn definitions_are_valid() -> Result<(), EditionError> {
    compute_manifests().map(|_| ())
}

#[test]
fn exactly_one_current_edition_per_family() -> Result<(), EditionError> {
    let manifests = compute_manifests()?;
    let core_current: Vec<&EditionManifest> = manifests
        .iter()
        .filter(|m| m.family == "core" && m.status == EditionStatus::Current)
        .collect();
    assert_eq!(core_current.len(), 1);
    assert_eq!(core_current[0].id, "core2026.07.0");
    Ok(())
}

#[test]
fn draft_carries_no_freeze_metadata() -> Result<(), EditionError> {
    let draft = manifest_of("core2026.10.0")?;
    assert_eq!(draft.status, EditionStatus::Draft);
    assert!(draft.frozen.is_none());
    assert!(draft.required_vortex_release.is_none());
    assert_eq!(draft.supersedes.as_deref(), Some("core2026.07.0"));
    assert!(
        draft.encodings.iter().all(|e| e.since != draft.id),
        "no encodings staged in the draft yet"
    );
    Ok(())
}

#[test]
fn manifests_round_trip_through_json() -> Result<(), EditionError> {
    for manifest in compute_manifests()? {
        let json = serde_json::to_string(&manifest)
            .map_err(|e| EditionError::new(format!("serialize: {e}")))?;
        let parsed: EditionManifest = serde_json::from_str(&json)
            .map_err(|e| EditionError::new(format!("deserialize: {e}")))?;
        assert_eq!(parsed.id, manifest.id);
        assert_eq!(parsed.encodings.len(), manifest.encodings.len());
    }
    Ok(())
}

#[test]
fn edition_ids_order_within_family_only() {
    assert!(CORE_2026_07_0.is_at_or_before(&CORE_2026_10_0));
    assert!(!CORE_2026_10_0.is_at_or_before(&CORE_2026_07_0));

    let geo = crate::EditionId::new("geo", 2026, 9, 0);
    assert!(!CORE_2026_07_0.is_at_or_before(&geo));
    assert!(!geo.is_at_or_before(&CORE_2026_07_0));
}

#[test]
fn edition_id_display() {
    assert_eq!(CORE_2026_07_0.to_string(), "core2026.07.0");
}
