// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Computed edition manifests: the fully resolved, validated encoding sets for every
//! edition, in the shape serialized to the per-edition JSON files.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use serde::Deserialize;
use serde::Serialize;

use crate::EDITIONS;
use crate::ENCODINGS;
use crate::Edition;
use crate::EditionError;

/// The lifecycle status of an edition, derived from the edition list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EditionStatus {
    /// Being assembled; carries no guarantee and may change freely.
    Draft,
    /// The newest frozen edition of its family; the default write target for `core`.
    Current,
    /// Frozen and replaced by a newer edition of the same family.
    Superseded,
}

/// One encoding's entry in a computed edition manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingManifest {
    /// The encoding ID, e.g. `vortex.alp`.
    pub id: String,
    /// The first edition this encoding was a member of.
    pub since: String,
    /// The earliest Vortex release able to read and execute this encoding, recorded on the
    /// membership edge. Absent until recorded from evidence.
    pub required_vortex_release: Option<String>,
}

/// A fully computed edition: metadata plus the resolved encoding set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditionManifest {
    /// The edition identifier, e.g. `core2026.07.0`.
    pub id: String,
    /// The edition family, e.g. `core`.
    pub family: String,
    /// Derived lifecycle status.
    pub status: EditionStatus,
    /// The year-month the edition was frozen, inferred from the edition identifier
    /// (`core2026.07.0` -> `2026-07`); absent for drafts.
    pub frozen: Option<String>,
    /// The earliest Vortex release guaranteed to read and execute the full edition.
    ///
    /// Never declared by hand: it is *derived* as the maximum of the members'
    /// per-encoding `required_vortex_release` values when every member records one.
    /// Otherwise absent, in which case the required release is the first release whose
    /// edition list contains this edition as frozen — recorded into the published
    /// artifacts by release tooling once that release exists. Always absent for drafts.
    pub required_vortex_release: Option<String>,
    /// The previous edition of the same family, if any.
    pub supersedes: Option<String>,
    /// The encodings in this edition, sorted by ID. The delta against the superseded
    /// edition is derivable: the members whose `since` equals this edition's id.
    pub encodings: Vec<EncodingManifest>,
}

/// Compute the manifest of every declared edition, validating the definitions.
///
/// Validation errors (rather than silently odd output) on: duplicate encoding IDs,
/// membership declarations referencing unknown editions, editions out of chronological
/// order within a family, and malformed release strings.
pub fn compute_manifests() -> Result<Vec<EditionManifest>, EditionError> {
    validate_definitions()?;

    let mut manifests = Vec::with_capacity(EDITIONS.len());
    for edition in EDITIONS {
        let encodings = members_of(edition);

        let supersedes = EDITIONS
            .iter()
            .filter(|e| e.id.family == edition.id.family && e.id != edition.id)
            .filter(|e| e.id.is_at_or_before(&edition.id))
            .next_back()
            .map(|e| e.id.to_string());

        manifests.push(EditionManifest {
            id: edition.id.to_string(),
            family: edition.id.family.to_string(),
            status: status_of(edition),
            frozen: (!edition.draft)
                .then(|| format!("{}-{:02}", edition.id.year, edition.id.month)),
            required_vortex_release: derive_required_release(edition, &encodings)?,
            supersedes,
            encodings,
        });
    }

    Ok(manifests)
}

/// Derive an edition's required Vortex release: the maximum of the members' recorded
/// per-encoding releases, provided every member records one. If any edge is unrecorded (or
/// the edition is a draft) the result is `None`, and the required release falls back to the
/// first release containing the frozen edition.
fn derive_required_release(
    edition: &Edition,
    members: &[EncodingManifest],
) -> Result<Option<String>, EditionError> {
    if edition.draft {
        return Ok(None);
    }
    let mut max: Option<(Vec<u64>, &str)> = None;
    for member in members {
        let Some(release) = member.required_vortex_release.as_deref() else {
            return Ok(None);
        };
        let key = parse_release(release).ok_or_else(|| {
            EditionError::new(format!(
                "encoding {} declares malformed required_vortex_release {release:?}",
                member.id
            ))
        })?;
        if max.as_ref().is_none_or(|(best, _)| key > *best) {
            max = Some((key, release));
        }
    }
    Ok(max.map(|(_, release)| release.to_string()))
}

/// Parse a `major.minor.patch` release string into a comparable key.
fn parse_release(release: &str) -> Option<Vec<u64>> {
    let parts: Vec<u64> = release
        .split('.')
        .map(|part| part.parse::<u64>().ok())
        .collect::<Option<_>>()?;
    (parts.len() == 3).then_some(parts)
}

fn status_of(edition: &Edition) -> EditionStatus {
    if edition.draft {
        return EditionStatus::Draft;
    }
    let newest_frozen = EDITIONS
        .iter()
        .filter(|e| e.id.family == edition.id.family && !e.draft)
        .next_back();
    if newest_frozen.map(|e| e.id) == Some(edition.id) {
        EditionStatus::Current
    } else {
        EditionStatus::Superseded
    }
}

fn members_of(edition: &Edition) -> Vec<EncodingManifest> {
    let mut members: Vec<EncodingManifest> = ENCODINGS
        .iter()
        .filter(|decl| decl.since.is_at_or_before(&edition.id))
        .map(|decl| EncodingManifest {
            id: decl.id.to_string(),
            since: decl.since.to_string(),
            required_vortex_release: decl.required_vortex_release.map(str::to_string),
        })
        .collect();
    members.sort_by(|l, r| l.id.cmp(&r.id));
    members
}

fn validate_definitions() -> Result<(), EditionError> {
    // Edition IDs are unique, and chronological within each family (drafts last).
    let mut seen_editions = BTreeSet::new();
    let mut newest_per_family: BTreeMap<&str, &Edition> = BTreeMap::new();
    for edition in EDITIONS {
        if !seen_editions.insert(edition.id.to_string()) {
            return Err(EditionError::new(format!(
                "duplicate edition {}",
                edition.id
            )));
        }
        if let Some(prev) = newest_per_family.get(edition.id.family) {
            if !prev.id.is_at_or_before(&edition.id) {
                return Err(EditionError::new(format!(
                    "edition {} is out of chronological order within family {}",
                    edition.id, edition.id.family,
                )));
            }
            if prev.draft && !edition.draft {
                return Err(EditionError::new(format!(
                    "frozen edition {} follows draft {}; drafts must be newest in a family",
                    edition.id, prev.id,
                )));
            }
        }
        newest_per_family.insert(edition.id.family, edition);
    }

    // Encoding IDs are unique, and `since` references a declared edition.
    let mut seen_encodings = BTreeSet::new();
    for decl in ENCODINGS {
        if !seen_encodings.insert(decl.id) {
            return Err(EditionError::new(format!("duplicate encoding {}", decl.id)));
        }
        if !EDITIONS.iter().any(|e| e.id == decl.since) {
            return Err(EditionError::new(format!(
                "encoding {} declares membership of unknown edition {}",
                decl.id, decl.since
            )));
        }
    }

    Ok(())
}
