// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Check that frozen edition records under `vortex/editions` never change.
//!
//! A draft record carries no read-forever guarantee and is not mechanically locked by this check.
//! Once `min_library_version` is backfilled, the record captures the frozen edition's complete
//! membership and may never change again. Whether a record was frozen is read from the base
//! revision, so a change cannot unfreeze an edition and edit it in the same diff.
//!
//! A newly added record must also be newer than every edition already recorded for its
//! family: editions are only ever added going forward. Records are grouped by family, so
//! `vortex/editions/core/core2025.05.0.toml` must sit under the family its name declares. The
//! `family.toml` beside them documents the family rather than pinning a contract, so it is
//! exempt.
//!
//! Both revisions are read out of the object database, so the check sees committed state only
//! and never the working tree.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::anyhow;
use anyhow::bail;
use git2::Commit;
use git2::Delta;
use git2::DiffFindOptions;
use git2::Repository;
use git2::TreeWalkMode;
use git2::TreeWalkResult;
use toml::Table;

use crate::generate_editions::FAMILY_FILE;
use crate::generate_editions::RECORD_DIR;

/// A record carries this key once the edition's freeze has been documented.
const FROZEN_MARKER: &str = "min_library_version";

const REMEDY: &str = "\
A frozen edition is immutable. To add component IDs, declare a NEW edition in
  the family's declaration module and regenerate the records with
  `cargo run -p xtask -- generate-editions`.";

/// An edition's position in its family's chronology, from `<family><year>.<month>.<version>`.
type Chronology = (u16, u8, u8);

/// Split a record file name into its family and its place in that family's chronology.
fn parse_name(name: &str) -> anyhow::Result<(&str, Chronology)> {
    let malformed = || {
        anyhow!(
            "{RECORD_DIR}/{name} is not a valid record name. Records are named after the \
             edition they record, e.g. `core/core2026.08.0.toml`."
        )
    };

    let stem = name.strip_suffix(".toml").ok_or_else(malformed)?;
    let split = stem
        .find(|c: char| c.is_ascii_digit())
        .ok_or_else(malformed)?;
    let (family, version) = stem.split_at(split);
    if family.is_empty() || !family.chars().all(|c| c.is_ascii_lowercase()) {
        return Err(malformed());
    }

    let parts: Vec<&str> = version.split('.').collect();
    let [year, month, version] = parts.as_slice() else {
        return Err(malformed());
    };
    if year.len() != 4 || month.len() != 2 {
        return Err(malformed());
    }
    let chronology = (
        year.parse().map_err(|_| malformed())?,
        month.parse().map_err(|_| malformed())?,
        version.parse().map_err(|_| malformed())?,
    );
    Ok((family, chronology))
}

/// Parse a record out of a commit's tree, or `None` when it holds no such file.
fn read_record(repo: &Repository, commit: &Commit, path: &str) -> anyhow::Result<Option<Table>> {
    let Ok(entry) = commit.tree()?.get_path(Path::new(path)) else {
        return Ok(None);
    };
    let blob = entry.to_object(repo)?.peel_to_blob()?;
    let text = std::str::from_utf8(blob.content())
        .with_context(|| format!("{path} at {} is not UTF-8", commit.id()))?;
    Ok(Some(text.parse::<Table>().with_context(|| {
        format!("{path} at {} is not valid TOML", commit.id())
    })?))
}

/// The newest edition already recorded for each family at `commit`.
fn newest_recorded(
    repo: &Repository,
    commit: &Commit,
) -> anyhow::Result<BTreeMap<String, Chronology>> {
    let mut newest = BTreeMap::new();
    let Ok(entry) = commit.tree()?.get_path(Path::new(RECORD_DIR)) else {
        return Ok(newest);
    };
    let records = entry.to_object(repo)?.peel_to_tree()?;

    let mut malformed = None;
    records.walk(TreeWalkMode::PreOrder, |_, entry| {
        let Ok(name) = entry.name() else {
            return TreeWalkResult::Ok;
        };
        if !name.ends_with(".toml") || name == FAMILY_FILE {
            return TreeWalkResult::Ok;
        }
        match parse_name(name) {
            Ok((family, chronology)) => {
                let slot = newest.entry(family.to_string()).or_insert(chronology);
                *slot = (*slot).max(chronology);
                TreeWalkResult::Ok
            }
            Err(error) => {
                malformed = Some(error);
                TreeWalkResult::Abort
            }
        }
    })?;
    match malformed {
        Some(error) => Err(error),
        None => Ok(newest),
    }
}

/// A frozen record may not change at all; name the fields that did.
fn check_modification(before: &Table, after: &Table, name: &str) -> Vec<String> {
    let mut changed: Vec<&str> = before
        .keys()
        .chain(after.keys())
        .map(String::as_str)
        .filter(|key| before.get(*key) != after.get(*key))
        .collect();
    changed.sort_unstable();
    changed.dedup();

    if changed.is_empty() {
        return vec![];
    }
    if changed.contains(&FROZEN_MARKER) && !after.contains_key(FROZEN_MARKER) {
        return vec![format!(
            "unfreezes {name}; an edition that recorded a {FROZEN_MARKER} carries a \
             read-forever guarantee and may never return to draft"
        )];
    }
    vec![format!(
        "modifies the frozen record {name}: {}",
        changed.join(", ")
    )]
}

/// A new record must extend its family's chronology, and be filed under that family.
fn check_addition(
    path: &str,
    record: &Table,
    newest: &BTreeMap<String, Chronology>,
) -> anyhow::Result<Vec<String>> {
    let mut errors = Vec::new();
    let name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("{path} has no file name"))?;
    let (family, chronology) = parse_name(name)?;

    if let Some(previous) = newest.get(family)
        && chronology <= *previous
    {
        errors.push(format!(
            "adds {name}, which is not newer than the {family} edition already recorded \
             ({family}{}.{:02}.{}). Editions may only be added going forward.",
            previous.0, previous.1, previous.2,
        ));
    }

    // A record's family decides which chronology it extends, so the directory it sits in has
    // to agree with the family its name declares.
    let directory = Path::new(path)
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if directory != family {
        errors.push(format!(
            "adds {name} under {directory}/, but it records a {family} edition; records are \
             grouped by family"
        ));
    }

    // The file name is the edition's identity, so it has to agree with the content.
    match record.get("edition").and_then(|edition| edition.as_str()) {
        None => errors.push(format!("adds {name}, which has no `edition` field")),
        Some(edition) if edition != name.trim_end_matches(".toml") => errors.push(format!(
            "adds {name}, which records edition {edition:?}; the file name must be the \
             edition id"
        )),
        Some(_) => {}
    }
    match record.get("origin").and_then(|origin| origin.as_str()) {
        None => errors.push(format!("adds {name}, which has no `origin` field")),
        Some(origin) if origin.trim().is_empty() => {
            errors.push(format!("adds {name}, which has an empty `origin` field"));
        }
        Some(_) => {}
    }
    Ok(errors)
}

fn under_record_dir(path: Option<&Path>) -> bool {
    path.is_some_and(|path| path.starts_with(RECORD_DIR))
}

fn is_family_record(path: Option<&Path>) -> bool {
    path.is_some_and(|path| path.file_name().is_some_and(|name| name == FAMILY_FILE))
}

fn path_str(path: Option<&Path>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_default()
}

pub fn check_editions(base: &str) -> anyhow::Result<()> {
    let repo = Repository::discover(".").context("opening the repository")?;
    let base_tip = repo
        .revparse_single(base)
        .with_context(|| format!("cannot resolve {base:?} in this repository"))?
        .peel_to_commit()?;
    let head = repo.head()?.peel_to_commit()?;

    let merge_base = repo.merge_base(base_tip.id(), head.id()).with_context(|| {
        format!(
            "{base} and HEAD have no common ancestor. The checkout is probably too shallow; \
             this check needs `fetch-depth: 0`."
        )
    })?;
    let base_commit = repo.find_commit(merge_base)?;

    let mut diff = repo.diff_tree_to_tree(Some(&base_commit.tree()?), Some(&head.tree()?), None)?;
    diff.find_similar(Some(DiffFindOptions::new().renames(true)))?;

    let newest = newest_recorded(&repo, &base_commit)?;
    let mut errors = Vec::new();
    let mut added = Vec::new();

    for delta in diff.deltas() {
        let (old_path, new_path) = (delta.old_file().path(), delta.new_file().path());
        if !under_record_dir(old_path) && !under_record_dir(new_path) {
            continue;
        }
        // The family record is documentation rather than a contract, so it stays editable.
        if is_family_record(new_path) || is_family_record(old_path) {
            continue;
        }

        if delta.status() == Delta::Added {
            added.push(path_str(new_path));
            continue;
        }

        // Frozen-ness comes from the base revision, so a diff cannot unfreeze an edition and
        // then edit it. Legacy draft records have no compatibility contract and are ignored.
        let old = path_str(old_path);
        let Some(before) = read_record(&repo, &base_commit, &old)? else {
            continue;
        };
        if !before.contains_key(FROZEN_MARKER) {
            continue;
        }

        let new = path_str(new_path);
        if delta.status() == Delta::Modified {
            let after = read_record(&repo, &head, &new)?.unwrap_or_default();
            let name = Path::new(&new)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&new);
            errors.extend(check_modification(&before, &after, name));
        } else {
            let verb = match delta.status() {
                Delta::Deleted => "deletes",
                Delta::Renamed => "renames",
                Delta::Copied => "copies",
                Delta::Typechange => "retypes",
                _ => "changes",
            };
            let moved = if old == new {
                old.clone()
            } else {
                format!("{old} -> {new}")
            };
            errors.push(format!("{verb} the frozen record {moved}"));
        }
    }

    added.sort();
    for path in &added {
        if let Some(record) = read_record(&repo, &head, path)? {
            errors.extend(check_addition(path, &record, &newest)?);
        }
    }

    if errors.is_empty() {
        println!("{RECORD_DIR} preserves every frozen record against {base}.");
        return Ok(());
    }

    let listed = errors
        .iter()
        .map(|error| format!("  - it {error}"))
        .collect::<Vec<_>>()
        .join("\n");
    bail!("This change breaks the edition records in {RECORD_DIR}:\n\n{listed}\n\n{REMEDY}");
}
