// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Commit upserts. Adapts a [`crate::v2::V2Commit`] into the v3
//! `commits` row shape (a [`vortex_bench_server::records::CommitInfo`]).

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Transaction;
use duckdb::params;

use crate::v2::V2Commit;

/// Insert a v3 `commits` row for one v2 commit. `tree_sha` and `url`
/// remain required and use a warning-bearing empty-string fallback;
/// the human-input fields (message, author/committer name and email)
/// are nullable in the v3 schema, so empty / missing values map to
/// SQL `NULL` instead of an empty string the UI would render as a
/// blank cell.
pub fn upsert_commit(tx: &Transaction<'_>, commit: &V2Commit) -> Result<UpsertOutcome> {
    let mut warnings = Vec::new();
    let timestamp = require_field(&commit.timestamp, "timestamp", &commit.id, &mut warnings);
    let message = optional_field(&commit.message);
    let author_name = optional_field(&commit.author.as_ref().and_then(|p| p.name.clone()));
    let author_email = optional_field(&commit.author.as_ref().and_then(|p| p.email.clone()));
    let committer_name = optional_field(&commit.committer.as_ref().and_then(|p| p.name.clone()));
    let committer_email = optional_field(&commit.committer.as_ref().and_then(|p| p.email.clone()));
    let tree_sha = require_field(&commit.tree_id, "tree_id", &commit.id, &mut warnings);
    let url = require_field(&commit.url, "url", &commit.id, &mut warnings);

    tx.execute(
        r#"
        INSERT INTO commits (
            commit_sha, timestamp, message, author_name, author_email,
            committer_name, committer_email, tree_sha, url
        ) VALUES (?, CAST(? AS TIMESTAMPTZ), ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (commit_sha) DO UPDATE SET
            timestamp       = excluded.timestamp,
            message         = excluded.message,
            author_name     = excluded.author_name,
            author_email    = excluded.author_email,
            committer_name  = excluded.committer_name,
            committer_email = excluded.committer_email,
            tree_sha        = excluded.tree_sha,
            url             = excluded.url
        "#,
        params![
            commit.id,
            timestamp,
            message,
            author_name,
            author_email,
            committer_name,
            committer_email,
            tree_sha,
            url,
        ],
    )
    .with_context(|| format!("upserting commit {}", commit.id))?;
    Ok(UpsertOutcome { warnings })
}

fn require_field(
    field: &Option<String>,
    name: &str,
    sha: &str,
    warnings: &mut Vec<String>,
) -> String {
    match field {
        Some(s) => s.clone(),
        None => {
            warnings.push(format!("commit {sha} missing {name}"));
            String::new()
        }
    }
}

/// Coerce a v2-supplied `Option<String>` into a SQL-bindable
/// `Option<String>`, treating an empty / whitespace-only value as
/// missing. v2 sometimes wrote `""` for blank author / committer /
/// message fields; storing those as actual `NULL` lets the UI
/// distinguish "missing metadata" from "deliberately blank".
fn optional_field(field: &Option<String>) -> Option<String> {
    field
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Per-call warning bag returned to the caller for logging.
#[derive(Debug, Default)]
pub struct UpsertOutcome {
    /// Human-readable warnings — typically one per missing required field on
    /// the v2 commit (timestamp, tree_id, url).
    pub warnings: Vec<String>,
}
