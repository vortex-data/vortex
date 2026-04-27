// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Commit upserts. Adapts a [`crate::v2::V2Commit`] into the v3
//! `commits` row shape (a [`vortex_bench_server::records::CommitInfo`]).

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Transaction;
use duckdb::params;

use crate::v2::V2Commit;

/// Insert a v3 `commits` row for one v2 commit. Missing fields are
/// filled with the empty string, matching the v3 schema's `NOT NULL`
/// constraints; the call site logs a warning for each fallback so
/// the operator can spot bad inputs.
pub fn upsert_commit(tx: &Transaction<'_>, commit: &V2Commit) -> Result<UpsertOutcome> {
    let mut warnings = Vec::new();
    let timestamp = require_field(&commit.timestamp, "timestamp", &commit.id, &mut warnings);
    let message = require_field(&commit.message, "message", &commit.id, &mut warnings);
    let author_name = require_field(
        &commit.author.as_ref().and_then(|p| p.name.clone()),
        "author.name",
        &commit.id,
        &mut warnings,
    );
    let author_email = require_field(
        &commit.author.as_ref().and_then(|p| p.email.clone()),
        "author.email",
        &commit.id,
        &mut warnings,
    );
    let committer_name = require_field(
        &commit.committer.as_ref().and_then(|p| p.name.clone()),
        "committer.name",
        &commit.id,
        &mut warnings,
    );
    let committer_email = require_field(
        &commit.committer.as_ref().and_then(|p| p.email.clone()),
        "committer.email",
        &commit.id,
        &mut warnings,
    );
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

/// Per-call warning bag returned to the caller for logging.
#[derive(Debug, Default)]
pub struct UpsertOutcome {
    pub warnings: Vec<String>,
}
