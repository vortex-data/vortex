// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data acquisition and table specs for SQLStorm StackOverflow / JOB origins.
//!
//! TPC-H and TPC-DS reuse vortex-bench's existing datasets, so only the two
//! non-TPC origins have a download → extract → DuckDB-convert pipeline. Both
//! origins share the same driver ([`generate_origin`]); each is parameterized
//! by an [`OriginData`] recipe.
//!
//! ## Identifier case
//!
//! The upstream StackOverflow DDL uses CamelCase column names (`OwnerUserId`,
//! `CreationDate`, …) and capitalized table names (`Posts`, `Users`, …). The
//! SQLStorm queries reference those names unquoted, which would fail under
//! DataFusion's default `enable_ident_normalization=true` (the parser
//! lowercases identifiers while the Parquet schema preserves case →
//! field-not-found).
//!
//! [`STACKOVERFLOW`]'s DDL inlines the schema with **lowercase** column names,
//! so `COPY (SELECT * FROM "Posts") TO 'posts.parquet'` writes lowercase
//! columns into the Parquet shard. DuckDB's case-insensitive unquoted
//! identifier resolution and DataFusion's identifier lowercasing then both
//! resolve the queries' CamelCase column references against the lowercased
//! schema. Table names in the DDL stay CamelCase so that each
//! `COPY "Posts" FROM 'Posts.csv'` reads naturally; the lowercase output
//! shard name is the second element of each entry in [`OriginData::tables`].

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::bail;
use tracing::info;
use url::Url;

use crate::Format;
use crate::TableSpec;
use crate::datasets::data_downloads::download_data;
use crate::sqlstorm::SqlstormOrigin;

/// Archive codec; selects the extraction command in [`extract_archive`].
enum Archive {
    /// gzip-compressed tar (`tar -xzf`).
    TarGz,
    /// zstd-compressed tar (`zstd -dc | tar -xf -`).
    TarZst,
}

/// Per-origin data-gen recipe consumed by [`generate_origin`].
pub struct OriginData {
    /// Upstream archive URL.
    url: &'static str,
    /// Local filename to save the downloaded archive as (relative to base dir).
    archive_name: &'static str,
    /// Archive codec.
    archive: Archive,
    /// Origin name, used only in log messages.
    log_name: &'static str,
    /// SQL DDL: one `CREATE TABLE` per entry in [`OriginData::tables`], with
    /// lowercase column names so `SELECT *` exports lowercase Parquet columns.
    ddl: &'static str,
    /// `(upstream_table, output_shard_stem)` for each table. The upstream
    /// name is the CamelCase table created by [`OriginData::ddl`] (and equals
    /// the CSV file stem); the output is the lowercase Parquet shard stem.
    /// For origins already lowercase upstream (JOB), both elements are equal.
    tables: &'static [(&'static str, &'static str)],
    /// Extra options spliced into each `COPY <table> FROM '<csv>' (..., {extra})`
    /// statement after the standard csv settings. Empty when only the
    /// defaults are needed.
    extra_copy_opts: &'static str,
}

/// StackOverflow `math` data (~12 GB gzip). Schema transcribed from
/// `https://db.in.tum.de/~schmidt/data/stackoverflow_schema.sql` with
/// `ALTER TABLE … ADD FOREIGN KEY` lines (which DuckDB rejects) dropped,
/// inline `primary key` / `references` clauses elided (not enforced by
/// COPY, just noise), and column names lowercased.
///
/// The `math` tier's large free-text columns (`Posts.body`, `PostHistory.text`,
/// …) contain rows whose embedded quotes don't strictly comply with RFC-4180,
/// which makes DuckDB's CSV dialect sniffer fail outright. `extra_copy_opts`
/// therefore disables auto-detection and pins the dialect explicitly (RFC-4180
/// doubled-quote escaping), with `strict_mode false` + `ignore_errors true` to
/// tolerate the non-compliant minority of rows. (The smaller `dba` tier happened
/// to be sniffable, so the original empty options worked there.)
pub const STACKOVERFLOW: OriginData = OriginData {
    url: "https://db.in.tum.de/~schmidt/data/stackoverflow_math.tar.gz",
    archive_name: "stackoverflow_math.tar.gz",
    archive: Archive::TarGz,
    log_name: "stackoverflow",
    ddl: r#"
CREATE TABLE "PostHistoryTypes" ("id" SMALLINT NOT NULL, "name" VARCHAR(50) NOT NULL);
CREATE TABLE "LinkTypes" ("id" SMALLINT NOT NULL, "name" VARCHAR(50) NOT NULL);
CREATE TABLE "PostTypes" ("id" SMALLINT NOT NULL, "name" VARCHAR(50) NOT NULL);
CREATE TABLE "CloseReasonTypes" ("id" SMALLINT NOT NULL, "name" VARCHAR(50) NOT NULL);
CREATE TABLE "VoteTypes" ("id" SMALLINT NOT NULL, "name" VARCHAR(50) NOT NULL);
CREATE TABLE "Users" ("id" INTEGER NOT NULL, "reputation" INTEGER NOT NULL, "creationdate" TIMESTAMP NOT NULL, "displayname" VARCHAR(40), "lastaccessdate" TIMESTAMP NOT NULL, "websiteurl" VARCHAR(200), "location" VARCHAR(300), "aboutme" TEXT, "views" INTEGER, "upvotes" INTEGER, "downvotes" INTEGER, "profileimageurl" VARCHAR(200), "accountid" INTEGER);
CREATE TABLE "Badges" ("id" INTEGER NOT NULL, "userid" INTEGER NOT NULL, "name" VARCHAR(50) NOT NULL, "date" TIMESTAMP NOT NULL, "class" SMALLINT NOT NULL, "tagbased" BOOLEAN NOT NULL);
CREATE TABLE "Posts" ("id" INTEGER NOT NULL, "posttypeid" SMALLINT, "acceptedanswerid" INTEGER, "parentid" INTEGER, "creationdate" TIMESTAMP, "score" INTEGER, "viewcount" INTEGER, "body" TEXT, "owneruserid" INTEGER, "ownerdisplayname" VARCHAR(40), "lasteditoruserid" INTEGER, "lasteditordisplayname" VARCHAR(40), "lasteditdate" TIMESTAMP, "lastactivitydate" TIMESTAMP, "title" VARCHAR(300), "tags" VARCHAR(4000), "answercount" INTEGER, "commentcount" INTEGER, "favoritecount" INTEGER, "closeddate" TIMESTAMP, "communityowneddate" TIMESTAMP, "contentlicense" VARCHAR(30));
CREATE TABLE "Comments" ("id" INTEGER NOT NULL, "postid" INTEGER NOT NULL, "score" INTEGER, "text" VARCHAR(2000) NOT NULL, "creationdate" TIMESTAMP NOT NULL, "userdisplayname" VARCHAR(40), "userid" INTEGER, "contentlicense" VARCHAR(30));
CREATE TABLE "PostHistory" ("id" INTEGER NOT NULL, "posthistorytypeid" SMALLINT, "postid" INTEGER, "revisionguid" VARCHAR(36), "creationdate" TIMESTAMP, "userid" INTEGER, "userdisplayname" VARCHAR(40), "comment" VARCHAR(800), "text" TEXT, "contentlicense" VARCHAR(30));
CREATE TABLE "PostLinks" ("id" BIGINT NOT NULL, "creationdate" TIMESTAMP NOT NULL, "postid" INTEGER NOT NULL, "relatedpostid" INTEGER NOT NULL, "linktypeid" SMALLINT NOT NULL);
CREATE TABLE "Tags" ("id" INTEGER NOT NULL, "tagname" VARCHAR(35), "count" INTEGER NOT NULL, "excerptpostid" INTEGER, "wikipostid" INTEGER, "ismoderatoronly" BOOLEAN, "isrequired" BOOLEAN);
CREATE TABLE "Votes" ("id" INTEGER NOT NULL, "postid" INTEGER NOT NULL, "votetypeid" SMALLINT NOT NULL, "userid" INTEGER, "creationdate" TIMESTAMP, "bountyamount" INTEGER);
"#,
    tables: &[
        ("PostHistoryTypes", "posthistorytypes"),
        ("LinkTypes", "linktypes"),
        ("PostTypes", "posttypes"),
        ("CloseReasonTypes", "closereasontypes"),
        ("VoteTypes", "votetypes"),
        ("Users", "users"),
        ("Badges", "badges"),
        ("Posts", "posts"),
        ("Comments", "comments"),
        ("PostHistory", "posthistory"),
        ("PostLinks", "postlinks"),
        ("Tags", "tags"),
        ("Votes", "votes"),
    ],
    extra_copy_opts: "AUTO_DETECT false, QUOTE '\"', ESCAPE '\"', strict_mode false, ignore_errors true",
};

/// IMDB/JOB data (zstd-compressed tar). Columns are already lowercase
/// upstream so no projection is needed at export time. `ESCAPE '\\'` +
/// `ignore_errors true` tolerate backslash-escaped quotes and dirty rows.
pub const JOB: OriginData = OriginData {
    url: "https://db.in.tum.de/~schmidt/dbgen/job/imdb.tzst",
    archive_name: "imdb.tzst",
    archive: Archive::TarZst,
    log_name: "job",
    ddl: r#"
CREATE TABLE "char_name" ("id" INTEGER, "name" VARCHAR, "imdb_index" VARCHAR, "imdb_id" INTEGER, "name_pcode_nf" VARCHAR, "surname_pcode" VARCHAR, "md5sum" VARCHAR);
CREATE TABLE "company_name" ("id" INTEGER, "name" VARCHAR, "country_code" VARCHAR, "imdb_id" INTEGER, "name_pcode_nf" VARCHAR, "name_pcode_sf" VARCHAR, "md5sum" VARCHAR);
CREATE TABLE "keyword" ("id" INTEGER, "keyword" VARCHAR, "phonetic_code" VARCHAR);
CREATE TABLE "name" ("id" INTEGER, "name" VARCHAR, "imdb_index" VARCHAR, "imdb_id" INTEGER, "gender" VARCHAR, "name_pcode_cf" VARCHAR, "name_pcode_nf" VARCHAR, "surname_pcode" VARCHAR, "md5sum" VARCHAR);
CREATE TABLE "comp_cast_type" ("id" INTEGER, "kind" VARCHAR);
CREATE TABLE "company_type" ("id" INTEGER, "kind" VARCHAR);
CREATE TABLE "info_type" ("id" INTEGER, "info" VARCHAR);
CREATE TABLE "kind_type" ("id" INTEGER, "kind" VARCHAR);
CREATE TABLE "link_type" ("id" INTEGER, "link" VARCHAR);
CREATE TABLE "role_type" ("id" INTEGER, "role" VARCHAR);
CREATE TABLE "title" ("id" INTEGER, "title" VARCHAR, "imdb_index" VARCHAR, "kind_id" INTEGER, "production_year" INTEGER, "imdb_id" INTEGER, "phonetic_code" VARCHAR, "episode_of_id" INTEGER, "season_nr" INTEGER, "episode_nr" INTEGER, "series_years" VARCHAR, "md5sum" VARCHAR);
CREATE TABLE "aka_name" ("id" INTEGER, "person_id" INTEGER, "name" VARCHAR, "imdb_index" VARCHAR, "name_pcode_cf" VARCHAR, "name_pcode_nf" VARCHAR, "surname_pcode" VARCHAR, "md5sum" VARCHAR);
CREATE TABLE "aka_title" ("id" INTEGER, "movie_id" INTEGER, "title" VARCHAR, "imdb_index" VARCHAR, "kind_id" INTEGER, "production_year" INTEGER, "phonetic_code" VARCHAR, "episode_of_id" INTEGER, "season_nr" INTEGER, "episode_nr" INTEGER, "note" VARCHAR, "md5sum" VARCHAR);
CREATE TABLE "cast_info" ("id" INTEGER, "person_id" INTEGER, "movie_id" INTEGER, "person_role_id" INTEGER, "note" VARCHAR, "nr_order" INTEGER, "role_id" INTEGER);
CREATE TABLE "complete_cast" ("id" INTEGER, "movie_id" INTEGER, "subject_id" INTEGER, "status_id" INTEGER);
CREATE TABLE "movie_companies" ("id" INTEGER, "movie_id" INTEGER, "company_id" INTEGER, "company_type_id" INTEGER, "note" VARCHAR);
CREATE TABLE "movie_info" ("id" INTEGER, "movie_id" INTEGER, "info_type_id" INTEGER, "info" VARCHAR, "note" VARCHAR);
CREATE TABLE "movie_info_idx" ("id" INTEGER, "movie_id" INTEGER, "info_type_id" INTEGER, "info" VARCHAR, "note" VARCHAR);
CREATE TABLE "movie_keyword" ("id" INTEGER, "movie_id" INTEGER, "keyword_id" INTEGER);
CREATE TABLE "movie_link" ("id" INTEGER, "movie_id" INTEGER, "linked_movie_id" INTEGER, "link_type_id" INTEGER);
CREATE TABLE "person_info" ("id" INTEGER, "person_id" INTEGER, "info_type_id" INTEGER, "info" VARCHAR, "note" VARCHAR);
"#,
    tables: &[
        ("char_name", "char_name"),
        ("company_name", "company_name"),
        ("keyword", "keyword"),
        ("name", "name"),
        ("comp_cast_type", "comp_cast_type"),
        ("company_type", "company_type"),
        ("info_type", "info_type"),
        ("kind_type", "kind_type"),
        ("link_type", "link_type"),
        ("role_type", "role_type"),
        ("title", "title"),
        ("aka_name", "aka_name"),
        ("aka_title", "aka_title"),
        ("cast_info", "cast_info"),
        ("complete_cast", "complete_cast"),
        ("movie_companies", "movie_companies"),
        ("movie_info", "movie_info"),
        ("movie_info_idx", "movie_info_idx"),
        ("movie_keyword", "movie_keyword"),
        ("movie_link", "movie_link"),
        ("person_info", "person_info"),
    ],
    extra_copy_opts: "ESCAPE '\\', QUOTE '\"', ignore_errors true",
};

/// Table names per origin, in the same order as the corresponding
/// [`OriginData::tables`] output column. Single source of truth shared by
/// [`table_specs`] (used by `SqlstormBenchmark`) and
/// `BenchmarkDataset::tables()` (the registration layer).
pub fn table_names(origin: SqlstormOrigin) -> &'static [&'static str] {
    match origin {
        SqlstormOrigin::TpcH => &[
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ],
        SqlstormOrigin::TpcDs => &[
            "call_center",
            "catalog_page",
            "catalog_returns",
            "catalog_sales",
            "customer",
            "customer_address",
            "customer_demographics",
            "date_dim",
            "household_demographics",
            "income_band",
            "inventory",
            "item",
            "promotion",
            "reason",
            "ship_mode",
            "store",
            "store_returns",
            "store_sales",
            "time_dim",
            "warehouse",
            "web_page",
            "web_returns",
            "web_sales",
            "web_site",
        ],
        SqlstormOrigin::StackOverflow => &[
            "posthistorytypes",
            "linktypes",
            "posttypes",
            "closereasontypes",
            "votetypes",
            "users",
            "badges",
            "posts",
            "comments",
            "posthistory",
            "postlinks",
            "tags",
            "votes",
        ],
        SqlstormOrigin::Job => &[
            "char_name",
            "company_name",
            "keyword",
            "name",
            "comp_cast_type",
            "company_type",
            "info_type",
            "kind_type",
            "link_type",
            "role_type",
            "title",
            "aka_name",
            "aka_title",
            "cast_info",
            "complete_cast",
            "movie_companies",
            "movie_info",
            "movie_info_idx",
            "movie_keyword",
            "movie_link",
            "person_info",
        ],
    }
}

/// Table specs for an origin (schema inferred at registration time — `None`).
pub fn table_specs(origin: SqlstormOrigin) -> Vec<TableSpec> {
    table_names(origin)
        .iter()
        .map(|n| TableSpec::new(n, None))
        .collect()
}

/// Download `cfg.url`, extract the archive, and convert each table to a
/// Parquet shard under `<data_url>/parquet/`. Idempotent via a `.success`
/// marker written after the DuckDB script returns 0.
///
/// Only runs for `file://` data URLs; remote dirs are assumed to already
/// contain the Parquet shards.
pub async fn generate_origin(data_url: &Url, cfg: &OriginData) -> anyhow::Result<()> {
    if data_url.scheme() != "file" {
        return Ok(());
    }

    let base_dir = data_url.to_file_path().map_err(|_| {
        anyhow::anyhow!(
            "Failed to convert data URL to filesystem path — ensure data_url uses 'file://' scheme"
        )
    })?;

    let parquet_dir = base_dir.join(Format::Parquet.name());
    fs::create_dir_all(&parquet_dir)?;

    let success_marker = parquet_dir.join(".success");
    if success_marker.exists() {
        info!(
            "{}: base data already generated ({} present)",
            cfg.log_name,
            success_marker.display(),
        );
        return Ok(());
    }

    let archive_path = download_data(base_dir.join(cfg.archive_name), cfg.url).await?;
    let csv_dir = extract_archive(&archive_path, &base_dir, &cfg.archive)?;
    let script = build_duckdb_script(&csv_dir, &parquet_dir, cfg);

    let output = Command::new("duckdb").arg("-c").arg(&script).output()?;
    if !output.status.success() {
        bail!(
            "duckdb {} COPY failed:\nstdout={}\nstderr={}",
            cfg.log_name,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    fs::write(&success_marker, b"")?;
    info!(
        "{} base data generated in {} ({} Parquet shards)",
        cfg.log_name,
        parquet_dir.display(),
        cfg.tables.len(),
    );
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Extract `archive_path` into `target_dir` and return the directory that
/// holds the resulting CSVs (either `target_dir` itself or a single
/// top-level subdirectory if the archive wraps its contents).
fn extract_archive(
    archive_path: &Path,
    target_dir: &Path,
    archive: &Archive,
) -> anyhow::Result<PathBuf> {
    info!(
        "Extracting {} into {}",
        archive_path.display(),
        target_dir.display()
    );
    let output = match archive {
        Archive::TarGz => Command::new("tar")
            .arg("-xzf")
            .arg(archive_path)
            .arg("--directory")
            .arg(target_dir)
            .output()
            .context("failed to spawn tar; ensure it is on PATH")?,
        // `tar` alone cannot read .tzst, so we pipe via shell.
        Archive::TarZst => Command::new("bash")
            .arg("-c")
            .arg(format!(
                "zstd -dc '{}' | tar -xf - -C '{}'",
                archive_path.display(),
                target_dir.display(),
            ))
            .output()
            .context(
                "failed to spawn bash for zstd/tar extraction; ensure zstd and tar are on PATH",
            )?,
    };
    if !output.status.success() {
        bail!(
            "archive extraction failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    let csv_dir = locate_csv_dir(target_dir)?;
    info!("CSVs located at {}", csv_dir.display());
    Ok(csv_dir)
}

/// Locate the directory holding the extracted CSV files: `target_dir` itself
/// if it has any `.csv` files, else its single subdirectory.
fn locate_csv_dir(target_dir: &Path) -> anyhow::Result<PathBuf> {
    if has_csv(target_dir)? {
        return Ok(target_dir.to_owned());
    }
    for entry in
        fs::read_dir(target_dir).with_context(|| format!("reading {}", target_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && has_csv(&path)? {
            return Ok(path);
        }
    }
    bail!(
        "no CSV files found in {} after extraction; verify the archive contents",
        target_dir.display()
    )
}

fn has_csv(dir: &Path) -> anyhow::Result<bool> {
    for entry in
        fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?
    {
        let entry = entry?;
        if entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("csv"))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Build the DuckDB SQL script: inline DDL, then for each table COPY the CSV
/// in and COPY out to Parquet. The DDL is inlined (not `.read`-ed) because
/// `duckdb -c` does not accept dot-commands.
fn build_duckdb_script(csv_dir: &Path, parquet_dir: &Path, cfg: &OriginData) -> String {
    let mut script = cfg.ddl.to_string();
    let extra = if cfg.extra_copy_opts.is_empty() {
        String::new()
    } else {
        format!(", {}", cfg.extra_copy_opts)
    };
    for (upstream, output) in cfg.tables {
        let csv_path = csv_dir.join(format!("{upstream}.csv"));
        script.push_str(&format!(
            "COPY \"{upstream}\" FROM '{csv}' (FORMAT csv, DELIMITER ',', NULL '', HEADER false{extra});\n",
            csv = csv_path.display(),
        ));
        let out_path = parquet_dir.join(format!("{output}.parquet"));
        script.push_str(&format!(
            "COPY (SELECT * FROM \"{upstream}\") TO '{out}' (FORMAT PARQUET);\n",
            out = out_path.display(),
        ));
    }
    script
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `(upstream, output)` pairs in each origin's `tables` must agree
    /// with the lowercase output names returned by `table_names(origin)`.
    /// Without this guard a future edit could rename one side without the
    /// other; both registration (`BenchmarkDataset::tables()`) and data-gen
    /// (`generate_origin`) would silently disagree.
    #[test]
    fn origin_data_tables_match_table_names() {
        for (origin, cfg) in [
            (SqlstormOrigin::StackOverflow, &STACKOVERFLOW),
            (SqlstormOrigin::Job, &JOB),
        ] {
            let names = table_names(origin);
            let outputs: Vec<&str> = cfg.tables.iter().map(|(_, out)| *out).collect();
            assert_eq!(
                outputs.as_slice(),
                names,
                "{} tables out of sync with table_names()",
                cfg.log_name,
            );
        }
    }

    /// StackOverflow is pinned to the `math` (12 GB) tier, not `dba`.
    #[test]
    fn stackoverflow_uses_math_tier() {
        assert!(
            STACKOVERFLOW.url.ends_with("stackoverflow_math.tar.gz"),
            "url={}",
            STACKOVERFLOW.url
        );
        assert_eq!(STACKOVERFLOW.archive_name, "stackoverflow_math.tar.gz");
    }

    /// Every upstream name in `tables` must have a matching `CREATE TABLE "<name>"`
    /// in the origin's DDL, and vice versa. A drift here (a renamed table on one
    /// side only) would otherwise surface only as a DuckDB COPY failure during
    /// nightly data-gen, never in CI.
    #[test]
    fn origin_data_ddl_tables_match_copy_tables() {
        for cfg in [&STACKOVERFLOW, &JOB] {
            let mut ddl_tables: Vec<&str> = cfg
                .ddl
                .split("CREATE TABLE \"")
                .skip(1)
                .map(|rest| &rest[..rest.find('"').expect("unterminated CREATE TABLE name")])
                .collect();
            ddl_tables.sort_unstable();
            let mut copy_tables: Vec<&str> =
                cfg.tables.iter().map(|(upstream, _)| *upstream).collect();
            copy_tables.sort_unstable();
            assert_eq!(
                ddl_tables, copy_tables,
                "{} DDL CREATE TABLE names disagree with tables[].0",
                cfg.log_name,
            );
        }
    }
}
