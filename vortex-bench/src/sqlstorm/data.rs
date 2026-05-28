// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data acquisition and table specs for SQLStorm origins.
//!
//! `table_names` is the single source of truth for each origin's table list;
//! both `table_specs` (used by `SqlstormBenchmark`) and
//! `BenchmarkDataset::tables()` (used by the registration layer) delegate here.
//!
//! ## StackOverflow identifier case
//!
//! The upstream CSVs have no column header row; column names come from the DDL
//! at `https://db.in.tum.de/~schmidt/data/stackoverflow_schema.sql`, transcribed
//! into the [`STACKOVERFLOW_DDL`] const so data-gen has no schema download to do.
//! That DDL uses camelCase column names (`OwnerUserId`, `CreationDate`, …) and
//! capitalized table names (`Posts`, `Users`, …). The SQLStorm queries reference
//! those names unquoted, which would break under DataFusion's default
//! `enable_ident_normalization=true` (the parser lowercases identifiers while
//! the Parquet schema preserves case → field-not-found).
//!
//! The conversion below lowercases every column at COPY time and the table names in
//! `table_names(StackOverflow)` are already lowercase. Both engines then resolve the
//! verbatim camelCase queries the same way: DataFusion lowercases the query
//! identifiers and matches them against the lowercased Parquet schema, while DuckDB's
//! case-insensitive unquoted identifier resolution makes the original case irrelevant.

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

/// URL for the upstream StackOverflow `dba` data tarball (~1 GB gzip).
const DATA_URL: &str = "https://db.in.tum.de/~schmidt/data/stackoverflow_dba.tar.gz";

/// URL for the upstream JOB (IMDB) data tarball (zstd-compressed tar).
const JOB_DATA_URL: &str = "https://db.in.tum.de/~schmidt/dbgen/job/imdb.tzst";

/// DDL for the 13 StackOverflow tables, transcribed from the upstream
/// `https://db.in.tum.de/~schmidt/data/stackoverflow_schema.sql`. Inlined (rather
/// than fetched at data-gen time) so that data-gen has one fewer network dep and
/// we don't need a line filter to strip `ALTER TABLE ... ADD FOREIGN KEY`
/// statements that the upstream DDL ships and DuckDB rejects. Types and `NOT
/// NULL` constraints are preserved verbatim; inline `references` clauses and
/// `primary key` declarations are stripped because they are not enforced by
/// COPY and would only add noise.
const STACKOVERFLOW_DDL: &str = r#"
CREATE TABLE "PostHistoryTypes" ("Id" SMALLINT NOT NULL, "Name" VARCHAR(50) NOT NULL);
CREATE TABLE "LinkTypes" ("Id" SMALLINT NOT NULL, "Name" VARCHAR(50) NOT NULL);
CREATE TABLE "PostTypes" ("Id" SMALLINT NOT NULL, "Name" VARCHAR(50) NOT NULL);
CREATE TABLE "CloseReasonTypes" ("Id" SMALLINT NOT NULL, "Name" VARCHAR(50) NOT NULL);
CREATE TABLE "VoteTypes" ("Id" SMALLINT NOT NULL, "Name" VARCHAR(50) NOT NULL);
CREATE TABLE "Users" ("Id" INTEGER NOT NULL, "Reputation" INTEGER NOT NULL, "CreationDate" TIMESTAMP NOT NULL, "DisplayName" VARCHAR(40), "LastAccessDate" TIMESTAMP NOT NULL, "WebsiteUrl" VARCHAR(200), "Location" VARCHAR(300), "AboutMe" TEXT, "Views" INTEGER, "UpVotes" INTEGER, "DownVotes" INTEGER, "ProfileImageUrl" VARCHAR(200), "AccountId" INTEGER);
CREATE TABLE "Badges" ("Id" INTEGER NOT NULL, "UserId" INTEGER NOT NULL, "Name" VARCHAR(50) NOT NULL, "Date" TIMESTAMP NOT NULL, "Class" SMALLINT NOT NULL, "TagBased" BOOLEAN NOT NULL);
CREATE TABLE "Posts" ("Id" INTEGER NOT NULL, "PostTypeId" SMALLINT, "AcceptedAnswerId" INTEGER, "ParentId" INTEGER, "CreationDate" TIMESTAMP, "Score" INTEGER, "ViewCount" INTEGER, "Body" TEXT, "OwnerUserId" INTEGER, "OwnerDisplayName" VARCHAR(40), "LastEditorUserId" INTEGER, "LastEditorDisplayName" VARCHAR(40), "LastEditDate" TIMESTAMP, "LastActivityDate" TIMESTAMP, "Title" VARCHAR(300), "Tags" VARCHAR(4000), "AnswerCount" INTEGER, "CommentCount" INTEGER, "FavoriteCount" INTEGER, "ClosedDate" TIMESTAMP, "CommunityOwnedDate" TIMESTAMP, "ContentLicense" VARCHAR(30));
CREATE TABLE "Comments" ("Id" INTEGER NOT NULL, "PostId" INTEGER NOT NULL, "Score" INTEGER, "Text" VARCHAR(2000) NOT NULL, "CreationDate" TIMESTAMP NOT NULL, "UserDisplayName" VARCHAR(40), "UserId" INTEGER, "ContentLicense" VARCHAR(30));
CREATE TABLE "PostHistory" ("Id" INTEGER NOT NULL, "PostHistoryTypeId" SMALLINT, "PostId" INTEGER, "RevisionGUID" VARCHAR(36), "CreationDate" TIMESTAMP, "UserId" INTEGER, "UserDisplayName" VARCHAR(40), "Comment" VARCHAR(800), "Text" TEXT, "ContentLicense" VARCHAR(30));
CREATE TABLE "PostLinks" ("Id" BIGINT NOT NULL, "CreationDate" TIMESTAMP NOT NULL, "PostId" INTEGER NOT NULL, "RelatedPostId" INTEGER NOT NULL, "LinkTypeId" SMALLINT NOT NULL);
CREATE TABLE "Tags" ("Id" INTEGER NOT NULL, "TagName" VARCHAR(35), "Count" INTEGER NOT NULL, "ExcerptPostId" INTEGER, "WikiPostId" INTEGER, "IsModeratorOnly" BOOLEAN, "IsRequired" BOOLEAN);
CREATE TABLE "Votes" ("Id" INTEGER NOT NULL, "PostId" INTEGER NOT NULL, "VoteTypeId" SMALLINT NOT NULL, "UserId" INTEGER, "CreationDate" TIMESTAMP, "BountyAmount" INTEGER);
"#;

/// DDL for the 21 JOB (IMDB) tables, derived from the upstream schema.
///
/// All column names are already lowercase; no projection-lowercasing is needed at export time.
const JOB_DDL: &str = r#"
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
"#;

/// Upstream CamelCase CSV file-stem / table names, in the same order as
/// `table_names(StackOverflow)`. Each entry maps to the corresponding lowercase
/// table name and Parquet output shard.
const UPSTREAM_TABLES: &[&str] = &[
    "PostHistoryTypes",
    "LinkTypes",
    "PostTypes",
    "CloseReasonTypes",
    "VoteTypes",
    "Users",
    "Badges",
    "Posts",
    "Comments",
    "PostHistory",
    "PostLinks",
    "Tags",
    "Votes",
];

/// Column names (original DDL case) for each table in `UPSTREAM_TABLES` order.
///
/// These are derived from `stackoverflow_schema.sql` and are the authoritative
/// source used to build the lowercased projections passed to DuckDB's `COPY`.
const TABLE_COLUMNS: &[&[&str]] = &[
    // PostHistoryTypes
    &["Id", "Name"],
    // LinkTypes
    &["Id", "Name"],
    // PostTypes
    &["Id", "Name"],
    // CloseReasonTypes
    &["Id", "Name"],
    // VoteTypes
    &["Id", "Name"],
    // Users
    &[
        "Id",
        "Reputation",
        "CreationDate",
        "DisplayName",
        "LastAccessDate",
        "WebsiteUrl",
        "Location",
        "AboutMe",
        "Views",
        "UpVotes",
        "DownVotes",
        "ProfileImageUrl",
        "AccountId",
    ],
    // Badges
    &["Id", "UserId", "Name", "Date", "Class", "TagBased"],
    // Posts
    &[
        "Id",
        "PostTypeId",
        "AcceptedAnswerId",
        "ParentId",
        "CreationDate",
        "Score",
        "ViewCount",
        "Body",
        "OwnerUserId",
        "OwnerDisplayName",
        "LastEditorUserId",
        "LastEditorDisplayName",
        "LastEditDate",
        "LastActivityDate",
        "Title",
        "Tags",
        "AnswerCount",
        "CommentCount",
        "FavoriteCount",
        "ClosedDate",
        "CommunityOwnedDate",
        "ContentLicense",
    ],
    // Comments
    &[
        "Id",
        "PostId",
        "Score",
        "Text",
        "CreationDate",
        "UserDisplayName",
        "UserId",
        "ContentLicense",
    ],
    // PostHistory
    &[
        "Id",
        "PostHistoryTypeId",
        "PostId",
        "RevisionGUID",
        "CreationDate",
        "UserId",
        "UserDisplayName",
        "Comment",
        "Text",
        "ContentLicense",
    ],
    // PostLinks
    &[
        "Id",
        "CreationDate",
        "PostId",
        "RelatedPostId",
        "LinkTypeId",
    ],
    // Tags
    &[
        "Id",
        "TagName",
        "Count",
        "ExcerptPostId",
        "WikiPostId",
        "IsModeratorOnly",
        "IsRequired",
    ],
    // Votes
    &[
        "Id",
        "PostId",
        "VoteTypeId",
        "UserId",
        "CreationDate",
        "BountyAmount",
    ],
];

/// Table names per origin (single source of truth).
///
/// TPC-H and TPC-DS mirror the corresponding benchmark's table lists.
/// StackOverflow lists the 13 tables defined by the upstream
/// `stackoverflow_schema.sql` DDL (see `SCHEMA_URL`).
/// JOB lists the 21 tables defined by `JOB_DDL`.
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

/// Download and convert the StackOverflow `dba` (~1 GB) dataset to Parquet.
///
/// Only runs for `file://` data URLs (remote data directories are assumed to already
/// contain the Parquet shards). The 13 typed Parquet files are written into
/// `<base>/parquet/` with lowercase table and column names so that both DataFusion
/// (which lowercases identifiers) and DuckDB (case-insensitive) can query them.
pub async fn generate_stackoverflow(data_url: &Url) -> anyhow::Result<()> {
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

    // Idempotency: skip if all 13 Parquet shards are already present.
    let lowercase_tables = table_names(SqlstormOrigin::StackOverflow);
    if lowercase_tables
        .iter()
        .all(|t| parquet_dir.join(format!("{t}.parquet")).exists())
    {
        info!(
            "stackoverflow: {} Parquet shards already present in {}",
            lowercase_tables.len(),
            parquet_dir.display(),
        );
        return Ok(());
    }

    let tarball_path = download_data(base_dir.join("stackoverflow_dba.tar.gz"), DATA_URL).await?;

    // Extract the tarball. The archive yields files named <UpstreamTable>.csv in the
    // current working directory (or a subdirectory — the extraction target directory
    // is passed as --directory so all contents land under `base_dir`).
    let csv_dir = extract_tarball(&tarball_path, &base_dir)?;

    // Build and run the single DuckDB COPY script:
    //   1. Execute the inlined STACKOVERFLOW_DDL to create the 13 typed tables.
    //   2. COPY each CSV into the corresponding typed table.
    //   3. COPY each table → Parquet with a lowercase column projection.
    let script = build_duckdb_script(&csv_dir, &parquet_dir);

    let output = Command::new("duckdb").arg("-c").arg(&script).output()?;
    if !output.status.success() {
        bail!(
            "duckdb stackoverflow COPY failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    info!(
        "stackoverflow base data generated in {} ({} Parquet shards)",
        parquet_dir.display(),
        lowercase_tables.len(),
    );
    Ok(())
}

/// Download and convert the IMDB/JOB dataset to Parquet.
///
/// Only runs for `file://` data URLs (remote data directories are assumed to already
/// contain the Parquet shards). The 21 Parquet files are written into `<base>/parquet/`.
/// JOB columns are already lowercase in the upstream schema, so no projection-lowercasing
/// is needed at export time.
pub async fn generate_job(data_url: &Url) -> anyhow::Result<()> {
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

    // Idempotency: skip if all 21 Parquet shards are already present.
    let tables = table_names(SqlstormOrigin::Job);
    if tables
        .iter()
        .all(|t| parquet_dir.join(format!("{t}.parquet")).exists())
    {
        info!(
            "job: {} Parquet shards already present in {}",
            tables.len(),
            parquet_dir.display(),
        );
        return Ok(());
    }

    // Download the zstd-compressed tarball into the base directory.
    let tzst_path = download_data(base_dir.join("imdb.tzst"), JOB_DATA_URL).await?;

    // Extract the .tzst archive: zstd decompresses to a tar stream, then tar extracts
    // the CSVs into base_dir. `tar` alone cannot handle .tzst, so we pipe via shell.
    info!(
        "Extracting {} into {}",
        tzst_path.display(),
        base_dir.display()
    );
    let extract_cmd = format!(
        "zstd -dc '{}' | tar -xf - -C '{}'",
        tzst_path.display(),
        base_dir.display()
    );
    let extract_output = Command::new("bash")
        .arg("-c")
        .arg(&extract_cmd)
        .output()
        .context("failed to spawn bash for zstd/tar extraction; ensure zstd and tar are on PATH")?;
    if !extract_output.status.success() {
        bail!(
            "zstd/tar extraction failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&extract_output.stdout),
            String::from_utf8_lossy(&extract_output.stderr),
        );
    }

    // Locate the extracted CSVs. The pinned `imdb.tzst` lands them flat in
    // `base_dir`, but `locate_csv_dir` also handles a wrapping subdirectory so we
    // don't silently break if the upstream archive ever changes layout. Same
    // contract `generate_stackoverflow` already uses.
    let csv_dir = locate_csv_dir(&base_dir)?;

    // Build and run the single DuckDB COPY script:
    //   1. Execute the embedded DDL to create the 21 typed tables.
    //   2. COPY each CSV into its table (backslash escape + ignore_errors for dirty rows).
    //   3. COPY each table → Parquet (columns are already lowercase — no projection needed).
    let script = build_job_duckdb_script(&csv_dir, &parquet_dir, tables);

    let output = Command::new("duckdb").arg("-c").arg(&script).output()?;
    if !output.status.success() {
        bail!(
            "duckdb job COPY failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    info!(
        "job base data generated in {} ({} Parquet shards)",
        parquet_dir.display(),
        tables.len(),
    );
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Extract `tarball_path` (gzip-compressed tar) into `target_dir`.
///
/// Shells out to `tar -xzf <tarball> --directory <target_dir>` so that no
/// additional Cargo dependencies are needed. Returns the directory where the
/// extracted CSVs reside (which equals `target_dir` for flat archives, or the
/// first subdirectory when the archive has a single top-level folder).
fn extract_tarball(tarball_path: &Path, target_dir: &Path) -> anyhow::Result<PathBuf> {
    info!(
        "Extracting {} into {}",
        tarball_path.display(),
        target_dir.display()
    );

    let output = Command::new("tar")
        .arg("-xzf")
        .arg(tarball_path)
        .arg("--directory")
        .arg(target_dir)
        .output()
        .context("failed to spawn tar; ensure it is on PATH")?;

    if !output.status.success() {
        bail!(
            "tar extraction failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    // Look for the CSV files. They may be directly in `target_dir` or inside a
    // single subdirectory if the archive has a top-level folder.
    let csv_dir = locate_csv_dir(target_dir)?;
    info!("CSVs located at {}", csv_dir.display());
    Ok(csv_dir)
}

/// Locate the directory that contains the extracted CSV files.
///
/// Checks `target_dir` itself first. If no `.csv` files are found there, looks
/// one level deeper (a single subdirectory, as some archives include a top-level
/// folder). Returns an error if neither search finds the expected CSVs.
fn locate_csv_dir(target_dir: &Path) -> anyhow::Result<PathBuf> {
    // Check for CSVs directly in target_dir.
    if has_csv(target_dir)? {
        return Ok(target_dir.to_owned());
    }

    // Check for a single subdirectory containing the CSVs.
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
        "no CSV files found in {} after extraction; verify the tarball contents",
        target_dir.display()
    )
}

/// Returns `true` if `dir` contains at least one file ending in `.csv`.
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

/// Build the DuckDB SQL script that:
///
/// 1. Inlines [`STACKOVERFLOW_DDL`] to create the 13 typed tables. (We inline the
///    DDL text rather than use the `.read` dot-command, which `duckdb -c` does
///    not accept.)
/// 2. Loads each upstream CSV into its table with `COPY … FROM`.
/// 3. Exports each table as a Parquet file with all column names lowercased.
///
/// The upstream CSVs have no header row, use comma as delimiter, and represent
/// NULL as the empty string — matching the parameters in the upstream `copy.sql`.
fn build_duckdb_script(csv_dir: &Path, parquet_dir: &Path) -> String {
    let mut script = STACKOVERFLOW_DDL.to_string();

    let lowercase_tables = table_names(SqlstormOrigin::StackOverflow);

    for (i, &upstream) in UPSTREAM_TABLES.iter().enumerate() {
        let csv_path = csv_dir.join(format!("{upstream}.csv"));
        script.push_str(&format!(
            "COPY \"{upstream}\" FROM '{csv}' \
             (DELIMITER ',', FORMAT csv, NULL '', HEADER false);\n",
            csv = csv_path.display(),
        ));

        let columns = TABLE_COLUMNS[i];
        let projection = build_projection(columns);
        let lowercase = lowercase_tables[i];
        let out_path = parquet_dir.join(format!("{lowercase}.parquet"));
        script.push_str(&format!(
            "COPY (SELECT {projection} FROM \"{upstream}\") \
             TO '{out}' (FORMAT PARQUET);\n",
            out = out_path.display(),
        ));
    }

    script
}

/// Build the DuckDB SQL script for the JOB (IMDB) dataset that:
///
/// 1. Inlines the embedded DDL to create the 21 typed tables.
/// 2. Loads each upstream CSV with `COPY … FROM`, using `ESCAPE '\'` and
///    `ignore_errors true` to handle backslash-escaped quotes and dirty rows.
/// 3. Exports each table as a Parquet file (columns are already lowercase).
fn build_job_duckdb_script(csv_dir: &Path, parquet_dir: &Path, tables: &[&str]) -> String {
    let mut script = JOB_DDL.to_string();

    for &table in tables {
        let csv_path = csv_dir.join(format!("{table}.csv"));
        script.push_str(&format!(
            "COPY \"{table}\" FROM '{csv}' \
             (FORMAT csv, DELIMITER ',', NULL '', HEADER false, ESCAPE '\\', QUOTE '\"', ignore_errors true);\n",
            csv = csv_path.display(),
        ));
    }

    for &table in tables {
        let out_path = parquet_dir.join(format!("{table}.parquet"));
        script.push_str(&format!(
            "COPY (SELECT * FROM \"{table}\") TO '{out}' (FORMAT PARQUET);\n",
            out = out_path.display(),
        ));
    }

    script
}

/// Build a DuckDB SELECT projection string that lowercases every column name.
///
/// Each column `"Col"` becomes `"Col" AS "col"`.
fn build_projection(columns: &[&str]) -> String {
    columns
        .iter()
        .map(|c| format!("\"{}\" AS \"{}\"", c, c.to_lowercase()))
        .collect::<Vec<_>>()
        .join(", ")
}
