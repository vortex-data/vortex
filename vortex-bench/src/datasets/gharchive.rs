// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GitHub Archive dataset for random access benchmarks.
//!
//! This module provides functions to generate and access the GitHub Archive dataset
//! in both Parquet and Vortex formats. The dataset contains deeply nested event data
//! which is useful for benchmarking nested field access patterns.

use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use async_trait::async_trait;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use tracing::info;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;

use crate::CompactionStrategy;
use crate::IdempotentPath;
use crate::SESSION;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::idempotent;
use crate::idempotent_async;

/// Template URL for raw JSON dataset.
fn raw_json_url(hour: usize) -> String {
    assert!(hour <= 23);
    format!("https://data.gharchive.org/2024-10-01-{hour}.json.gz")
}

pub struct GhArchiveData;

#[async_trait]
impl Dataset for GhArchiveData {
    fn name(&self) -> &str {
        "gharchive"
    }

    async fn to_vortex_array(&self) -> Result<ArrayRef> {
        fetch_gharchive_data().await
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        gharchive_parquet().await
    }
}

/// Get the path to the compressed JSON data.
fn gharchive_json_path() -> PathBuf {
    "gharchive/json/events.json.gz".to_data_path()
}

/// Get the path to the Parquet file.
fn gharchive_parquet_path() -> PathBuf {
    "gharchive/parquet/events.parquet".to_data_path()
}

/// Download the GitHub Archive JSON data for all 24 hours of 2024-10-01.
pub async fn gharchive_json() -> Result<PathBuf> {
    idempotent_async(&gharchive_json_path(), |json_path| async move {
        info!("Downloading GithubArchive JSON source files");
        let mut w = TokioFile::create(&json_path).await?;
        let client = reqwest::Client::new();
        for hour in 0..=23 {
            let url = raw_json_url(hour);
            info!("Downloading archive {url}");
            let response = client
                .get(url)
                .send()
                .await?
                .error_for_status()
                .map_err(|err| anyhow::anyhow!("error fetching gharchive data: {err}"))?;

            let body = response.bytes().await?;

            w.write_all(&body).await?;
            w.flush().await?;
        }

        Ok(json_path)
    })
    .await
}

/// Get the path to the Parquet file, generating it from JSON if necessary.
///
/// This uses DuckDB to convert the JSON data to Parquet format.
pub async fn gharchive_parquet() -> Result<PathBuf> {
    let json = gharchive_json().await?;
    let json_path_str = json.display().to_string();

    idempotent(&gharchive_parquet_path(), move |parquet_path| {
        let parquet_str = parquet_path.display().to_string();
        info!(
            "Converting GithubArchive JSON to Parquet with DuckDB @ {}",
            parquet_path.display()
        );
        let result = Command::new("duckdb")
            .arg("-c")
            .arg(format!(
                "
                CREATE TABLE events AS select * from read_ndjson_auto('{json_path_str}', ignore_errors = true);
                COPY events TO '{parquet_str}' (FORMAT parquet);
                "
            ))
            .spawn()?
            .wait()?;

        if !result.success() {
            anyhow::bail!("DuckDB subprocess failed converting JSON to Parquet");
        }

        Ok(())
    })
}

/// Load the GitHub Archive data as a Vortex array.
pub async fn fetch_gharchive_data() -> Result<ArrayRef> {
    let vortex_data = gharchive_vortex().await?;
    Ok(SESSION
        .open_options()
        .open_path(vortex_data)
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?)
}

/// Get the path to the Vortex file, converting from Parquet if necessary.
pub async fn gharchive_vortex() -> Result<PathBuf> {
    idempotent_async(
        "gharchive/vortex/events.vortex",
        |output_fname| async move {
            let buf = output_fname.to_path_buf();
            let mut output_file = TokioFile::create(output_fname).await?;

            let data = parquet_to_vortex_chunks(gharchive_parquet().await?).await?;

            SESSION
                .write_options()
                .write(&mut output_file, data.to_array_stream())
                .await?;
            output_file.flush().await?;
            Ok(buf)
        },
    )
    .await
}

/// Get the path to a compact Vortex file, converting from Parquet if necessary.
pub async fn gharchive_vortex_compact() -> Result<PathBuf> {
    idempotent_async(
        "gharchive/vortex/events-compact.vortex",
        |output_fname| async move {
            let buf = output_fname.to_path_buf();
            let mut output_file = TokioFile::create(output_fname).await?;

            let write_options = CompactionStrategy::Compact.apply_options(SESSION.write_options());

            let data = parquet_to_vortex_chunks(gharchive_parquet().await?).await?;

            write_options
                .write(&mut output_file, data.to_array_stream())
                .await?;

            output_file.flush().await?;
            Ok(buf)
        },
    )
    .await
}

/// Deeply nested fields in the GitHub Archive dataset that are useful for benchmarking.
///
/// These fields represent common access patterns for nested data:
/// - `payload.ref` - String field nested under payload struct
/// - `repo.name` - String field nested under repo struct
/// - `actor.login` - String field nested under actor struct
/// - `org.id` - Integer field nested under org struct
pub const NESTED_FIELDS: &[(&str, &str)] = &[
    ("payload", "ref"),
    ("repo", "name"),
    ("actor", "login"),
    ("org", "id"),
];
