// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming readers for v2's public S3 bucket.
//!
//! The bucket is `--no-sign-request`, so we fetch the underlying
//! HTTPS URL directly and stream-decompress with `flate2`. The
//! downloads are wrapped in [`reqwest::blocking`] to keep the read
//! path synchronous; the binary's hot path is single-threaded
//! per-source already (DuckDB is a single-writer).
//!
//! For tests and offline runs, [`Source::Local`](crate::source::Source::Local) accepts a local
//! directory of dumps; the migrator's `--source` flag picks the
//! variant.

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use flate2::read::GzDecoder;
use tracing::info;

/// Public S3 bucket the live v2 server reads from.
pub const PUBLIC_BUCKET_BASE: &str = "https://vortex-ci-benchmark-results.s3.amazonaws.com";

/// Where to read the v2 dataset from. Either the public S3 bucket
/// (the live deployment) or a local directory of dumps.
#[derive(Debug, Clone)]
pub enum Source {
    /// HTTPS GETs against `s3.amazonaws.com`.
    PublicS3,
    /// A directory containing `data.json.gz`, `commits.json`, and
    /// `file-sizes-*.json.gz` files.
    Local(PathBuf),
}

impl Source {
    /// Short human-readable description for log messages.
    pub fn describe(&self) -> String {
        match self {
            Source::PublicS3 => "public S3 bucket".to_string(),
            Source::Local(p) => format!("local dir {}", p.display()),
        }
    }

    /// Open `data.json.gz` for streaming, decompressing on the fly.
    pub fn open_data_jsonl(&self) -> Result<Box<dyn BufRead>> {
        let stream = self.open_raw("data.json.gz")?;
        Ok(Box::new(BufReader::new(GzDecoder::new(stream))))
    }

    /// Open `commits.json` (uncompressed).
    pub fn open_commits_jsonl(&self) -> Result<Box<dyn BufRead>> {
        let stream = self.open_raw("commits.json")?;
        Ok(Box::new(BufReader::new(stream)))
    }

    /// Enumerate `file-sizes-*.json.gz` files. For local sources this
    /// is a directory glob; for the public bucket we hit the documented
    /// suite ids.
    pub fn list_file_sizes(&self) -> Result<Vec<String>> {
        match self {
            Source::Local(dir) => {
                let mut out = Vec::new();
                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    let name = entry.file_name();
                    let s = name.to_string_lossy();
                    if s.starts_with("file-sizes-") && s.ends_with(".json.gz") {
                        out.push(s.into_owned());
                    }
                }
                out.sort();
                Ok(out)
            }
            Source::PublicS3 => {
                // The S3 bucket's ListObjects is denied for unsigned
                // requests, so we hit the documented per-suite keys
                // emitted by `.github/workflows/sql-benchmarks.yml`.
                Ok(KNOWN_FILE_SIZES_SUITES
                    .iter()
                    .map(|id| format!("file-sizes-{id}.json.gz"))
                    .collect())
            }
        }
    }

    /// Open one `file-sizes-*.json.gz` for streaming.
    pub fn open_file_sizes(&self, name: &str) -> Result<Box<dyn BufRead>> {
        let stream = self.open_raw(name)?;
        Ok(Box::new(BufReader::new(GzDecoder::new(stream))))
    }

    fn open_raw(&self, name: &str) -> Result<Box<dyn Read + Send>> {
        match self {
            Source::Local(dir) => open_local(&dir.join(name)),
            Source::PublicS3 => open_s3(name),
        }
    }
}

fn open_local(path: &Path) -> Result<Box<dyn Read + Send>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(Box::new(f))
}

fn open_s3(name: &str) -> Result<Box<dyn Read + Send>> {
    let url = format!("{PUBLIC_BUCKET_BASE}/{name}");
    info!(url = %url, "GET");
    let resp = reqwest::blocking::get(&url).with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("GET {url} returned {}", resp.status());
    }
    Ok(Box::new(resp))
}

/// Suite IDs we know publish a `file-sizes-{id}.json.gz` to S3.
///
/// Source of truth: the `matrix.id` values in
/// `.github/workflows/sql-benchmarks.yml`'s `benchmark_matrix` default.
/// The post-bench `file-sizes` step uploads `file-sizes-${{ matrix.id
/// }}.json.gz`, so this list must match those IDs verbatim. Adding a
/// new matrix entry to that workflow means adding the same ID here.
pub(crate) const KNOWN_FILE_SIZES_SUITES: &[&str] = &[
    "clickbench-nvme",
    "tpch-nvme",
    "tpch-s3",
    "tpch-nvme-10",
    "tpch-s3-10",
    "tpcds-nvme",
    "statpopgen",
    "fineweb",
    "fineweb-s3",
    "polarsignals",
];
