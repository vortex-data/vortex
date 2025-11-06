// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Standardized URL creation and management
//!
//! This module provides consistent patterns for creating URLs
//! for benchmark data, avoiding duplication and inconsistencies.

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use tracing::{info, warn};
use url::Url;

use crate::IdempotentPath;

/// Create a data URL for a benchmark
///
/// This function handles both local file paths and remote URLs consistently.
///
/// # Arguments
/// * `benchmark_name` - Name of the benchmark (e.g., "tpch", "clickbench")
/// * `variant` - Optional variant identifier (e.g., scale factor, flavor)
/// * `remote_url` - Optional remote URL override
///
/// # Returns
/// A URL pointing to the benchmark data location
pub fn benchmark_data_url(
    benchmark_name: &str,
    variant: Option<&str>,
    remote_url: &Option<String>,
) -> Result<Url> {
    match remote_url {
        None => {
            // Local file path
            let mut data_path = benchmark_name.to_data_path();

            // Add variant subdirectory if specified
            if let Some(v) = variant {
                data_path = data_path.join(v);
            }

            Url::from_directory_path(&data_path).map_err(|_| {
                anyhow!(
                    "Failed to create URL from directory path: {:?}",
                    &data_path
                )
            })
        }
        Some(remote) => {
            // Remote URL
            if !remote.ends_with('/') {
                warn!(
                    "Remote URL should end with a slash for proper path joining: {}",
                    remote
                );
            }

            info!(
                concat!(
                    "Using remote data URL: {}\n",
                    "Assuming data already exists at this location.\n",
                    "If not, generate locally first and upload to the remote location."
                ),
                remote
            );

            // Parse and optionally append variant
            let mut url = Url::parse(remote)?;
            if let Some(v) = variant {
                url = url.join(&format!("{}/", v))?;
            }

            Ok(url)
        }
    }
}

/// Create a format-specific URL
///
/// Given a base URL and format, creates the appropriate subdirectory URL.
pub fn format_data_url(base_url: &Url, format: crate::Format) -> Result<Url> {
    Ok(base_url.join(&format!("{}/", format.name()))?)
}

/// Convert a URL to a file path
///
/// Helper for safely converting file:// URLs to PathBuf.
pub fn url_to_path(url: &Url) -> Result<PathBuf> {
    url.to_file_path()
        .map_err(|_| anyhow!("Invalid file URL: {}", url))
}

/// Check if a URL points to local storage
pub fn is_local_url(url: &Url) -> bool {
    url.scheme() == "file"
}

/// Check if a URL points to remote storage (S3, GCS, etc.)
pub fn is_remote_url(url: &Url) -> bool {
    matches!(url.scheme(), "s3" | "gs" | "gcs" | "http" | "https")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_benchmark_url() {
        let url = benchmark_data_url("tpch", Some("sf1"), &None).unwrap();
        assert!(url.scheme() == "file");
        assert!(url.path().contains("tpch"));
        assert!(url.path().contains("sf1"));
    }

    #[test]
    fn test_remote_benchmark_url() {
        let remote = Some("s3://bucket/data/".to_string());
        let url = benchmark_data_url("tpch", Some("sf1"), &remote).unwrap();
        assert!(url.scheme() == "s3");
        assert!(url.path().contains("sf1"));
    }

    #[test]
    fn test_format_url() {
        let base = Url::parse("file:///data/tpch/").unwrap();
        let format_url = format_data_url(&base, crate::Format::Parquet).unwrap();
        assert!(format_url.path().ends_with("/parquet/"));
    }
}