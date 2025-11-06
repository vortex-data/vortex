// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Standardized query loading and management
//!
//! This module provides consistent patterns for loading benchmark queries
//! from various sources (files, directories, embedded strings).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use tracing::info;

/// Source for benchmark queries
#[derive(Debug, Clone)]
pub enum QuerySource {
    /// Load queries from a directory of SQL files
    Directory(PathBuf),
    /// Load queries from a single SQL file (with delimiters)
    SingleFile(PathBuf),
    /// Use an override file instead of default queries
    Override(PathBuf),
    /// Use embedded queries (provided directly)
    Embedded(Vec<(usize, String)>),
}

impl QuerySource {
    /// Create a directory source
    pub fn directory(path: impl Into<PathBuf>) -> Self {
        Self::Directory(path.into())
    }

    /// Create a single file source
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::SingleFile(path.into())
    }

    /// Create an override source
    pub fn override_with(path: impl Into<PathBuf>) -> Self {
        Self::Override(path.into())
    }

    /// Create an embedded source
    pub fn embedded(queries: Vec<(usize, String)>) -> Self {
        Self::Embedded(queries)
    }
}

/// Load queries from the specified source
///
/// Returns a vector of (query_number, query_sql) tuples.
pub fn load_queries(source: QuerySource) -> Result<Vec<(usize, String)>> {
    match source {
        QuerySource::Directory(dir) => load_queries_from_directory(&dir),
        QuerySource::SingleFile(file) => load_queries_from_file(&file),
        QuerySource::Override(file) => {
            info!("Using override query file: {:?}", file);
            load_queries_from_file(&file)
        }
        QuerySource::Embedded(queries) => Ok(queries),
    }
}

/// Load queries from a directory of SQL files
///
/// Expects files named like "q1.sql", "q2.sql", etc.
fn load_queries_from_directory(dir: &Path) -> Result<Vec<(usize, String)>> {
    if !dir.exists() {
        return Err(anyhow!("Query directory does not exist: {:?}", dir));
    }

    let mut queries = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "sql"))
        .collect();

    // Sort by filename to ensure consistent ordering
    entries.sort_by_key(|e| e.file_name());

    for (idx, entry) in entries.iter().enumerate() {
        let path = entry.path();
        let query = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read query file: {:?}", path))?;

        // Extract query number from filename (e.g., "q1.sql" -> 1)
        let query_num = extract_query_number(&path).unwrap_or(idx + 1);

        queries.push((query_num, query.trim().to_string()));
    }

    if queries.is_empty() {
        return Err(anyhow!("No SQL files found in directory: {:?}", dir));
    }

    Ok(queries)
}

/// Load queries from a single file
///
/// Expects queries separated by semicolons or special delimiters.
fn load_queries_from_file(file: &Path) -> Result<Vec<(usize, String)>> {
    if !file.exists() {
        return Err(anyhow!("Query file does not exist: {:?}", file));
    }

    let content = fs::read_to_string(file)
        .with_context(|| format!("Failed to read query file: {:?}", file))?;

    // Try to parse as a delimited file first
    if let Ok(queries) = parse_delimited_queries(&content) {
        return Ok(queries);
    }

    // Otherwise treat as a single query
    Ok(vec![(1, content.trim().to_string())])
}

/// Parse queries from a delimited string
///
/// Supports various delimiter formats:
/// - Semicolon-separated
/// - Comment-delimited (e.g., "-- Query 1")
fn parse_delimited_queries(content: &str) -> Result<Vec<(usize, String)>> {
    let mut queries = Vec::new();

    // Check for comment delimiters first
    if content.contains("-- Query") || content.contains("-- Q") {
        let parts: Vec<&str> = content.split("-- Q").collect();
        for (idx, part) in parts.iter().skip(1).enumerate() {
            let query = part
                .lines()
                .skip(1) // Skip the comment line
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();

            if !query.is_empty() {
                queries.push((idx + 1, query));
            }
        }
    } else {
        // Fall back to semicolon separation
        let parts: Vec<&str> = content.split(';').collect();
        for (idx, part) in parts.iter().enumerate() {
            let query = part.trim().to_string();
            if !query.is_empty() {
                queries.push((idx + 1, query));
            }
        }
    }

    if queries.is_empty() {
        return Err(anyhow!("No queries found in content"));
    }

    Ok(queries)
}

/// Extract query number from filename
///
/// Handles patterns like "q1.sql", "query1.sql", "01.sql"
fn extract_query_number(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_str()?;

    // Try different patterns
    if let Some(num_str) = stem.strip_prefix("query") {
        return num_str.parse().ok();
    }

    if let Some(num_str) = stem.strip_prefix("q") {
        return num_str.parse().ok();
    }

    // Try parsing the whole stem as a number
    stem.parse().ok()
}

/// Filter queries by index
///
/// Useful for running only specific queries from a benchmark.
pub fn filter_queries(queries: Vec<(usize, String)>, indices: &[usize]) -> Vec<(usize, String)> {
    if indices.is_empty() {
        return queries;
    }

    queries
        .into_iter()
        .filter(|(idx, _)| indices.contains(idx))
        .collect()
}

/// Format query for display
///
/// Truncates long queries for readability in logs.
pub fn format_query_display(query: &str, max_len: usize) -> String {
    if query.len() <= max_len {
        query.to_string()
    } else {
        format!("{}...", &query[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_semicolon_queries() {
        let content = "SELECT 1; SELECT 2; SELECT 3";
        let queries = parse_delimited_queries(content).unwrap();
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0].1, "SELECT 1");
    }

    #[test]
    fn test_parse_comment_queries() {
        let content = "-- Query 1\nSELECT 1\n-- Query 2\nSELECT 2";
        let queries = parse_delimited_queries(content).unwrap();
        assert_eq!(queries.len(), 2);
    }

    #[test]
    fn test_extract_query_number() {
        assert_eq!(extract_query_number(Path::new("q1.sql")), Some(1));
        assert_eq!(extract_query_number(Path::new("query10.sql")), Some(10));
        assert_eq!(extract_query_number(Path::new("05.sql")), Some(5));
    }

    #[test]
    fn test_filter_queries() {
        let queries = vec![
            (1, "SELECT 1".to_string()),
            (2, "SELECT 2".to_string()),
            (3, "SELECT 3".to_string()),
        ];

        let filtered = filter_queries(queries.clone(), &[1, 3]);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, 1);
        assert_eq!(filtered[1].0, 3);
    }
}
