// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Full scans of real Vortex datasets (TPC-H, TPC-DS) through an object store.
//!
//! Every table is opened and scanned end to end with `vortex-file`, decoding each chunk to its
//! canonical form, so the runtime includes both the IO and the decompression of a full scan. The
//! same command runs against real S3 or against a [`crate::simulated::LatencyStore`] over an
//! in-memory copy of the files, so the two can be compared table by table.

use std::collections::BTreeMap;
use std::path::Path as FsPath;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Instant;

use anyhow::Context;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::PutPayload;
use object_store::path::Path;
use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| VortexSession::default().with_tokio());

#[derive(Debug, Serialize)]
pub struct TableScan {
    pub table: String,
    pub files: usize,
    pub bytes: u64,
    pub rows: u64,
    /// Wall time of each iteration's full scan of the table, in seconds.
    pub runs_s: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct DatasetScan {
    pub dataset: String,
    pub store: String,
    pub iterations: usize,
    pub tables: Vec<TableScan>,
}

/// The table a file belongs to: its first directory under the dataset prefix, or for files
/// directly under it, the file stem without a trailing partition number (`lineitem_3`).
fn table_name(relative: &str) -> String {
    let mut parts = relative.split('/');
    let first = parts.next().unwrap_or(relative);
    if parts.next().is_some() {
        return first.to_string();
    }
    let stem = first.strip_suffix(".vortex").unwrap_or(first);
    match stem.rsplit_once(['_', '-']) {
        Some((table, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
            table.to_string()
        }
        _ => stem.to_string(),
    }
}

/// Copy every `.vortex` file under `dir` into `store` under `prefix`, keeping relative paths.
pub async fn load_dir(
    store: &Arc<dyn ObjectStore>,
    prefix: &Path,
    dir: &FsPath,
) -> anyhow::Result<u64> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(d) = pending.pop() {
        for entry in std::fs::read_dir(&d).with_context(|| format!("reading {}", d.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == "vortex") {
                files.push(path);
            }
        }
    }
    let mut total = 0;
    for path in files {
        let relative = path.strip_prefix(dir)?.to_string_lossy().replace('\\', "/");
        let bytes = tokio::fs::read(&path).await?;
        total += bytes.len() as u64;
        let mut location = prefix.clone();
        for part in relative.split('/') {
            location = location.join(part);
        }
        store.put(&location, PutPayload::from(bytes)).await?;
    }
    Ok(total)
}

/// Scan one file end to end, returning the number of rows.
async fn scan_file(store: &Arc<dyn ObjectStore>, path: Path) -> anyhow::Result<u64> {
    let file = SESSION
        .open_options()
        .open_object_store(store, path)
        .await?;
    let ctx = SESSION.create_execution_ctx();
    let rows = file
        .scan()?
        .map(move |array| {
            let mut ctx = ctx.clone();
            Ok(array.execute::<Canonical>(&mut ctx)?.into_array())
        })
        .into_array_stream()?
        .try_fold(
            0u64,
            |rows, array| async move { Ok(rows + array.len() as u64) },
        )
        .await?;
    Ok(rows)
}

/// Fully scan every table of the dataset under `prefix`, `iterations` times each. The files of a
/// table are scanned concurrently, as a query engine would; tables run one after another.
pub async fn scan_dataset(
    store: &Arc<dyn ObjectStore>,
    prefix: &Path,
    dataset: &str,
    store_kind: &str,
    iterations: usize,
    file_concurrency: usize,
) -> anyhow::Result<DatasetScan> {
    let objects: Vec<_> = store
        .list(Some(prefix))
        .try_filter(|m| futures::future::ready(m.location.as_ref().ends_with(".vortex")))
        .try_collect()
        .await?;
    anyhow::ensure!(!objects.is_empty(), "no .vortex files under {prefix}");
    let mut tables: BTreeMap<String, Vec<(Path, u64)>> = BTreeMap::new();
    for meta in objects {
        let relative = meta
            .location
            .prefix_match(prefix)
            .map(|parts| {
                parts
                    .map(|p| p.as_ref().to_string())
                    .collect::<Vec<_>>()
                    .join("/")
            })
            .unwrap_or_else(|| meta.location.to_string());
        tables
            .entry(table_name(&relative))
            .or_default()
            .push((meta.location, meta.size));
    }

    let mut results = Vec::new();
    for (table, files) in tables {
        let bytes = files.iter().map(|(_, size)| size).sum();
        let mut runs_s = Vec::with_capacity(iterations);
        let mut rows = 0;
        for _ in 0..iterations {
            let start = Instant::now();
            rows = stream::iter(files.iter().map(|(path, _)| scan_file(store, path.clone())))
                .buffer_unordered(file_concurrency.max(1))
                .try_fold(0u64, |acc, n| async move { Ok(acc + n) })
                .await?;
            runs_s.push(start.elapsed().as_secs_f64());
        }
        println!(
            "  {dataset:<10} {table:<24} {:>4} files {:>9.1} MiB {:>11} rows  median {:>7.2}s",
            files.len(),
            bytes as f64 / f64::from(1 << 20),
            rows,
            median(&runs_s),
        );
        results.push(TableScan {
            table,
            files: files.len(),
            bytes,
            rows,
            runs_s,
        });
    }
    Ok(DatasetScan {
        dataset: dataset.to_string(),
        store: store_kind.to_string(),
        iterations,
        tables: results,
    })
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_names() {
        assert_eq!(table_name("lineitem/part-0.vortex"), "lineitem");
        assert_eq!(table_name("lineitem_3.vortex"), "lineitem");
        assert_eq!(table_name("store_sales.vortex"), "store_sales");
        assert_eq!(table_name("customer-12.vortex"), "customer");
        assert_eq!(table_name("web_site.vortex"), "web_site");
    }
}
