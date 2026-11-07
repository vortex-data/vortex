// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Clustering TPC-DS fact tables by `(date_sk, item_sk)` with an external merge sort.
//!
//! Generated batches are buffered into runs of about a fixed number of bytes. Each run is sorted
//! in memory and spilled to an LZ4-compressed Arrow IPC file; the runs are then merged with a
//! k-way heap merge that holds one small batch per run. Memory is bounded by the run budget plus
//! `runs * SPILL_BATCH_ROWS` rows, independent of the table size, while the spill files take
//! about the table's compressed Arrow size on disk.
//!
//! The sort is stable: rows with equal keys keep their generation order, so the output is
//! deterministic.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::mem;
use std::path::Path;
use std::path::PathBuf;
use std::pin::pin;

use anyhow::Result;
use anyhow::anyhow;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_cast::cast;
use arrow_ipc::CompressionType;
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::DataType;
use arrow_select::interleave::interleave_record_batch;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future;
use futures::stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tpcdsgen::config::Table;

/// Rows per batch in the spilled runs, and so per run held in memory while merging.
const SPILL_BATCH_ROWS: usize = 8192;

/// The columns a fact table is clustered by: its date key, then its item key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ClusterColumns {
    pub date: &'static str,
    pub item: &'static str,
}

/// The clustering columns of `table`, or `None` for dimension tables, which stay in generation
/// order.
pub(super) fn cluster_columns(table: Table) -> Option<ClusterColumns> {
    let (date, item) = match table {
        Table::StoreSales => ("ss_sold_date_sk", "ss_item_sk"),
        Table::StoreReturns => ("sr_returned_date_sk", "sr_item_sk"),
        Table::CatalogSales => ("cs_sold_date_sk", "cs_item_sk"),
        Table::CatalogReturns => ("cr_returned_date_sk", "cr_item_sk"),
        Table::WebSales => ("ws_sold_date_sk", "ws_item_sk"),
        Table::WebReturns => ("wr_returned_date_sk", "wr_item_sk"),
        Table::Inventory => ("inv_date_sk", "inv_item_sk"),
        _ => return None,
    };
    Some(ClusterColumns { date, item })
}

/// The sort key of one row: ascending date then item, each with nulls last.
type SortKey = (bool, i64, bool, i64);

/// The sort keys of every row of `batch`.
fn sort_keys(batch: &RecordBatch, columns: ClusterColumns) -> Result<Vec<SortKey>> {
    let column = |name: &str| -> Result<_> {
        let array = batch
            .column_by_name(name)
            .ok_or_else(|| anyhow!("missing clustering column {name}"))?;
        Ok(cast(array, &DataType::Int64)?)
    };
    let date = column(columns.date)?;
    let item = column(columns.item)?;
    let date = date.as_primitive::<Int64Type>();
    let item = item.as_primitive::<Int64Type>();
    Ok(date
        .iter()
        .zip(item.iter())
        .map(|(date, item)| {
            (
                date.is_none(),
                date.unwrap_or_default(),
                item.is_none(),
                item.unwrap_or_default(),
            )
        })
        .collect())
}

/// Sort the batches of `chunks` by `columns`, spilling sorted runs of about `run_bytes` into
/// `spill_dir`, and stream the merged rows in batches of `batch_size` rows.
pub(super) async fn sort_batches(
    chunks: impl Stream<Item = Result<Vec<RecordBatch>>>,
    columns: ClusterColumns,
    spill_dir: &Path,
    run_bytes: usize,
    batch_size: usize,
) -> Result<impl Stream<Item = Result<Vec<RecordBatch>>>> {
    let runs = spill_sorted_runs(chunks, columns, spill_dir, run_bytes).await?;
    Ok(merge_runs(runs, columns, batch_size))
}

/// Buffer `chunks` into runs of at least `run_bytes`, and spill each run, sorted, to `spill_dir`.
async fn spill_sorted_runs(
    chunks: impl Stream<Item = Result<Vec<RecordBatch>>>,
    columns: ClusterColumns,
    spill_dir: &Path,
    run_bytes: usize,
) -> Result<Vec<PathBuf>> {
    let mut chunks = pin!(chunks);
    let mut runs = Vec::new();
    let mut run = Vec::new();
    let mut bytes = 0;
    loop {
        let next = chunks.try_next().await?;
        let done = next.is_none();
        for batch in next.into_iter().flatten() {
            bytes += batch.get_array_memory_size();
            run.push(batch);
        }
        if (done && !run.is_empty()) || bytes >= run_bytes {
            let path = spill_dir.join(format!("run_{:06}.arrow", runs.len()));
            let batches = mem::take(&mut run);
            bytes = 0;
            let spilled = path.clone();
            tokio::task::spawn_blocking(move || spill_run(batches, columns, &spilled)).await??;
            runs.push(path);
        }
        if done {
            return Ok(runs);
        }
    }
}

/// Write the rows of `batches`, stably sorted by `columns`, to an Arrow IPC file at `path`.
fn spill_run(batches: Vec<RecordBatch>, columns: ClusterColumns, path: &Path) -> Result<()> {
    let schema = batches
        .first()
        .map(RecordBatch::schema)
        .ok_or_else(|| anyhow!("cannot spill an empty run"))?;

    // Sorting on the source position as well makes the unstable sort stable.
    let mut rows = Vec::with_capacity(batches.iter().map(RecordBatch::num_rows).sum());
    for (batch_idx, batch) in batches.iter().enumerate() {
        rows.extend(
            sort_keys(batch, columns)?
                .into_iter()
                .enumerate()
                .map(|(row, key)| (key, batch_idx, row)),
        );
    }
    rows.sort_unstable();

    let options =
        IpcWriteOptions::default().try_with_compression(Some(CompressionType::LZ4_FRAME))?;
    let mut writer =
        FileWriter::try_new_with_options(BufWriter::new(File::create(path)?), &schema, options)?;
    let batches: Vec<&RecordBatch> = batches.iter().collect();
    let mut indices = Vec::with_capacity(SPILL_BATCH_ROWS);
    for chunk in rows.chunks(SPILL_BATCH_ROWS) {
        indices.clear();
        indices.extend(chunk.iter().map(|&(_, batch, row)| (batch, row)));
        writer.write(&interleave_record_batch(&batches, &indices)?)?;
    }
    writer.finish()?;
    Ok(())
}

/// Merge the sorted `runs` into batches of `batch_size` rows, on a blocking thread.
///
/// The stream ends with the merge's error, if any, so a failed merge cannot pass for a short
/// table.
fn merge_runs(
    runs: Vec<PathBuf>,
    columns: ClusterColumns,
    batch_size: usize,
) -> impl Stream<Item = Result<Vec<RecordBatch>>> {
    let (tx, rx) = mpsc::channel(2);
    let merge =
        tokio::task::spawn_blocking(move || merge_runs_blocking(&runs, columns, batch_size, &tx));
    ReceiverStream::new(rx).map(|batch| Ok(vec![batch])).chain(
        stream::once(async move {
            match merge.await {
                Ok(Ok(())) => None,
                Ok(Err(err)) => Some(Err(err)),
                Err(err) => Some(Err(err.into())),
            }
        })
        .filter_map(future::ready),
    )
}

/// A sorted run being merged.
struct Cursor {
    reader: FileReader<BufReader<File>>,
    /// Index of the current batch in the merge's source batches.
    source: usize,
    keys: Vec<SortKey>,
    row: usize,
}

impl Cursor {
    /// Advance to the next non-empty batch of the run, returning it, or `None` at its end.
    fn next_batch(&mut self, columns: ClusterColumns) -> Result<Option<RecordBatch>> {
        for batch in self.reader.by_ref() {
            let batch = batch?;
            if batch.num_rows() > 0 {
                self.keys = sort_keys(&batch, columns)?;
                self.row = 0;
                return Ok(Some(batch));
            }
        }
        Ok(None)
    }
}

/// Merge the sorted `runs`, sending batches of `batch_size` rows to `tx`.
///
/// Ties are broken by run, which follows generation order, so the merge is stable.
fn merge_runs_blocking(
    runs: &[PathBuf],
    columns: ClusterColumns,
    batch_size: usize,
    tx: &mpsc::Sender<RecordBatch>,
) -> Result<()> {
    // Batches that rows of `pending` point into, plus the current batch of every live cursor.
    let mut sources: Vec<RecordBatch> = Vec::new();
    let mut cursors: Vec<Option<Cursor>> = Vec::with_capacity(runs.len());
    let mut heap = BinaryHeap::with_capacity(runs.len());
    for (run, path) in runs.iter().enumerate() {
        let mut cursor = Cursor {
            reader: FileReader::try_new(BufReader::new(File::open(path)?), None)?,
            source: sources.len(),
            keys: Vec::new(),
            row: 0,
        };
        if let Some(batch) = cursor.next_batch(columns)? {
            heap.push(Reverse((cursor.keys[0], run)));
            sources.push(batch);
            cursors.push(Some(cursor));
        } else {
            cursors.push(None);
        }
    }

    let mut pending: Vec<(usize, usize)> = Vec::with_capacity(batch_size);
    while let Some(Reverse((_, run))) = heap.pop() {
        let slot = &mut cursors[run];
        let cursor = slot
            .as_mut()
            .ok_or_else(|| anyhow!("merged an exhausted run"))?;
        pending.push((cursor.source, cursor.row));
        cursor.row += 1;
        if cursor.row == cursor.keys.len() {
            match cursor.next_batch(columns)? {
                Some(batch) => {
                    cursor.source = sources.len();
                    sources.push(batch);
                }
                None => *slot = None,
            }
        }
        if let Some(cursor) = slot {
            heap.push(Reverse((cursor.keys[cursor.row], run)));
        }

        if pending.len() == batch_size {
            if !send(&sources, &mut pending, tx)? {
                return Ok(());
            }
            // Keep only the batches that live cursors still read.
            let mut kept = Vec::with_capacity(cursors.len());
            for cursor in cursors.iter_mut().flatten() {
                kept.push(sources[cursor.source].clone());
                cursor.source = kept.len() - 1;
            }
            sources = kept;
        }
    }
    if !pending.is_empty() {
        send(&sources, &mut pending, tx)?;
    }
    Ok(())
}

/// Send the `pending` rows of `sources` as one batch, returning `false` if the receiver is gone.
fn send(
    sources: &[RecordBatch],
    pending: &mut Vec<(usize, usize)>,
    tx: &mpsc::Sender<RecordBatch>,
) -> Result<bool> {
    let sources: Vec<&RecordBatch> = sources.iter().collect();
    let batch = interleave_record_batch(&sources, pending)?;
    pending.clear();
    Ok(tx.blocking_send(batch).is_ok())
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::Arc;

    use arrow_array::Int32Array;
    use arrow_array::UInt32Array;
    use arrow_schema::Field;
    use arrow_schema::Schema;

    use super::*;

    /// Whether the rows of `batch` are in non-decreasing [`SortKey`] order, continuing from
    /// `previous`, which is updated to the last key.
    pub(in crate::tpcds) fn is_sorted_after(
        batch: &RecordBatch,
        columns: ClusterColumns,
        previous: &mut Option<SortKey>,
    ) -> Result<bool> {
        let keys = sort_keys(batch, columns)?;
        let sorted = previous.iter().chain(&keys).is_sorted();
        if let Some(last) = keys.last() {
            *previous = Some(*last);
        }
        Ok(sorted)
    }

    /// Indices that stably sort `batch` by `columns`, ascending with nulls last.
    pub(in crate::tpcds) fn stable_sort(
        batch: &RecordBatch,
        columns: ClusterColumns,
    ) -> Result<UInt32Array> {
        let column = |name: &str| -> Result<Vec<Option<i64>>> {
            let array = cast(batch.column_by_name(name).unwrap(), &DataType::Int64)?;
            Ok(array.as_primitive::<Int64Type>().iter().collect())
        };
        let date = column(columns.date)?;
        let item = column(columns.item)?;
        let mut indices: Vec<u32> = (0..u32::try_from(batch.num_rows())?).collect();
        indices.sort_by_key(|&row| {
            let (date, item) = (date[row as usize], item[row as usize]);
            (date.is_none(), date, item.is_none(), item)
        });
        Ok(UInt32Array::from(indices))
    }

    #[test]
    fn test_sort_keys_nulls_last() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Int32, true),
            Field::new("i", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![None, Some(2), Some(1), Some(1)])),
                Arc::new(Int32Array::from(vec![Some(0), Some(0), None, Some(5)])),
            ],
        )?;
        let columns = ClusterColumns {
            date: "d",
            item: "i",
        };
        let mut keys = sort_keys(&batch, columns)?;
        keys.sort_unstable();
        let order: Vec<_> = keys
            .iter()
            .map(|&(date_null, date, item_null, item)| {
                ((!date_null).then_some(date), (!item_null).then_some(item))
            })
            .collect();
        assert_eq!(
            order,
            [
                (Some(1), Some(5)),
                (Some(1), None),
                (Some(2), Some(0)),
                (None, Some(0)),
            ]
        );
        Ok(())
    }
}
