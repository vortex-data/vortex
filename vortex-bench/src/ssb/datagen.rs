// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Star Schema Benchmark data generation.
//!
//! Writes the five SSB tables straight to Parquet from the native generator in
//! the crate-private `ssbgen` module, whose output is byte-identical to the reference C `dbgen`.
//! Nothing is shelled out and nothing is built from source: no `cmake`, no C compiler, no
//! `duckdb` CLI, and no `.tbl` intermediates (SF 10 alone is ~6.5 GB of text).
//!
//! Converting the Parquet into the Vortex formats is a separate step, handled by the shared
//! conversion pipeline like every other benchmark's base data.

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tracing::info;

use crate::Format;
use crate::ssb::ssbgen::arrow::SsbTable;
use crate::ssb::ssbgen::validate_scale_factor;

/// Rows per `RecordBatch` handed to the Parquet writer.
const BATCH_SIZE: NonZeroUsize = NonZeroUsize::new(8192 * 64).unwrap();

/// Generate the SSB Parquet base data for `scale_factor` under `base_dir/parquet/`.
///
/// Idempotent: returns immediately once every table's Parquet is in place.
pub fn generate_tables(scale_factor: &str, base_dir: &Path) -> anyhow::Result<()> {
    // Validate before creating anything: an unsupported scale factor otherwise writes a directory
    // of plausible-looking but relationally invalid Parquet and exits successfully.
    let scale_factor: f64 = scale_factor.parse()?;
    validate_scale_factor(scale_factor)?;

    let parquet_dir = base_dir.join(Format::Parquet.name());
    fs::create_dir_all(&parquet_dir)?;

    let path_of = |table: &SsbTable| parquet_dir.join(format!("{}.parquet", table.name()));

    if SsbTable::ALL.iter().all(|t| path_of(t).exists()) {
        info!(
            "ssb: {} Parquet shards already present in {}",
            SsbTable::ALL.len(),
            parquet_dir.display(),
        );
        return Ok(());
    }

    for table in SsbTable::ALL {
        let path = path_of(&table);
        if path.exists() {
            continue;
        }

        // Write beside the target and rename, so an interrupted run cannot leave a truncated file
        // that the existence check above would then accept. The temporary name carries the process
        // id: two generators racing on one directory would otherwise share the same inode, and one
        // could publish a file the other was still writing.
        let partial = path.with_extension(format!("parquet.{}.partial", std::process::id()));
        info!(
            scale_factor,
            table = table.name(),
            "ssb: generating Parquet"
        );

        let properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(
            fs::File::create(&partial)?,
            Arc::clone(&table.schema()),
            Some(properties),
        )?;
        for batch in table.batches(scale_factor, BATCH_SIZE) {
            writer.write(&batch)?;
        }
        writer.close()?;
        fs::rename(&partial, &path)?;
    }

    info!(
        "ssb base data generated in {} ({} Parquet shards)",
        parquet_dir.display(),
        SsbTable::ALL.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::datasets::SSB_TABLES;
    use crate::ssb::ssbgen::arrow::SsbTable;

    /// `datasets::SSB_TABLES` drives table registration while [`SsbTable`] drives generation; a
    /// divergence would silently register a table nothing writes (or vice versa).
    #[test]
    fn table_lists_agree() {
        let mut generated = SsbTable::ALL.iter().map(|t| t.name()).collect::<Vec<_>>();
        let mut registered = SSB_TABLES.to_vec();
        generated.sort_unstable();
        registered.sort_unstable();
        assert_eq!(generated, registered);
    }
}
