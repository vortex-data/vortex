// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converts JSONBench's gzipped NDJSON source files to Parquet files with a Variant column.

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Schema;
use flate2::read::MultiGzDecoder;
use parquet::arrow::ArrowWriter;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::shred_variant;
use serde::de::IgnoredAny;
use tracing::warn;

use crate::utils::parquet::parquet_writer_properties;

/// Documents parsed and shredded together. Each batch becomes one Parquet row group.
const ROWS_PER_BATCH: usize = 65_536;

/// Name of the table's only column.
const COLUMN: &str = "data";

/// Converts the gzipped NDJSON file `json_gz` into a Parquet file at `parquet_path` with one
/// Variant column, shredded as `shredding` when given.
///
/// Lines that are not valid JSON are skipped with a warning, as JSONBench's own loaders do.
pub(super) fn convert_json_to_parquet(
    json_gz: &Path,
    parquet_path: &Path,
    shredding: Option<&DataType>,
) -> anyhow::Result<()> {
    let reader = BufReader::new(MultiGzDecoder::new(File::open(json_gz)?));
    let mut writer: Option<ArrowWriter<File>> = None;
    let mut skipped = 0usize;

    let mut lines = Vec::with_capacity(ROWS_PER_BATCH);
    let mut flush = |lines: &mut Vec<String>| -> anyhow::Result<()> {
        let batch = variant_batch(lines, shredding, &mut skipped)?;
        lines.clear();
        let writer = match writer.as_mut() {
            Some(writer) => writer,
            None => {
                let properties = parquet_writer_properties()
                    .set_max_row_group_row_count(Some(ROWS_PER_BATCH))
                    .build();
                writer.insert(ArrowWriter::try_new(
                    File::create(parquet_path)?,
                    batch.schema(),
                    Some(properties),
                )?)
            }
        };
        writer.write(&batch)?;
        Ok(())
    };

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        lines.push(line);
        if lines.len() == ROWS_PER_BATCH {
            flush(&mut lines)?;
        }
    }
    if !lines.is_empty() {
        flush(&mut lines)?;
    }

    if skipped > 0 {
        warn!(
            "Skipped {skipped} invalid JSON documents in {}",
            json_gz.display()
        );
    }
    writer
        .ok_or_else(|| anyhow::anyhow!("{} holds no JSON documents", json_gz.display()))?
        .close()?;
    Ok(())
}

/// Parses `lines` into a single-column batch of Variant values.
fn variant_batch(
    lines: &[String],
    shredding: Option<&DataType>,
    skipped: &mut usize,
) -> anyhow::Result<RecordBatch> {
    let to_strings = |lines: &mut dyn Iterator<Item = &str>| {
        Arc::new(StringArray::from_iter_values(lines)) as ArrayRef
    };

    let strings = to_strings(&mut lines.iter().map(String::as_str));
    let variant = match json_to_variant(&strings) {
        Ok(variant) => variant,
        // One invalid document fails the whole batch, so drop the invalid ones and retry.
        Err(_) => {
            let is_json = |line: &&str| serde_json::from_str::<IgnoredAny>(line).is_ok();
            let valid = to_strings(&mut lines.iter().map(String::as_str).filter(is_json));
            *skipped += lines.len() - valid.len();
            json_to_variant(&valid)?
        }
    };
    let variant = match shredding {
        Some(shredding) => shred_variant(&variant, shredding)?,
        None => variant,
    };

    let schema = Arc::new(Schema::new(vec![variant.field(COLUMN)]));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(variant.into_inner())],
    )?)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use arrow_array::cast::AsArray;
    use arrow_schema::Field;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet_variant::VariantPath;
    use parquet_variant_compute::GetOptions;
    use parquet_variant_compute::variant_get;
    use rstest::rstest;

    use super::*;
    use crate::jsonbench::Shredding;

    #[rstest]
    #[case::unshredded(Shredding::None)]
    #[case::partial(Shredding::Partial)]
    #[case::query_fields(Shredding::QueryFields)]
    fn converts_ndjson_to_variant_parquet(#[case] shredding: Shredding) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let json_gz = dir.path().join("events.json.gz");
        let parquet_path = dir.path().join("events.parquet");

        let mut encoder = GzEncoder::new(File::create(&json_gz)?, Compression::default());
        encoder.write_all(
            concat!(
                r#"{"did":"a","kind":"commit","commit":{"collection":"post","operation":"create"}}"#,
                "\n",
                "not json\n",
                "\n",
                r#"{"did":"b","kind":"identity"}"#,
                "\n",
            )
            .as_bytes(),
        )?;
        encoder.finish()?;

        convert_json_to_parquet(&json_gz, &parquet_path, shredding.schema()?.as_ref())?;

        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&parquet_path)?)?;
        assert_eq!(
            reader.schema().field(0).extension_type_name(),
            Some("arrow.parquet.variant")
        );
        let batches = reader.build()?.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(batches.len(), 1);

        // The invalid and empty lines are dropped, and shredding does not change the values.
        let collection = variant_get(
            batches[0].column(0),
            GetOptions::new_with_path(VariantPath::try_from("commit.collection")?)
                .with_as_type(Some(Arc::new(Field::new("", DataType::Utf8, true)))),
        )?;
        let collection = collection.as_string::<i32>().iter().collect::<Vec<_>>();
        assert_eq!(collection, vec![Some("post"), None]);
        Ok(())
    }
}
