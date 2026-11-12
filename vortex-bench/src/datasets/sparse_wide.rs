// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wide, sparse tables with mutually exclusive groups and nullable nested fields.

use std::fs::File;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use async_trait::async_trait;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::dtype::Nullability;
use vortex::mask::Mask;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

use crate::IdempotentPath;
use crate::SESSION;
use crate::datasets::Dataset;
use crate::idempotent_async;

/// Synthetic sparse-wide workloads in the compression benchmark's CPU suite.
#[derive(Clone, Copy)]
pub enum SparseWide {
    /// Flat nullable integer fields with clustered group presence.
    FlatRuns,
    /// Integer fields under four nullable struct levels.
    NestedRuns,
    /// Nested list fields with clustered group presence.
    ListsRuns,
    /// Nested list fields with shuffled group presence.
    ListsShuffled,
    /// Nested integers with genuine nulls inside present groups.
    NestedChildNulls,
}

impl SparseWide {
    /// All workloads included in the default CPU compression suite.
    pub const ALL: [Self; 5] = [
        Self::FlatRuns,
        Self::NestedRuns,
        Self::ListsRuns,
        Self::ListsShuffled,
        Self::NestedChildNulls,
    ];

    fn generate(&self, rows: usize, groups: usize, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut group_ids: Vec<usize> = (0..rows).map(|row| row / (rows / groups)).collect();
        if matches!(self, Self::ListsShuffled) {
            group_ids.shuffle(&mut rng);
        }
        // Values use a separate RNG so shuffling changes only row positions, not the group payloads.
        let mut values_rng = StdRng::seed_from_u64(42);
        let mut wide_fields = Vec::new();
        for group in 0..groups {
            let positions: Vec<usize> = group_ids
                .iter()
                .enumerate()
                .filter_map(|(row, &id)| (id == group).then_some(row))
                .collect();
            let present = Mask::from_buffer(BitBuffer::from_iter(
                group_ids.iter().map(|&id| id == group),
            ));
            let group_validity = Validity::from_mask(present, Nullability::Nullable);
            let mut fields = Vec::new();
            for field in 0..3 {
                let mut valid = vec![false; rows];
                let mut values = vec![0i64; rows];
                for (index, &row) in positions.iter().enumerate() {
                    valid[row] =
                        !matches!(self, Self::NestedChildNulls) || (index + field) % 10 != 0;
                    if valid[row] {
                        values[row] = values_rng.random_range(0..1_000_000);
                    }
                }
                let validity = Validity::from(BitBuffer::from_iter(valid.iter().copied()));
                let array = if matches!(self, Self::ListsRuns | Self::ListsShuffled) {
                    let mut offsets = Vec::with_capacity(rows + 1);
                    let mut elements = Vec::with_capacity(positions.len() * 3);
                    offsets.push(0u32);
                    for &is_valid in &valid {
                        if is_valid {
                            elements
                                .extend((0..3).map(|_| values_rng.random_range(0..1_000_000i64)));
                        }
                        offsets.push(u32::try_from(elements.len())?);
                    }
                    ListArray::try_new(
                        PrimitiveArray::new(Buffer::from(elements), Validity::AllValid)
                            .into_array(),
                        PrimitiveArray::new(Buffer::from(offsets), Validity::NonNullable)
                            .into_array(),
                        validity,
                        ctx,
                    )?
                    .into_array()
                } else {
                    PrimitiveArray::new(Buffer::from(values), validity).into_array()
                };
                fields.push((format!("g{group:03}_f{field}"), array));
            }
            if !matches!(self, Self::FlatRuns) {
                for depth in 0..4 {
                    let names = fields.iter().map(|(name, _)| name.as_str()).collect();
                    let child = StructArray::try_new(
                        names,
                        fields.into_iter().map(|(_, array)| array),
                        rows,
                        group_validity.clone(),
                    )?
                    .into_array();
                    let name = if depth == 3 {
                        format!("g{group:03}")
                    } else {
                        format!("level{depth}")
                    };
                    fields = vec![(name, child)];
                }
            }
            wide_fields.extend(fields);
        }
        struct_from_fields(&wide_fields)
    }
}

fn struct_from_fields(fields: &[(String, ArrayRef)]) -> Result<ArrayRef> {
    Ok(StructArray::from_fields(
        &fields
            .iter()
            .map(|(name, array)| (name.as_str(), array.clone()))
            .collect::<Vec<_>>(),
    )?
    .into_array())
}

#[async_trait]
impl Dataset for SparseWide {
    fn name(&self) -> &str {
        match self {
            Self::FlatRuns => "sparse-wide-flat-runs",
            Self::NestedRuns => "sparse-wide-nested-runs",
            Self::ListsRuns => "sparse-wide-lists-runs",
            Self::ListsShuffled => "sparse-wide-lists-shuffled",
            Self::NestedChildNulls => "sparse-wide-nested-child-nulls",
        }
    }

    async fn to_vortex_array(&self, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        self.generate(100_000, 100, ctx)
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        let path = format!("sparse_wide/v1/{}.parquet", self.name()).to_data_path();
        idempotent_async(path, |temp_path| async move {
            let mut ctx = SESSION.create_execution_ctx();
            let array = self.to_vortex_array(&mut ctx).await?;
            write_parquet(&temp_path, array, &mut ctx)
        })
        .await
    }
}

fn write_parquet(path: &Path, array: ArrayRef, ctx: &mut ExecutionCtx) -> Result<()> {
    let schema = SESSION.arrow().to_arrow_schema(array.dtype())?;
    let root = Field::new_struct("", schema.fields, false);
    let arrow = array.execute_arrow(Some(&root), ctx)?;
    let batch = RecordBatch::from(arrow.as_struct());
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use tempfile::NamedTempFile;
    use vortex_array::assert_arrays_eq;

    use super::*;
    use crate::conversions::parquet_to_vortex_chunks;

    #[rstest]
    #[case::flat(SparseWide::FlatRuns)]
    #[case::nested(SparseWide::NestedRuns)]
    #[case::lists(SparseWide::ListsRuns)]
    #[case::shuffled_lists(SparseWide::ListsShuffled)]
    #[case::child_nulls(SparseWide::NestedChildNulls)]
    #[tokio::test]
    async fn parquet_roundtrip(#[case] dataset: SparseWide) -> Result<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let expected = dataset.generate(1_000, 10, &mut ctx)?;
        let file = NamedTempFile::new()?;
        write_parquet(file.path(), expected.clone(), &mut ctx)?;
        let actual = parquet_to_vortex_chunks(file.path().to_path_buf())
            .await?
            .into_array();
        assert_eq!(actual.len(), 1_000);
        assert_eq!(actual.dtype(), expected.dtype());
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }
}
