// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end recall smoke test on a tiny synthetic dataset.
//!
//! We construct a 16-row × 8-dim dataset where row `i` is the i-th standard basis vector
//! (so `cos(row_i, row_i) = 1` and `cos(row_i, row_j) = 0` for `i != j`). Ground-truth
//! top-1 for query row `i` is the train row `i`. After writing the train set as an
//! uncompressed Vortex file, we should see recall@1 = 1.0 — proving the recall machinery
//! agrees with the cosine semantics the scan path uses.
//!
//! We also write an `id` column on the train set since [`measure_recall`] uses an
//! ord-by-scan-order id when comparing against ground truth — that is verified
//! independently by the unit tests in `recall.rs`. This integration test is the assembly
//! check.

#![cfg(test)]

use std::fs::File;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_array::builder::FixedSizeListBuilder;
use arrow_array::builder::Float32Builder;
use arrow_array::builder::Int64Builder;
use arrow_array::builder::ListBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;
use vector_search_bench::compression::VortexCompression;
use vector_search_bench::prepare::CompressionResult;
use vector_search_bench::recall::RecallConfig;
use vector_search_bench::recall::measure_recall;
use vector_search_bench::session::SESSION;
use vortex::array::IntoArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::extension::EmptyMetadata;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::validity::Validity;
use vortex::buffer::BufferMut;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDType;
use vortex_tensor::vector::Vector;

// One row per basis vector — no two rows are equal, so top-1 cosine is unambiguous and the
// id we compare against `neighbors.parquet` is uniquely determined.
const DIM: u32 = 8;
const N_ROWS: u32 = 8;

fn write_synthetic_neighbors_parquet(tmp: &TempDir) -> Result<std::path::PathBuf> {
    // Each query i's top-1 ground truth is row i.
    let path = tmp.path().join("neighbors.parquet");
    let schema = Arc::new(Schema::new(vec![Field::new(
        "neighbors_id",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int64, true)), 1),
        false,
    )]));
    let mut builder = FixedSizeListBuilder::new(Int64Builder::new(), 1);
    for i in 0..N_ROWS {
        builder.values().append_value(i64::from(i));
        builder.append(true);
    }
    let array = builder.finish();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)])?;
    let mut writer = ArrowWriter::try_new(File::create(&path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(path)
}

fn write_synthetic_test_parquet(tmp: &TempDir) -> Result<std::path::PathBuf> {
    // The "test" parquet has the same standard basis vectors as the train set, so
    // sampling row i gives a query identical to train row i. Uses a List<f32> column
    // because that's what real VectorDBBench parquet files emit.
    let path = tmp.path().join("test.parquet");
    let schema = Arc::new(Schema::new(vec![Field::new(
        "emb",
        DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
        false,
    )]));
    let mut builder = ListBuilder::new(Float32Builder::new());
    for i in 0..N_ROWS {
        for d in 0..DIM {
            builder
                .values()
                .append_value(if d == i % DIM { 1.0 } else { 0.0 });
        }
        builder.append(true);
    }
    let array = builder.finish();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)])?;
    let mut writer = ArrowWriter::try_new(File::create(&path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(path)
}

async fn write_synthetic_train_vortex(
    tmp: &TempDir,
    flavor: VortexCompression,
) -> Result<std::path::PathBuf> {
    let path = tmp.path().join("train.vortex");
    let mut buf = BufferMut::<f32>::with_capacity((DIM * N_ROWS) as usize);
    for i in 0..N_ROWS {
        for d in 0..DIM {
            buf.push(if d == i % DIM { 1.0 } else { 0.0 });
        }
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable).into_array();
    let fsl = FixedSizeListArray::try_new(elements, DIM, Validity::NonNullable, N_ROWS as usize)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    let emb = ExtensionArray::new(ext_dtype, fsl.into_array()).into_array();
    let st = StructArray::from_fields(&[("emb", emb)])?.into_array();
    let dtype = st.dtype().clone();
    let stream = ArrayStreamExt::boxed(ArrayStreamAdapter::new(
        dtype,
        futures::stream::iter(std::iter::once(Ok(st))),
    ));
    let mut file = tokio::fs::File::create(&path).await?;
    flavor
        .write_options(&SESSION)
        .write(&mut file, stream)
        .await?;
    file.flush().await?;
    Ok(path)
}

#[tokio::test]
async fn uncompressed_recall_at_one_is_perfect() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let train = write_synthetic_train_vortex(&tmp, VortexCompression::Uncompressed).await?;
    let test = write_synthetic_test_parquet(&tmp)?;
    let neighbors = write_synthetic_neighbors_parquet(&tmp)?;

    let prep = CompressionResult {
        flavor: VortexCompression::Uncompressed,
        vortex_files: vec![train],
        total_wall_time: std::time::Duration::ZERO,
        total_input_bytes: 0,
        total_output_bytes: 0,
    };

    let config = RecallConfig {
        k: 1,
        num_queries: 8,
        query_seed: 7,
    };
    let recall = measure_recall(&prep, &test, &neighbors, PType::F32, &config).await?;
    assert!(
        (recall.mean_recall - 1.0).abs() < 1e-9,
        "mean recall should be 1.0 for lossless flavor on identity data, got {}",
        recall.mean_recall
    );
    Ok(())
}
