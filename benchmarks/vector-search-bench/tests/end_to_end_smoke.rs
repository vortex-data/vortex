// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end smoke test: write a synthetic dataset to a `.vortex` file, then scan it with
//! the cosine-similarity filter expression and assert the right rows survive.
//!
//! This is the first proof that the prepare-side write strategy and the expression-side
//! filter agree: the row that exactly matches the query vector (cosine = 1.0) must clear
//! any reasonable threshold.

#![cfg(test)]

use anyhow::Result;
use futures::TryStreamExt;
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;
use vector_search_bench::compression::VortexCompression;
use vector_search_bench::expression::similarity_filter;
use vector_search_bench::session::SESSION;
use vortex::array::ArrayRef;
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
use vortex::dtype::extension::ExtDType;
use vortex::file::OpenOptionsSessionExt;
use vortex_tensor::vector::Vector;

const DIM: u32 = 4;
const ROWS: u32 = 3;

/// `Struct { emb: Vector<f32, 4> }` with row 0 = [1, 0, 0, 0], rows 1+2 orthogonal.
fn synthetic_struct() -> Result<ArrayRef> {
    let rows: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let mut buf = BufferMut::<f32>::with_capacity((DIM * ROWS) as usize);
    for row in rows.iter() {
        for &v in row.iter() {
            buf.push(v);
        }
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable).into_array();
    let fsl = FixedSizeListArray::try_new(elements, DIM, Validity::NonNullable, ROWS as usize)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    let emb = ExtensionArray::new(ext_dtype, fsl.into_array()).into_array();
    Ok(StructArray::from_fields(&[("emb", emb)])?.into_array())
}

async fn write_vortex(struct_array: ArrayRef, flavor: VortexCompression) -> Result<TempDir> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("synthetic.vortex");
    let mut file = tokio::fs::File::create(&path).await?;
    let dtype = struct_array.dtype().clone();
    let stream = ArrayStreamExt::boxed(ArrayStreamAdapter::new(
        dtype,
        futures::stream::iter(std::iter::once(Ok(struct_array))),
    ));
    flavor
        .write_options(&SESSION)
        .write(&mut file, stream)
        .await?;
    file.flush().await?;
    Ok(tmp)
}

#[tokio::test]
async fn uncompressed_scan_returns_self_match() -> Result<()> {
    let tmp = write_vortex(synthetic_struct()?, VortexCompression::Uncompressed).await?;
    let path = tmp.path().join("synthetic.vortex");

    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let filter = similarity_filter(&query, 0.5)?;

    let file = SESSION.open_options().open_path(&path).await?;
    let chunks: Vec<ArrayRef> = file
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .try_collect()
        .await?;

    let total: usize = chunks.iter().map(|c| c.len()).sum();
    assert_eq!(total, 1, "expected exactly one self-match row, got {total}");
    Ok(())
}

#[tokio::test]
async fn turboquant_scan_returns_self_match() -> Result<()> {
    // Make this slightly larger so TurboQuant has enough rows to fit a dictionary; 64 rows
    // of 16 dimensions is well above the encoding's minimums.
    const TQ_DIM: u32 = 16;
    const TQ_ROWS: u32 = 64;
    let mut buf = BufferMut::<f32>::with_capacity((TQ_DIM * TQ_ROWS) as usize);
    for r in 0..TQ_ROWS {
        for d in 0..TQ_DIM {
            // Distinct unit-norm vectors: each row has a single 1.0 in a deterministic slot.
            buf.push(if d == r % TQ_DIM { 1.0 } else { 0.0 });
        }
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable).into_array();
    let fsl =
        FixedSizeListArray::try_new(elements, TQ_DIM, Validity::NonNullable, TQ_ROWS as usize)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    let emb = ExtensionArray::new(ext_dtype, fsl.into_array()).into_array();
    let chunk = StructArray::from_fields(&[("emb", emb)])?.into_array();

    let tmp = write_vortex(chunk, VortexCompression::TurboQuant).await?;
    let path = tmp.path().join("synthetic.vortex");

    let mut query = vec![0.0f32; TQ_DIM as usize];
    query[0] = 1.0;
    let filter = similarity_filter(&query, 0.5)?;

    let file = SESSION.open_options().open_path(&path).await?;
    let chunks: Vec<ArrayRef> = file
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .try_collect()
        .await?;

    let total: usize = chunks.iter().map(|c| c.len()).sum();
    // TurboQuant is lossy; we just want the self-matching rows (at least row 0) to survive
    // the threshold. Anything between 1 and TQ_ROWS is plausible — the important property
    // is that the round-trip through the file did not break the filter pipeline entirely.
    assert!(
        total >= 1,
        "expected at least the self-match row to survive turboquant scan, got {total}"
    );
    Ok(())
}
