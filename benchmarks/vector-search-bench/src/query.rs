// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sample one query vector from `test.parquet`.
//!
//! VectorDBBench corpora ship a `test.parquet` alongside the train split — these are the
//! query vectors meant to be issued against the index. The benchmark picks a single random
//! row (seeded for reproducibility) and uses it as the query for the scan.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex_bench::conversions::list_to_vector_ext;
use vortex_bench::conversions::parquet_to_vortex_chunks;

use crate::session::SESSION;

/// One query vector sampled from `test.parquet`.
#[derive(Debug, Clone)]
pub struct QuerySample {
    /// f32 query values, length `dim`.
    pub query: Vec<f32>,
    /// Row index in the test parquet, kept for log lines / reproducibility.
    pub query_row_idx: u64,
    /// Vector dimension.
    pub dim: u32,
}

/// Sample one query row from `test.parquet`.
///
/// The cast to f32 happens here when the source is f64 (matching the prepare-side cast),
/// so all downstream code is uniformly f32.
pub async fn sample_query(
    test_parquet: &Path,
    expected_dim: u32,
    src_ptype: PType,
    seed: u64,
) -> Result<QuerySample> {
    let chunked = parquet_to_vortex_chunks(test_parquet.to_path_buf())
        .await
        .with_context(|| format!("read test parquet {}", test_parquet.display()))?;
    let mut ctx = SESSION.create_execution_ctx();
    let arr: ArrayRef = chunked.into_array();
    let struct_view = arr
        .as_opt::<Struct>()
        .context("test parquet root is not a struct")?;
    let emb = struct_view
        .unmasked_field_by_name("emb")
        .context("test parquet missing `emb` column")?
        .clone();
    let emb_ext: ExtensionArray = list_to_vector_ext(emb)?.execute(&mut ctx)?;
    let fsl: FixedSizeListArray = emb_ext.storage_array().clone().execute(&mut ctx)?;
    let dim = match fsl.dtype() {
        DType::FixedSizeList(_, dim, _) => *dim,
        other => bail!("test parquet emb dtype is not FSL: {other}"),
    };
    if dim != expected_dim {
        bail!("test parquet emb dim {dim} disagrees with catalog dim {expected_dim}",);
    }
    let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx)?;
    let num_rows = u64::try_from(fsl.len()).unwrap_or(u64::MAX);
    if num_rows == 0 {
        bail!("test parquet has zero rows");
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let query_row_idx = rng.random_range(0..num_rows);
    let row_idx = usize::try_from(query_row_idx)
        .with_context(|| format!("query row index {query_row_idx} does not fit in usize"))?;
    let dim_usize =
        usize::try_from(dim).with_context(|| format!("dim {dim} does not fit in usize"))?;
    let query = extract_row(&elements, src_ptype, row_idx, dim_usize)?;

    Ok(QuerySample {
        query,
        query_row_idx,
        dim,
    })
}

fn extract_row(
    elements: &PrimitiveArray,
    src_ptype: PType,
    row_idx: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    let start = row_idx * dim;
    let end = start + dim;
    match src_ptype {
        PType::F32 => {
            anyhow::ensure!(elements.ptype() == PType::F32, "expected f32 elements");
            Ok(elements.as_slice::<f32>()[start..end].to_vec())
        }
        PType::F64 => {
            anyhow::ensure!(elements.ptype() == PType::F64, "expected f64 elements");
            #[expect(clippy::cast_possible_truncation)]
            Ok(elements.as_slice::<f64>()[start..end]
                .iter()
                .map(|&v| v as f32)
                .collect())
        }
        other => bail!("unsupported test parquet ptype {other}"),
    }
}
