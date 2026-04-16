// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sample one query vector from `test.parquet`.
//!
//! The vector datasets ship a `test.parquet` alongside the train split: these are the query vectors
//! meant to be issued against the index.
//!
//! The benchmark picks a single random row (seeded for reproducibility) and uses it as the query
//! for the scan.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex_bench::conversions::parquet_to_vortex_chunks;

use crate::SESSION;

/// One query vector sampled from `test.parquet`.
#[derive(Debug, Clone)]
pub struct QuerySample {
    /// The ID of the vector.
    pub id: i64,
    /// f32 query values, length `dim`.
    pub query: Vec<f32>,
}

/// Sample one query row from `test.parquet`.
///
/// The cast to f32 happens here when the source is f64 (matching the prepare-side cast), so that
/// all downstream code is uniformly f32.
pub async fn get_random_query_vector(
    test_parquet: &Path,
    expected_dim: u32,
    src_ptype: PType,
    seed: u64,
) -> Result<QuerySample> {
    let mut ctx = SESSION.create_execution_ctx();

    let chunked = parquet_to_vortex_chunks(test_parquet.to_path_buf())
        .await
        .with_context(|| format!("read test parquet {}", test_parquet.display()))?;
    // The `test.parquet` files are generally small enough that this is not a big deal.
    let struct_array: StructArray = chunked.into_array().execute(&mut ctx)?;

    let id = struct_array
        .unmasked_field_by_name("id")
        .context("test parquet missing `id` column")?
        .clone();
    let emb = struct_array
        .unmasked_field_by_name("emb")
        .context("test parquet missing `emb` column")?
        .clone();

    let mut rng = StdRng::seed_from_u64(seed);
    let query_row_idx = rng.random_range(0..id.len());

    let id_scalar = id.execute_scalar(query_row_idx, &mut ctx)?;
    let emb_scalar = emb.execute_scalar(query_row_idx, &mut ctx)?;

    ensure!(emb_scalar.as_list().len() == expected_dim as usize);

    let id = id_scalar
        .as_primitive()
        .as_::<i64>()
        .ok_or_else(|| vortex_err!("vector ID was not a i64"))?;

    let query_vector = match src_ptype {
        PType::F32 => emb_scalar
            .as_list()
            .elements()
            .vortex_expect("somehow had a null test vector")
            .iter()
            .map(|element| {
                element
                    .as_primitive()
                    .as_::<f32>()
                    .vortex_expect("value was not a f32")
            })
            .collect(),
        PType::F64 =>
        {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "this is intentionally lossy"
            )]
            emb_scalar
                .as_list()
                .elements()
                .vortex_expect("somehow had a null test vector")
                .iter()
                .map(|element| {
                    element
                        .as_primitive()
                        .as_::<f64>()
                        .vortex_expect("value was not a f64") as f32
                })
                .collect()
        }
        ptype => bail!("source ptype {ptype} was somehow not f32 or f64"),
    };

    Ok(QuerySample {
        query: query_vector,
        id,
    })
}
