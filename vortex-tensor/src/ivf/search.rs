// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration of IVF index with Vortex vector arrays and similarity search.
//!
//! Provides functions to build an IVF index from a Vortex [`Vector`](crate::vector::Vector)
//! extension array and to use it for accelerated similarity search by pruning clusters
//! that are unlikely to contain relevant results.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::PValue;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use super::IvfBuildConfig;
use super::IvfIndex;
use crate::utils::cast_to_f32;
use crate::vector::AnyVector;
use crate::vector_search::build_similarity_search_tree;

/// Build an IVF index from a Vortex [`Vector`](crate::vector::Vector) extension array.
///
/// The input `data` must be a `Vector<dim, float>` extension array. The vectors are materialized
/// to f32 for clustering. For TurboQuant-compressed data, pass the *uncompressed* array or
/// decompress first.
///
/// # Errors
///
/// Returns an error if `data` is not a Vector extension array, or if the underlying
/// data cannot be cast to f32.
pub fn build_ivf_index(
    data: &ArrayRef,
    config: &IvfBuildConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<IvfIndex> {
    let ext = data
        .as_opt::<Extension>()
        .ok_or_else(|| vortex_error::vortex_err!("IVF build requires a Vector extension array"))?;

    let vector_meta = ext
        .dtype()
        .as_extension()
        .metadata_opt::<AnyVector>()
        .ok_or_else(|| vortex_error::vortex_err!("IVF build requires a Vector extension type"))?;

    let dim = vector_meta.dimensions() as usize;
    let storage = ext.storage_array();
    let fsl: FixedSizeListArray = storage.clone().execute(ctx)?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    let f32_data = cast_to_f32(elements)?;

    IvfIndex::build(f32_data.as_ref(), dim, config)
}

/// Run a cosine similarity search accelerated by an IVF index.
///
/// This function:
/// 1. Uses the IVF index to identify which clusters to probe
/// 2. Evaluates the full cosine similarity search tree
/// 3. ANDs the result with the IVF probe mask
///
/// Returns a [`BoolArray`] of length `data.len()` where `true` indicates the row's cosine
/// similarity to the query exceeds `threshold` **and** the row belongs to a probed cluster.
///
/// Rows in non-probed clusters are always `false` — they are pruned without being examined.
/// This means recall is not 100% unless `nprobes == index.num_clusters()`.
///
/// # Arguments
///
/// * `data` - The database vectors (Vector extension array, possibly TurboQuant-compressed)
/// * `index` - The IVF index built from the same data (or from the uncompressed version)
/// * `query` - The query vector
/// * `threshold` - Cosine similarity threshold
/// * `nprobes` - Number of clusters to search
pub fn ivf_similarity_search<T: NativePType + Into<PValue>>(
    data: ArrayRef,
    index: &IvfIndex,
    query: &[T],
    threshold: T,
    nprobes: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BoolArray> {
    let num_rows = data.len();

    // Build the IVF probe mask: which rows to actually scan.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64 values outside f32 range become infinity, acceptable for IVF centroid matching"
    )]
    let query_f32: Vec<f32> = query
        .iter()
        .map(|v| {
            let pval: PValue = (*v).into();
            match pval {
                PValue::F32(fv) => fv,
                PValue::F64(fv) => fv as f32,
                PValue::F16(fv) => f32::from(fv),
                _ => 0.0,
            }
        })
        .collect();

    let probe_clusters = index.probe(&query_f32, nprobes)?;
    let probe_mask = index.build_probe_mask(&probe_clusters);

    // Build and execute the full similarity search tree.
    let search_tree = build_similarity_search_tree(data, query, threshold)?;
    let full_result: BoolArray = search_tree.execute(ctx)?;
    let result_bits = full_result.to_bit_buffer();

    // Intersect with probe mask: result[i] = full_result[i] AND probe_mask[i]
    let final_bits =
        BitBufferMut::collect_bool(num_rows, |i| probe_mask[i] && result_bits.value(i));

    Ok(BoolArray::from(final_bits.freeze()))
}
