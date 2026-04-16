// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant-aware IVF index construction.
//!
//! When a column is TurboQuant-compressed, each vector is stored as a sequence of small integer
//! codes that index into a shared 1-D centroid codebook. All codes live in the SORF-rotated
//! unit-norm space. We can build the IVF cluster centroids directly in that rotated space and
//! rotate the query vector once at read time instead of inverting SORF for every database vector.
//!
//! # API
//!
//! - [`build_ivf_from_turboquant`] reads the TurboQuant-compressed array, materialises each row's
//!   rotated f32 coordinates from the dict codes, and runs k-means on that.
//! - [`rotate_query`] applies the forward SORF transform to a query vector so it is comparable
//!   against the rotated cluster centroids.
//!
//! The caller composes these as: `rotate_query` → `TurboQuantIvfIndex::probe` → execute the
//! cosine-similarity search. Because the stored data remains TQ-compressed, the existing
//! cosine-similarity fast paths in `vortex-tensor` (dict+constant direct lookup, SORF
//! pull-through) continue to apply inside each cluster chunk — the IVF layer only decides which
//! chunks to open.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::scalar_fns::sorf_transform::SorfMatrix;
use vortex_tensor::scalar_fns::sorf_transform::SorfOptions;
use vortex_tensor::scalar_fns::sorf_transform::SorfTransform;

use crate::IvfBuildConfig;
use crate::IvfIndex;

/// An IVF index built on TurboQuant-compressed data.
///
/// Stores the recovered [`SorfOptions`] so callers can rotate queries into the index's space.
#[derive(Clone, Debug)]
pub struct TurboQuantIvfIndex {
    index: IvfIndex,
    sorf: SorfOptions,
}

impl TurboQuantIvfIndex {
    /// Returns the underlying [`IvfIndex`]. Its centroids live in the SORF-rotated space.
    pub fn index(&self) -> &IvfIndex {
        &self.index
    }

    /// Returns the [`SorfOptions`] needed to rotate a query into the same space.
    pub fn sorf_options(&self) -> &SorfOptions {
        &self.sorf
    }

    /// Rotate a query vector into the space where the centroids live, then probe.
    ///
    /// The returned `Vec<usize>` is the list of cluster indices to search, sorted by descending
    /// similarity.
    pub fn probe(&self, query: &[f32], nprobes: usize) -> VortexResult<Vec<usize>> {
        let rotated = rotate_query(query, &self.sorf)?;
        self.index.probe(&rotated, nprobes)
    }
}

/// Build an IVF index directly from a TurboQuant-compressed Vector extension array.
///
/// The expected shape is the canonical TurboQuant pipeline output:
///
/// ```text
/// ScalarFnArray(L2Denorm, [
///     ScalarFnArray(SorfTransform, [FSL(Dict(codes, centroids))]),
///     norms
/// ])
/// ```
///
/// This function:
/// 1. Descends to the `Dict(codes, centroids)` FSL stored under the SorfTransform wrapper.
/// 2. Materialises each row's **rotated** f32 coordinates by looking up `centroids[codes[i][j]]`
///    for each of the `padded_dim` coordinates. This is the same dict-lookup the inner-product
///    fast path uses.
/// 3. Runs k-means on those rotated coordinates to find cluster centroids in the rotated space.
///
/// The resulting [`TurboQuantIvfIndex`] wraps an [`IvfIndex`] whose centroids live in the
/// SORF-rotated space plus the [`SorfOptions`] required to rotate queries.
///
/// Compared to [`crate::search::build_ivf_index`], this variant:
/// - avoids inverting SORF (which is log-linear per row), and
/// - avoids materialising per-row f32 buffers in the *original* space; we pay a small constant
///   factor to rotate the query once, instead of N rotations for the database.
///
/// # Errors
///
/// Returns an error if the input is not a canonical TurboQuant pipeline output.
pub fn build_ivf_from_turboquant(
    data: &ArrayRef,
    config: &IvfBuildConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TurboQuantIvfIndex> {
    // Unwrap L2Denorm(ScalarFnArray, [SorfTransform(Vector(FSL(Dict))), norms])
    let l2 = data.as_opt::<ExactScalarFn<L2Denorm>>().ok_or_else(|| {
        vortex_err!(
            "TQ IVF build expected L2Denorm wrapper, got {}",
            data.encoding_id()
        )
    })?;

    let sorf_child = l2
        .nth_child(0)
        .ok_or_else(|| vortex_err!("L2Denorm is missing its normalized child"))?;

    let sorf = sorf_child
        .as_opt::<ExactScalarFn<SorfTransform>>()
        .ok_or_else(|| {
            vortex_err!(
                "TQ IVF build expected SorfTransform wrapper, got {}",
                sorf_child.encoding_id()
            )
        })?;

    // Record the SORF options so we can build a matching rotation for queries.
    let sorf_options: SorfOptions = sorf.options.clone();

    let vec_child = sorf
        .nth_child(0)
        .ok_or_else(|| vortex_err!("SorfTransform is missing its input child"))?;

    let ext = vec_child
        .as_opt::<Extension>()
        .ok_or_else(|| vortex_err!("SorfTransform input must be a Vector extension array"))?;

    let fsl_dyn = ext.storage_array();
    let fsl = fsl_dyn
        .as_opt::<FixedSizeList>()
        .ok_or_else(|| vortex_err!("SorfTransform input storage must be a FixedSizeList"))?;

    // The FSL elements must be a DictArray of (u8 codes, f32 centroids).
    let dict_dyn = fsl.elements();
    let dict = dict_dyn
        .as_opt::<Dict>()
        .ok_or_else(|| vortex_err!("TQ IVF build expected DictArray under the FSL"))?;

    let padded_dim = usize::try_from(fsl.list_size())?;
    let num_vectors = fsl.len();

    // Materialise codes and centroid values.
    let codes_arr: PrimitiveArray = dict.codes().clone().execute(ctx)?;
    let codes = codes_arr.as_slice::<u8>();
    let values_arr: PrimitiveArray = dict.values().clone().execute(ctx)?;
    let values = values_arr.as_slice::<f32>();

    // Reconstruct the rotated-space f32 buffer by dict lookup. This is O(N * padded_dim) with
    // a tight inner loop — the same cost as computing distances against a query.
    let mut flat = vec![0.0f32; num_vectors * padded_dim];
    for row in 0..num_vectors {
        let row_codes = &codes[row * padded_dim..(row + 1) * padded_dim];
        let row_out = &mut flat[row * padded_dim..(row + 1) * padded_dim];
        for (dst, &code) in row_out.iter_mut().zip(row_codes.iter()) {
            *dst = values[code as usize];
        }
    }

    let ivf = IvfIndex::build(&flat, padded_dim, config)?;
    Ok(TurboQuantIvfIndex {
        index: ivf,
        sorf: sorf_options,
    })
}

/// Apply the forward SORF transform to a query vector so it matches the rotated space of
/// TQ-compressed centroids.
///
/// The input `query` has length `options.dimension`; the output has length
/// `padded_dim = next_power_of_two(dimension)`. Values beyond the original dimension are
/// zero-padded before rotation.
pub fn rotate_query(query: &[f32], options: &SorfOptions) -> VortexResult<Vec<f32>> {
    let dim = usize::try_from(options.dimension)?;
    if query.len() != dim {
        return Err(vortex_err!(
            "query length {} does not match SORF dimension {}",
            query.len(),
            dim
        ));
    }

    let matrix = SorfMatrix::try_new(options.seed, dim, options.num_rounds as usize)?;
    let padded_dim = matrix.padded_dim();
    let mut padded = vec![0.0f32; padded_dim];
    padded[..dim].copy_from_slice(query);
    let mut out = vec![0.0f32; padded_dim];
    matrix.rotate(&padded, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::EmptyMetadata;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_tensor::vector::Vector;
    use vortex_tensor::vector_search::compress_turboquant;

    use super::*;
    use crate::IvfBuildConfig;

    fn session() -> VortexSession {
        let session = VortexSession::empty().with::<ArraySession>();
        vortex_tensor::initialize(&session);
        session
    }

    fn vector_array_f32(dim: u32, values: &[f32]) -> VortexResult<ArrayRef> {
        let row_count = values.len() / dim as usize;
        let mut buf = BufferMut::<f32>::with_capacity(values.len());
        for &v in values {
            buf.push(v);
        }
        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        let fsl = FixedSizeListArray::try_new(
            elements.into_array(),
            dim,
            Validity::NonNullable,
            row_count,
        )?;
        let ext = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
        Ok(ExtensionArray::new(ext, fsl.into_array()).into_array())
    }

    #[test]
    fn build_from_tq_and_query_finds_self() -> VortexResult<()> {
        const DIM: u32 = 128;
        const NUM_ROWS: usize = 64;

        let mut values = Vec::<f32>::with_capacity(NUM_ROWS * DIM as usize);
        for row in 0..NUM_ROWS {
            for i in 0..DIM as usize {
                values.push(((row as f32 * 0.7 + i as f32) * 0.03).sin());
            }
        }
        let data = vector_array_f32(DIM, &values)?;
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let compressed = compress_turboquant(data, &mut ctx)?;

        let config = IvfBuildConfig {
            num_clusters: 4,
            max_iterations: 20,
            seed: 42,
        };
        let index = build_ivf_from_turboquant(&compressed, &config, &mut ctx)?;

        // Rotated centroids should have dim = padded_dim.
        assert_eq!(index.index().dim(), (DIM as usize).next_power_of_two());
        assert_eq!(index.index().num_clusters(), 4);
        assert_eq!(index.index().num_vectors(), NUM_ROWS);

        // Use row 0 as query. Probe nprobes=1 — must return row 0's cluster.
        let query: &[f32] = &values[..DIM as usize];
        let probed = index.probe(query, 1)?;
        let self_cluster = index.index().assignments()[0] as usize;
        assert!(
            probed.contains(&self_cluster),
            "query == row 0 must be in row 0's own cluster"
        );
        Ok(())
    }

    #[test]
    fn rejects_non_tq_input() -> VortexResult<()> {
        // Plain vector array is not TQ-compressed; the function should reject it.
        let data = vector_array_f32(128, &vec![0.0f32; 128])?;
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let config = IvfBuildConfig::default();
        let result = build_ivf_from_turboquant(&data, &config, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }
}
