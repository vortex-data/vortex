// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::IvfBuildConfig;
use super::IvfIndex;
use super::search::build_ivf_index;
use super::search::ivf_similarity_search;
use crate::vector::Vector;
use crate::vector_search::compress_turboquant;

/// Generate `n` random-ish f32 vectors of the given dimension using a simple deterministic
/// formula. Vectors are placed into `num_real_clusters` tight clusters for testing purposes.
pub(super) fn generate_clustered_vectors(
    n: usize,
    dim: usize,
    num_real_clusters: usize,
    seed: u64,
) -> Vec<f32> {
    let mut vectors = vec![0.0f32; n * dim];
    let mut state = seed;

    // Generate cluster centers.
    let mut centers = vec![0.0f32; num_real_clusters * dim];
    for val in &mut centers {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *val = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
    }

    // Assign each vector to a cluster and add small noise.
    for i in 0..n {
        let cluster = i % num_real_clusters;
        let center = &centers[cluster * dim..(cluster + 1) * dim];
        let v = &mut vectors[i * dim..(i + 1) * dim];
        for (dst, &c) in v.iter_mut().zip(center.iter()) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.05;
            *dst = c + noise;
        }
    }

    vectors
}

/// Normalize all vectors in-place to unit length.
pub(super) fn normalize_vectors(vectors: &mut [f32], dim: usize) {
    let n = vectors.len() / dim;
    for i in 0..n {
        let v = &mut vectors[i * dim..(i + 1) * dim];
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}

#[test]
fn build_empty_index() -> VortexResult<()> {
    let config = IvfBuildConfig {
        num_clusters: 4,
        ..Default::default()
    };
    let index = IvfIndex::build(&[], 8, &config)?;
    assert_eq!(index.num_vectors(), 0);
    assert_eq!(index.num_clusters(), 4);
    assert_eq!(index.dim(), 8);
    Ok(())
}

#[test]
fn build_single_vector() -> VortexResult<()> {
    let vectors = vec![1.0f32, 0.0, 0.0, 0.0];
    let config = IvfBuildConfig {
        num_clusters: 2,
        ..Default::default()
    };
    // With 1 vector, k is clamped to 1.
    let index = IvfIndex::build(&vectors, 4, &config)?;
    assert_eq!(index.num_vectors(), 1);
    assert_eq!(index.num_clusters(), 1);
    assert_eq!(index.assignments(), &[0]);
    Ok(())
}

#[test]
fn build_and_probe_small() -> VortexResult<()> {
    // 8 vectors in 3D, arranged in 2 obvious clusters.
    #[rustfmt::skip]
    let vectors: Vec<f32> = vec![
        // Cluster near [1, 0, 0]
        0.95, 0.05, 0.0,
        1.00, 0.00, 0.0,
        0.90, 0.10, 0.0,
        0.98, 0.02, 0.0,
        // Cluster near [0, 1, 0]
        0.0, 0.95, 0.05,
        0.0, 1.00, 0.00,
        0.0, 0.90, 0.10,
        0.0, 0.98, 0.02,
    ];
    let config = IvfBuildConfig {
        num_clusters: 2,
        max_iterations: 50,
        seed: 42,
    };
    let index = IvfIndex::build(&vectors, 3, &config)?;

    // Verify: all vectors should be assigned to one of 2 clusters.
    assert_eq!(index.num_clusters(), 2);
    assert_eq!(index.num_vectors(), 8);

    // The first 4 vectors should have the same assignment and the last 4 should have a different one.
    let a = index.assignments();
    assert_eq!(a[0], a[1]);
    assert_eq!(a[0], a[2]);
    assert_eq!(a[0], a[3]);
    assert_eq!(a[4], a[5]);
    assert_eq!(a[4], a[6]);
    assert_eq!(a[4], a[7]);
    assert_ne!(a[0], a[4]);

    // Probe with a query near [1, 0, 0] should return the first cluster.
    let query = vec![0.99f32, 0.01, 0.0];
    let probed = index.probe(&query, 1)?;
    assert_eq!(probed.len(), 1);

    // The probed cluster should contain the first 4 vectors.
    let mask = index.build_probe_mask(&probed);
    assert_eq!(mask.len(), 8);
    let first_cluster_val = a[0] as usize;
    let probed_cluster = probed[0];
    assert_eq!(probed_cluster, first_cluster_val);

    // All first-cluster vectors should be in the mask.
    for i in 0..4 {
        assert!(mask[i], "vector {i} should be in probe mask");
    }
    for i in 4..8 {
        assert!(!mask[i], "vector {i} should not be in probe mask");
    }

    Ok(())
}

#[test]
fn cluster_sizes_are_consistent() -> VortexResult<()> {
    let vectors = generate_clustered_vectors(100, 16, 4, 123);
    let config = IvfBuildConfig {
        num_clusters: 4,
        max_iterations: 20,
        seed: 42,
    };
    let index = IvfIndex::build(&vectors, 16, &config)?;

    let sizes = index.cluster_sizes();
    assert_eq!(sizes.len(), 4);
    assert_eq!(sizes.iter().sum::<usize>(), 100);

    // Every size should be > 0 since we have well-separated clusters.
    for (i, &s) in sizes.iter().enumerate() {
        assert!(s > 0, "cluster {i} should have at least one vector");
    }

    Ok(())
}

#[rstest]
#[case(100, 8, 4, 2)]
#[case(500, 32, 8, 4)]
#[case(1000, 64, 16, 4)]
fn probe_recall_is_good(
    #[case] n: usize,
    #[case] dim: usize,
    #[case] k: u32,
    #[case] nprobes: usize,
) -> VortexResult<()> {
    // Generate clustered data where we know the ground truth.
    let mut vectors = generate_clustered_vectors(n, dim, k as usize, 42);
    normalize_vectors(&mut vectors, dim);

    let config = IvfBuildConfig {
        num_clusters: k,
        max_iterations: 30,
        seed: 42,
    };
    let index = IvfIndex::build(&vectors, dim, &config)?;

    // Use each vector's actual cluster centroid as the query. The correct cluster
    // should always be in the probe set.
    let mut correct_probes = 0;
    let total_queries = n.min(50); // Test a subset to keep it fast.
    for qi in 0..total_queries {
        let query = &vectors[qi * dim..(qi + 1) * dim];
        let probed = index.probe(query, nprobes)?;
        let assignment = index.assignments()[qi] as usize;
        if probed.contains(&assignment) {
            correct_probes += 1;
        }
    }

    let recall = correct_probes as f64 / total_queries as f64;
    assert!(
        recall > 0.8,
        "recall {recall:.2} is too low (expected > 0.8) for n={n} dim={dim} k={k} nprobes={nprobes}"
    );
    Ok(())
}

#[test]
fn query_mask_convenience() -> VortexResult<()> {
    let mut vectors = generate_clustered_vectors(200, 16, 4, 42);
    normalize_vectors(&mut vectors, 16);

    let config = IvfBuildConfig {
        num_clusters: 4,
        max_iterations: 20,
        seed: 42,
    };
    let index = IvfIndex::build(&vectors, 16, &config)?;

    let query = &vectors[0..16];
    let mask = index.query_mask(query, 2)?;
    assert_eq!(mask.len(), 200);

    // At least some vectors should be selected, and some should be pruned.
    let selected = mask.iter().filter(|&&v| v).count();
    assert!(selected > 0, "no vectors selected");
    assert!(selected < 200, "all vectors selected (no pruning happened)");

    Ok(())
}

#[test]
fn probe_all_clusters() -> VortexResult<()> {
    let vectors = generate_clustered_vectors(100, 8, 4, 42);
    let config = IvfBuildConfig {
        num_clusters: 4,
        max_iterations: 20,
        seed: 42,
    };
    let index = IvfIndex::build(&vectors, 8, &config)?;

    // Probing all clusters should select all rows.
    let query = &vectors[0..8];
    let probed = index.probe(query, 100)?; // nprobes > k
    assert_eq!(probed.len(), 4);

    let mask = index.build_probe_mask(&probed);
    assert!(mask.iter().all(|&v| v));

    Ok(())
}

#[test]
fn rejects_mismatched_query_dim() {
    let vectors = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let config = IvfBuildConfig {
        num_clusters: 1,
        ..Default::default()
    };
    let index = IvfIndex::build(&vectors, 3, &config).unwrap();

    let wrong_dim_query = vec![1.0f32, 0.0]; // dim=2, should be 3
    assert!(index.probe(&wrong_dim_query, 1).is_err());
}

#[test]
fn rejects_zero_dim() {
    let config = IvfBuildConfig::default();
    assert!(IvfIndex::build(&[], 0, &config).is_err());
}

#[test]
fn rejects_mismatched_vector_length() {
    let vectors = vec![1.0f32, 2.0, 3.0, 4.0, 5.0]; // not a multiple of dim=3
    let config = IvfBuildConfig::default();
    assert!(IvfIndex::build(&vectors, 3, &config).is_err());
}

// ---------------------------------------------------------------------------
// Integration tests with Vortex Vector arrays
// ---------------------------------------------------------------------------

fn test_session() -> VortexSession {
    let session = VortexSession::empty().with::<ArraySession>();
    crate::initialize(&session);
    session
}

/// Build a `Vector<dim, f32>` extension array from a flat f32 slice.
fn vector_array_f32(dim: u32, values: &[f32]) -> VortexResult<ArrayRef> {
    let dim_usize = dim as usize;
    assert_eq!(values.len() % dim_usize, 0);
    let num_rows = values.len() / dim_usize;

    let mut buf = BufferMut::<f32>::with_capacity(values.len());
    for &v in values {
        buf.push(v);
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    let fsl =
        FixedSizeListArray::try_new(elements.into_array(), dim, Validity::NonNullable, num_rows)?;

    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
}

#[test]
fn build_index_from_vortex_array() -> VortexResult<()> {
    // 6 vectors of dim=4
    #[rustfmt::skip]
    let values: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.9, 0.1, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.9, 0.1, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.9, 0.1,
    ];
    let data = vector_array_f32(4, &values)?;

    let config = IvfBuildConfig {
        num_clusters: 3,
        max_iterations: 20,
        seed: 42,
    };
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let index = build_ivf_index(&data, &config, &mut ctx)?;

    assert_eq!(index.num_vectors(), 6);
    assert_eq!(index.num_clusters(), 3);
    assert_eq!(index.dim(), 4);

    // Vectors 0,1 should be in the same cluster; 2,3 in another; 4,5 in a third.
    let a = index.assignments();
    assert_eq!(a[0], a[1]);
    assert_eq!(a[2], a[3]);
    assert_eq!(a[4], a[5]);
    assert_ne!(a[0], a[2]);
    assert_ne!(a[0], a[4]);
    assert_ne!(a[2], a[4]);

    Ok(())
}

#[test]
fn ivf_search_finds_correct_matches() -> VortexResult<()> {
    // 8 vectors of dim=3 in 2 clusters.
    #[rustfmt::skip]
    let values: Vec<f32> = vec![
        // Cluster A: near [1, 0, 0]
        1.0, 0.0, 0.0,
        0.95, 0.05, 0.0,
        0.9, 0.1, 0.0,
        0.98, 0.02, 0.0,
        // Cluster B: near [0, 1, 0]
        0.0, 1.0, 0.0,
        0.05, 0.95, 0.0,
        0.1, 0.9, 0.0,
        0.02, 0.98, 0.0,
    ];
    let data = vector_array_f32(3, &values)?;

    let config = IvfBuildConfig {
        num_clusters: 2,
        max_iterations: 20,
        seed: 42,
    };
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let index = build_ivf_index(&data, &config, &mut ctx)?;

    // Search with query near [1, 0, 0], threshold 0.8, probing 1 cluster.
    let query = [0.99f32, 0.01, 0.0];
    let result = ivf_similarity_search(data, &index, &query, 0.8f32, 1, &mut ctx)?;

    let bits = result.to_bit_buffer();
    assert_eq!(bits.len(), 8);

    // First 4 vectors should match (they're in the probed cluster and have high similarity).
    for i in 0..4 {
        assert!(
            bits.value(i),
            "vector {i} should match (high similarity in probed cluster)"
        );
    }
    // Last 4 are in the non-probed cluster, so they should be false regardless.
    for i in 4..8 {
        assert!(
            !bits.value(i),
            "vector {i} should not match (not in probed cluster)"
        );
    }

    Ok(())
}

#[test]
fn ivf_search_with_turboquant() -> VortexResult<()> {
    // Test that IVF works correctly when data is TurboQuant-compressed.
    // TQ requires dim >= 128, so we use 128-dim vectors.
    const DIM: u32 = 128;
    const NUM_ROWS: usize = 200;
    const NUM_CLUSTERS: usize = 4;

    let mut vectors = generate_clustered_vectors(NUM_ROWS, DIM as usize, NUM_CLUSTERS, 42);
    normalize_vectors(&mut vectors, DIM as usize);

    let data = vector_array_f32(DIM, &vectors)?;
    let session = test_session();
    let mut ctx = session.create_execution_ctx();

    // Build IVF index from uncompressed data.
    #[expect(clippy::cast_possible_truncation, reason = "test constant fits in u32")]
    let ivf_config = IvfBuildConfig {
        num_clusters: NUM_CLUSTERS as u32,
        max_iterations: 20,
        seed: 42,
    };
    let index = build_ivf_index(&data, &ivf_config, &mut ctx)?;

    // Compress data with TurboQuant.
    let compressed = compress_turboquant(data, &mut ctx)?;
    assert_eq!(compressed.len(), NUM_ROWS);

    // Search: use the query vector = first vector in the dataset.
    // It should match itself with high cosine similarity.
    let query: Vec<f32> = vectors[..DIM as usize].to_vec();

    // IVF search with nprobes=2 (search half the clusters).
    let result = ivf_similarity_search(compressed, &index, &query, 0.8f32, 2, &mut ctx)?;
    let bits = result.to_bit_buffer();
    assert_eq!(bits.len(), NUM_ROWS);

    // The query's own row (row 0) should be found.
    assert!(
        bits.value(0),
        "row 0 (identical to query) must be found even with IVF pruning"
    );

    // Count how many rows were selected — should be significantly fewer than all rows.
    let selected: usize = (0..NUM_ROWS).filter(|&i| bits.value(i)).count();
    assert!(
        selected < NUM_ROWS,
        "IVF should prune some rows, but all {NUM_ROWS} were selected"
    );

    Ok(())
}
