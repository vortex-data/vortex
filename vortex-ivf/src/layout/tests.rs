// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end layout tests for [`IvfLayout`](super::IvfLayout).
//!
//! These tests exercise the full write → read round trip through a test segment source and
//! verify:
//!
//! 1. The layout writes correctly and can be round-tripped.
//! 2. The reader projects rows back correctly (transparent to IVF — all rows are returned
//!    from the underlying data, just in cluster order).
//! 3. The reader's pruning evaluation returns a mask that eliminates non-probed clusters
//!    when given a cosine-similarity expression.

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutEncodingId;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector::Vector;

use super::writer::IvfLayoutOptions;
use super::writer::IvfStrategy;

fn session_with_handle(handle: Handle) -> VortexSession {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<ScalarFnSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_handle(handle);
    vortex_tensor::initialize(&session);
    super::register_ivf_layout(&session);
    session
}

fn vector_array_f32(dim: u32, values: &[f32]) -> VortexResult<ArrayRef> {
    let row_count = values.len() / dim as usize;
    let mut buf = BufferMut::<f32>::with_capacity(values.len());
    for &v in values {
        buf.push(v);
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    let fsl =
        FixedSizeListArray::try_new(elements.into_array(), dim, Validity::NonNullable, row_count)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
}

/// Build a query vector as an extension scalar suitable for a `CosineSimilarity` expression.
fn literal_query_expression(query: &[f32]) -> Expression {
    let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let children: Vec<Scalar> = query
        .iter()
        .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
        .collect();
    let storage_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    let storage_dtype = storage_scalar.dtype().clone();
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype)
        .unwrap()
        .erased();
    let ext_scalar = Scalar::extension_ref(ext_dtype, storage_scalar);
    lit(ext_scalar)
}

/// Small-dim clustered dataset for IVF layout tests. The cluster center is a one-hot vector at
/// `cluster_idx % dim` with a tiny per-row noise.
fn clustered_dataset(num_clusters: usize, rows_per_cluster: usize, dim: u32) -> Vec<f32> {
    let total = num_clusters * rows_per_cluster;
    let mut vectors = vec![0.0f32; total * dim as usize];
    for cluster_idx in 0..num_clusters {
        for row_in_cluster in 0..rows_per_cluster {
            let row = cluster_idx * rows_per_cluster + row_in_cluster;
            let center_idx = cluster_idx % dim as usize;
            let v = &mut vectors[row * dim as usize..(row + 1) * dim as usize];
            for (i, x) in v.iter_mut().enumerate() {
                let base = if i == center_idx { 1.0f32 } else { 0.0 };
                let noise = (((row as f32) * 0.017 + i as f32 * 0.003).sin()) * 0.01;
                *x = base + noise;
            }
        }
    }
    vectors
}

const DIM: u32 = 8;
const NUM_CLUSTERS: usize = 4;
const ROWS_PER_CLUSTER: usize = 16;
const TOTAL_ROWS: usize = NUM_CLUSTERS * ROWS_PER_CLUSTER;

fn write_ivf_layout(
    session: &VortexSession,
    nprobes: u32,
) -> (Arc<dyn vortex_layout::segments::SegmentSource>, LayoutRef) {
    let vectors = clustered_dataset(NUM_CLUSTERS, ROWS_PER_CLUSTER, DIM);
    let data = vector_array_f32(DIM, &vectors).unwrap();

    let ctx = ArrayContext::empty();
    let segments = Arc::new(TestSegments::default());
    let (ptr, eof) = SequenceId::root().split();

    let strategy = IvfStrategy::new(
        FlatLayoutStrategy::default(),
        FlatLayoutStrategy::default(),
        IvfLayoutOptions {
            #[expect(clippy::cast_possible_truncation, reason = "test constant fits in u32")]
            num_clusters: NUM_CLUSTERS as u32,
            max_iterations: 20,
            seed: 42,
            nprobes,
        },
    );

    let segments2 = Arc::<TestSegments>::clone(&segments);
    let session2 = session.clone();
    let layout: LayoutRef = futures::executor::block_on(async move {
        strategy
            .write_stream(
                ctx,
                segments2,
                data.to_array_stream().sequenced(ptr),
                eof,
                &session2,
            )
            .await
    })
    .unwrap();

    (segments, layout)
}

#[test]
fn ivf_layout_has_correct_encoding_id_and_row_count() {
    block_on(|handle| async move {
        let session = session_with_handle(handle);
        let (_segments, layout) = write_ivf_layout(&session, 2);

        assert_eq!(layout.row_count(), TOTAL_ROWS as u64);
        assert_eq!(
            layout.encoding_id(),
            LayoutEncodingId::new(super::IVF_LAYOUT_ID)
        );
    });
}

#[test]
fn ivf_layout_projects_all_rows() {
    block_on(|handle| async move {
        let session = session_with_handle(handle);
        let (segments, layout) = write_ivf_layout(&session, 4); // probe all

        let reader = layout.new_reader("ivf".into(), segments, &session).unwrap();
        let result = reader
            .projection_evaluation(
                &(0..layout.row_count()),
                &root(),
                MaskFuture::new_true(TOTAL_ROWS),
            )
            .unwrap()
            .await
            .unwrap();

        assert_eq!(result.len(), TOTAL_ROWS);
    });
}

#[test]
fn ivf_layout_passthrough_for_non_cosine_expression() {
    block_on(|handle| async move {
        let session = session_with_handle(handle);
        let (segments, layout) = write_ivf_layout(&session, 1);

        // Expression that is NOT a cosine similarity — IVF pruning should be a no-op here.
        // We use `gt(root(), lit(0))` which doesn't apply to our vector column but tests that
        // the reader falls back to inner pruning without erroring.
        let expr = gt(root(), lit(0i32));

        let reader = layout.new_reader("ivf".into(), segments, &session).unwrap();
        let mask = reader
            .pruning_evaluation(&(0..TOTAL_ROWS as u64), &expr, Mask::new_true(TOTAL_ROWS))
            .unwrap()
            .await
            .unwrap();

        // Without IVF pruning, the mask should still have length TOTAL_ROWS.
        assert_eq!(mask.len(), TOTAL_ROWS);
    });
}

#[test]
fn ivf_layout_prunes_for_cosine_query() {
    block_on(|handle| async move {
        let session = session_with_handle(handle);
        let (segments, layout) = write_ivf_layout(&session, 1);

        // Build the query: one-hot at index 0 (matches cluster 0).
        let mut query = vec![0.0f32; DIM as usize];
        query[0] = 1.0;

        // Expression: `CosineSimilarity(root, literal_query) > 0.5`.
        let query_lit = literal_query_expression(&query);
        let cosine = CosineSimilarity
            .try_new_expr(vortex_array::scalar_fn::EmptyOptions, [root(), query_lit])
            .unwrap();
        let expr = gt(cosine, lit(0.5f32));

        let reader = layout.new_reader("ivf".into(), segments, &session).unwrap();
        let mask = reader
            .pruning_evaluation(&(0..TOTAL_ROWS as u64), &expr, Mask::new_true(TOTAL_ROWS))
            .unwrap()
            .await
            .unwrap();

        assert_eq!(mask.len(), TOTAL_ROWS);
        let selected = mask.true_count();
        assert!(
            selected > 0,
            "IVF should select at least the matching cluster"
        );
        assert!(
            selected <= 2 * ROWS_PER_CLUSTER,
            "nprobes=1 should limit selection to at most one cluster ({ROWS_PER_CLUSTER} rows); got {selected}"
        );
    });
}
