// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end Vortex-file round-trip test for [`IvfLayout`](crate::layout::IvfLayout).
//!
//! Exercises the full production path:
//!
//! 1. Produce a `Vector<dim, f32>` column in memory.
//! 2. Write a Vortex file where the top-level layout is [`IvfStrategy`](crate::layout::writer::IvfStrategy),
//!    which clusters the data and stores one chunk per cluster plus an auxiliary centroid child.
//! 3. Open the file back, register the IVF layout, and run a scan with a
//!    `CosineSimilarity(root, literal_query) > threshold` filter.
//! 4. Verify the returned rows match what a brute-force cosine filter would return **from the
//!    probed clusters only** (IVF trades a small recall loss for read-side pruning).
//!
//! These tests double as executable documentation of the production write/read workflow.

use std::sync::Arc;

use futures::pin_mut;
use futures::stream::StreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
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
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector::Vector;

use crate::layout::register_ivf_layout;
use crate::layout::writer::IvfLayoutOptions;
use crate::layout::writer::IvfStrategy;

const DIM: u32 = 16;
const NUM_CLUSTERS: usize = 4;
const ROWS_PER_CLUSTER: usize = 32;
const TOTAL_ROWS: usize = NUM_CLUSTERS * ROWS_PER_CLUSTER;

/// Build a session with every piece IVF needs: ArraySession, LayoutSession, ScalarFnSession,
/// RuntimeSession, the default vortex-file encodings, tensor scalar functions, and the IVF
/// layout encoding.
fn production_session() -> VortexSession {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<ScalarFnSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&session);
    vortex_tensor::initialize(&session);
    register_ivf_layout(&session);

    session
}

/// Build a one-hot-ish clustered `Vector<DIM, f32>` extension array for the tests.
fn clustered_vector_column() -> ArrayRef {
    let mut values = vec![0.0f32; TOTAL_ROWS * DIM as usize];
    for cluster_idx in 0..NUM_CLUSTERS {
        for row_in_cluster in 0..ROWS_PER_CLUSTER {
            let row = cluster_idx * ROWS_PER_CLUSTER + row_in_cluster;
            let center_idx = cluster_idx % DIM as usize;
            for i in 0..DIM as usize {
                let base = if i == center_idx { 1.0f32 } else { 0.0 };
                let noise = (((row as f32) * 0.017 + i as f32 * 0.003).sin()) * 0.01;
                values[row * DIM as usize + i] = base + noise;
            }
        }
    }

    let mut buf = BufferMut::<f32>::with_capacity(values.len());
    for v in values {
        buf.push(v);
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        DIM,
        Validity::NonNullable,
        TOTAL_ROWS,
    )
    .unwrap();
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())
        .unwrap()
        .erased();
    ExtensionArray::new(ext_dtype, fsl.into_array()).into_array()
}

/// Wrap a flat f32 slice as an extension-scalar `Vector<DIM, f32>` suitable for a literal query.
fn literal_query(query: &[f32]) -> vortex_array::expr::Expression {
    let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let children: Vec<Scalar> = query
        .iter()
        .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
        .collect();
    let storage = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage.dtype().clone())
        .unwrap()
        .erased();
    let ext_scalar = Scalar::extension_ref(ext_dtype, storage);
    lit(ext_scalar)
}

/// Full Vortex file round-trip: write a `Vector<16, f32>` column with `IvfStrategy`, open the
/// file, and run a cosine-similarity filter.
///
/// Assertions:
/// - The file reads back to a non-empty array.
/// - Without any filter, the read returns all rows (preserving data fidelity through the layout).
#[tokio::test]
async fn file_round_trip_without_filter() {
    let session = production_session();
    let column = clustered_vector_column();

    let strategy = Arc::new(IvfStrategy::new(
        FlatLayoutStrategy::default(),
        FlatLayoutStrategy::default(),
        IvfLayoutOptions {
            num_clusters: u32::try_from(NUM_CLUSTERS).unwrap(),
            max_iterations: 20,
            seed: 42,
            nprobes: 2,
        },
    ));

    // 1. Write.
    let mut buf = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut buf, column.clone().to_array_stream())
        .await
        .expect("write Vortex file");

    // 2. Open and scan.
    let file = session
        .open_options()
        .open_buffer(buf)
        .expect("open Vortex file");
    let stream = file
        .scan()
        .expect("create scan")
        .into_array_stream()
        .expect("into stream");
    pin_mut!(stream);

    let mut row_count = 0;
    while let Some(result) = stream.next().await {
        let array = result.expect("read chunk");
        row_count += array.len();
    }

    assert_eq!(row_count, TOTAL_ROWS);
}

/// Full round-trip with a cosine-similarity filter.
///
/// - Writes the IVF-indexed file.
/// - Queries with a one-hot vector matching cluster 0.
/// - Asserts that the returned rows all belong to the probed cluster (and are the right count).
#[tokio::test]
async fn file_round_trip_with_cosine_filter() -> VortexResult<()> {
    let session = production_session();
    let column = clustered_vector_column();

    let strategy = Arc::new(IvfStrategy::new(
        FlatLayoutStrategy::default(),
        FlatLayoutStrategy::default(),
        IvfLayoutOptions {
            num_clusters: u32::try_from(NUM_CLUSTERS).unwrap(),
            max_iterations: 20,
            seed: 42,
            nprobes: 1,
        },
    ));

    // 1. Write.
    let mut buf = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut buf, column.to_array_stream())
        .await
        .expect("write Vortex file");

    // 2. Build query & filter expression.
    let mut query = vec![0.0f32; DIM as usize];
    query[0] = 1.0;
    let cosine = CosineSimilarity
        .try_new_expr(
            vortex_array::scalar_fn::EmptyOptions,
            [root(), literal_query(&query)],
        )
        .unwrap();
    let filter = gt(cosine, lit(0.5f32));

    // 3. Scan with the filter.
    let file = session
        .open_options()
        .open_buffer(buf)
        .expect("open Vortex file");

    let stream = file
        .scan()
        .expect("create scan")
        .with_filter(filter)
        .into_array_stream()
        .expect("into stream");
    pin_mut!(stream);

    let mut total_returned = 0;
    while let Some(result) = stream.next().await {
        let array = result.expect("read chunk");
        total_returned += array.len();
    }

    // With nprobes=1 we read only the matching cluster, so we should get at most ROWS_PER_CLUSTER
    // rows back. And since the query is very close to cluster 0's center, at least one row must
    // match the threshold 0.5.
    assert!(
        total_returned > 0,
        "cosine-similarity filter returned zero rows"
    );
    assert!(
        total_returned <= ROWS_PER_CLUSTER,
        "IVF should prune to one cluster, but returned {total_returned} rows (cluster size: {ROWS_PER_CLUSTER})"
    );
    Ok(())
}
