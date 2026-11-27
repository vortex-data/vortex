// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builds IVF indexes inside a file and as a standalone index file, and checks searches against
//! exact cosine.

#![expect(clippy::tests_outside_test_module)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test data sizes are small constants"
)]

use std::sync::Arc;

use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_index::FileSelection;
use vortex_index::Hit;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::IndexFileInfo;
use vortex_index::open_in_file_index;
use vortex_index_ivf::IvfIndex;
use vortex_index_ivf::IvfOptions;
use vortex_index_ivf::VectorSearch;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::indexed::IndexConfig;
use vortex_layout::layouts::indexed::IndexedStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

const ROWS: usize = 3000;
const BLOCK_LEN: usize = 64;
const DIM: usize = 8;

fn session() -> VortexResult<VortexSession> {
    let session = VortexSession::default();
    vortex_index::initialize(&session)?;
    vortex_index_ivf::register(&session);
    Ok(session)
}

/// A small deterministic generator, so test data needs no extra dependencies.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f32 {
        (self.next() % 10_000) as f32 / 10_000.0
    }
}

/// Vectors clustered around 4 directions, so IVF lists are meaningful.
fn vectors(seed: u64) -> Vec<[f32; DIM]> {
    let mut rng = Lcg(seed);
    (0..ROWS)
        .map(|row| {
            let mut v = [0f32; DIM];
            v[row % 4] = 4.0;
            v.iter_mut().for_each(|x| *x += rng.unit() - 0.5);
            v
        })
        .collect()
}

fn array(vectors: &[[f32; DIM]]) -> VortexResult<ArrayRef> {
    let ids: PrimitiveArray = (0..vectors.len() as i32).collect();
    let elements: PrimitiveArray = vectors.iter().flatten().copied().collect();
    let vectors = FixedSizeListArray::new(
        elements.into_array(),
        DIM as u32,
        Validity::NonNullable,
        vectors.len(),
    );
    Ok(
        StructArray::from_fields(&[("id", ids.into_array()), ("vec", vectors.into_array())])?
            .into_array(),
    )
}

/// Exact cosine top-k rows.
fn nearest(vectors: &[[f32; DIM]], query: &[f32], k: usize) -> Vec<u64> {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut scored: Vec<(u64, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(row, v)| {
            let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
            (row as u64, dot / (norm(v) * norm(query)))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().take(k).map(|(row, _)| row).collect()
}

fn data_strategy() -> Arc<dyn LayoutStrategy> {
    Arc::new(RepartitionStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: BLOCK_LEN,
            block_size_target: None,
            canonicalize: false,
        },
    ))
}

/// Write `vectors` with an IVF index on `vec` inside the file.
async fn write_indexed_file(
    session: &VortexSession,
    vectors: &[[f32; DIM]],
) -> VortexResult<VortexFile> {
    let indexed = Arc::new(
        IndexedStrategy::new(
            data_strategy(),
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            vec![IndexConfig::with_defaults(IvfIndex::new_ref())],
        )
        .with_data_block_len(BLOCK_LEN as u64),
    );
    let strategy = WriteStrategyBuilder::from_session(session)
        .with_row_block_size(BLOCK_LEN)
        .with_field_writer(FieldPath::from_name("vec"), indexed)
        .build();
    let mut bytes = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut bytes, array(vectors)?.to_array_stream())
        .await?;
    session.open_options().open_buffer(bytes.freeze())
}

#[tokio::test]
async fn in_file_ivf_finds_nearest_vectors() -> VortexResult<()> {
    let session = session()?;
    let vectors = vectors(7);
    let file = write_indexed_file(&session, &vectors).await?;
    let index = open_in_file_index(IvfIndex, &file, "vec")?.expect("ivf index");
    let lists = index.options().lists as usize;
    assert!(lists > 1);

    let query = vectors[1234];
    let hits = index.search_vectors(&query, 10, lists).await?;
    assert_eq!(hits[0].row, 1234);
    assert!(hits[0].score > 0.99);

    // Probing every list leaves only quantization error, so recall is near perfect.
    let expected = nearest(&vectors, &query, 10);
    let found = hits.iter().filter(|h| expected.contains(&h.row)).count();
    assert!(found >= 8, "recall {found}/10");
    Ok(())
}

#[tokio::test]
async fn ivf_searches_answer_the_same_from_cached_lists() -> VortexResult<()> {
    let session = session()?;
    let vectors = vectors(7);
    let file = write_indexed_file(&session, &vectors).await?;
    let index = open_in_file_index(IvfIndex, &file, "vec")?.expect("ivf index");
    let lists = index.options().lists as usize;
    let rows = |hits: Vec<Hit>| {
        hits.into_iter()
            .map(|h| (h.row, h.score))
            .collect::<Vec<_>>()
    };

    // The first search probes half the lists and keeps them; the second probes them all, so it
    // reads the rest and takes the first half from the cache.
    let query = vectors[777];
    let half = index.search_vectors(&query, 20, lists / 2).await?;
    let all_cold = index.search_vectors(&query, 20, lists).await?;
    let all_warm = index.search_vectors(&query, 20, lists).await?;
    assert_eq!(rows(all_cold.clone()), rows(all_warm));
    assert_eq!(
        rows(half),
        rows(index.search_vectors(&query, 20, lists / 2).await?)
    );

    let fresh = open_in_file_index(IvfIndex, &file, "vec")?.expect("ivf index");
    assert_eq!(
        rows(all_cold),
        rows(fresh.search_vectors(&query, 20, lists).await?)
    );
    Ok(())
}

/// Every stored vector is its own nearest neighbour when every list is probed, so a lookup that
/// read the wrong lists, or a build that numbered rows out of order, shows up as a wrong row.
#[tokio::test]
async fn standalone_ivf_reads_probed_lists_by_position() -> VortexResult<()> {
    let session = session()?;
    let vectors = vectors(3);
    let mut bytes = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut bytes, array(&vectors)?.to_array_stream())
        .await?;
    let file = session.open_options().open_buffer(bytes.freeze())?;
    let mut builder = IndexFileBuilder::new(
        IvfIndex,
        IvfOptions { lists: 16, dim: 0 },
        "vec",
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            DIM as u32,
            Nullability::NonNullable,
        ),
        &session,
    )?;
    builder.add_file("part.vortex", &file).await?;
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let ivf = IndexFile::open(
        IvfIndex,
        &session
            .open_options()
            .include_metadata()
            .open_buffer(bytes.freeze())?,
    )?;
    let lists = ivf.rows().options().lists as usize;
    for row in (0..ROWS).step_by(97) {
        let hits = ivf.rows().search_vectors(&vectors[row], 1, lists).await?;
        assert_eq!(hits[0].row, row as u64);
    }
    Ok(())
}

/// Hits in an index over several files map to a selection of local rows in each file they fall
/// in, and an index opened from a separately decoded description searches the same.
#[tokio::test]
async fn standalone_hits_select_local_rows_per_file() -> VortexResult<()> {
    let session = session()?;
    let parts = [vectors(5), vectors(6)];
    let mut builder = IndexFileBuilder::new(
        IvfIndex,
        IvfOptions { lists: 16, dim: 0 },
        "vec",
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            DIM as u32,
            Nullability::NonNullable,
        ),
        &session,
    )?;
    for (i, part) in parts.iter().enumerate() {
        let mut bytes = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut bytes, array(part)?.to_array_stream())
            .await?;
        let file = session.open_options().open_buffer(bytes.freeze())?;
        builder.add_file(format!("part{i}.vortex"), &file).await?;
    }
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let file = session
        .open_options()
        .include_metadata()
        .open_buffer(bytes.freeze())?;
    let ivf = IndexFile::open(IvfIndex, &file)?;
    let metadata = file
        .metadata_segment(vortex_index::standalone::INDEX_FILE_METADATA_KEY)
        .expect("index description");
    let info = IndexFileInfo::decode(metadata.as_slice(), &session)?;
    let decoded = IndexFile::open_with_info(IvfIndex, &file, info)?;

    let lists = ivf.rows().options().lists as usize;
    let query = parts[1][42];
    let hits = ivf.rows().search_vectors(&query, 20, lists).await?;
    assert_eq!(
        hits,
        decoded.rows().search_vectors(&query, 20, lists).await?
    );

    let selections = ivf.hit_selections(&hits)?;
    let mut expected: Vec<(usize, u64)> = ivf
        .hits_by_file(&hits)?
        .into_iter()
        .map(|hit| (hit.file, hit.row))
        .collect();
    expected.sort_unstable();
    let mut selected = Vec::new();
    for FileSelection { file, selection } in selections {
        let Selection::IncludeRoaring(rows) = selection else {
            panic!("hit selections are roaring row sets");
        };
        selected.extend(rows.iter().map(|row| (file, row)));
    }
    assert_eq!(selected, expected);
    assert!(selected.contains(&(1, 42)));
    Ok(())
}
