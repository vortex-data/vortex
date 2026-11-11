// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use futures::TryStreamExt;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::WriteOptionsSessionExt;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::InvertedIndex;
use vortex_index::IvfIndex;
use vortex_index::IvfOptions;
use vortex_index::text_match;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_scan::planning::drive::ParallelDriver;
use vortex_scan::planning::drive::ScopedBatch;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use super::*;
use crate::ManifestBuilder;
use crate::MemoryStorage;
use crate::TableScan;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default();
    vortex_index::initialize(&session).unwrap_or_else(|e| panic!("{e}"));
    session
});

const ROWS: u64 = 500;
const DIM: usize = 8;
const DIM_U32: u32 = 8;

/// Component `d` of row `id`'s vector: a hash, so no two rows' vectors point the same way.
fn component(id: u64, d: usize) -> f32 {
    let mut x =
        id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (d as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    (x % 2001) as f32 / 1000.0 - 1.0
}

/// File `f` holds ids `f * 500..`; every 97th id's text holds "needle", and each vector is
/// distinct.
async fn data_file(file: u64) -> VortexResult<ByteBuffer> {
    let ids: Vec<u64> = (file * ROWS..(file + 1) * ROWS).collect();
    let text: Vec<String> = ids
        .iter()
        .map(|id| {
            if id % 97 == 0 {
                format!("hay needle {id}")
            } else {
                format!("hay stack {id}")
            }
        })
        .collect();
    let vectors: Vec<f32> = ids
        .iter()
        .flat_map(|&id| (0..DIM).map(move |d| component(id, d)))
        .collect();
    let vec = FixedSizeListArray::try_new(
        PrimitiveArray::from_iter(vectors).into_array(),
        DIM_U32,
        Validity::NonNullable,
        ids.len(),
    )?;
    let array = StructArray::from_fields(&[
        ("id", PrimitiveArray::from_iter(ids).into_array()),
        ("text", VarBinArray::from(text).into_array()),
        ("vec", vec.into_array()),
    ])?;
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.into_array().to_array_stream())
        .await?;
    Ok(buf.freeze())
}

async fn table(files: u64) -> VortexResult<(Arc<MemoryStorage>, Manifest)> {
    let storage = Arc::new(MemoryStorage::new());
    let mut builder = ManifestBuilder::new(SESSION.clone());
    for file in 0..files {
        let path = format!("data/part-{file}.vortex");
        let bytes = data_file(file).await?;
        builder.add(path.clone(), &bytes).await?;
        storage.put(path, bytes);
    }
    Ok((storage, builder.finish()))
}

async fn build_index<V: IndexVTable + Clone>(
    vtable: V,
    options: V::Options,
    column: &str,
    dtype: DType,
    manifest: &Manifest,
    storage: &MemoryStorage,
) -> VortexResult<IndexFile<V>> {
    let mut builder = IndexFileBuilder::new(vtable.clone(), options, column, dtype, &SESSION)?;
    index_table(&mut builder, manifest, storage, &SESSION).await?;
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let file = open_index_file(&SESSION, Arc::new(bytes.freeze())).await?;
    IndexFile::open(vtable, &file)
}

async fn text_index(
    manifest: &Manifest,
    storage: &MemoryStorage,
) -> VortexResult<IndexFile<InvertedIndex>> {
    build_index(
        InvertedIndex,
        (),
        "text",
        DType::Utf8(Nullability::NonNullable),
        manifest,
        storage,
    )
    .await
}

/// The ids an ordered scan of `selections` returns: by file, then row.
async fn scan_ids(
    storage: &Arc<MemoryStorage>,
    manifest: &Manifest,
    selections: HashMap<u64, Selection>,
) -> VortexResult<Vec<u64>> {
    let batches: Vec<ScopedBatch> = TableScan::new(
        SESSION.clone(),
        Arc::clone(storage) as _,
        Arc::new(manifest.clone()),
    )
    .with_projection(select(["id"], root()))
    .with_file_selections(selections)
    .execute(&ParallelDriver::new(TokioRuntime::current()).with_ordered(true))
    .try_collect()
    .await?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut ids = Vec::new();
    for batch in batches {
        let column: ArrayRef = batch
            .batch
            .execute::<StructArray>(&mut ctx)?
            .unmasked_field_by_name("id")?
            .clone();
        ids.extend_from_slice(
            column
                .execute::<PrimitiveArray>(&mut ctx)?
                .as_slice::<u64>(),
        );
    }
    Ok(ids)
}

fn needles(files: u64) -> Vec<u64> {
    (0..files * ROWS).filter(|id| id % 97 == 0).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_index_selects_the_matching_rows_of_each_file() -> VortexResult<()> {
    let (storage, manifest) = table(3).await?;
    let index = TableIndex::bind(text_index(&manifest, &storage).await?, &manifest);
    assert!(index.uncovered().is_empty());
    let found = index
        .locate(&text_match(root(), "needle"))
        .await?
        .ok_or_else(|| vortex_err!("the inverted index claims text_match"))?;
    assert_eq!(found.exactness, IndexExactness::Exact);
    assert_eq!(
        scan_ids(&storage, &manifest, found.selections).await?,
        needles(3)
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn files_added_or_rewritten_after_the_index_are_scanned_in_full() -> VortexResult<()> {
    let (storage, manifest) = table(3).await?;
    let index = text_index(&manifest, &storage).await?;

    // The table grows by a file, and file 1 is rewritten with different rows.
    let (_, grown) = table(4).await?;
    let mut entries = grown.entries().to_vec();
    let rewritten = data_file(7).await?;
    let mut builder = ManifestBuilder::new(SESSION.clone());
    builder.add(entries[1].path.clone(), &rewritten).await?;
    entries[1] = builder.finish().entries()[0].clone();
    storage.put(entries[1].path.clone(), rewritten);
    storage.put(entries[3].path.clone(), data_file(3).await?);
    let manifest = Manifest::new(entries);

    let index = TableIndex::bind(index, &manifest);
    assert_eq!(index.uncovered(), &[1, 3]);
    let found = index
        .locate(&text_match(root(), "needle"))
        .await?
        .ok_or_else(|| vortex_err!("the inverted index claims text_match"))?;
    assert!(matches!(found.selections.get(&1), Some(Selection::All)));
    assert!(matches!(found.selections.get(&3), Some(Selection::All)));
    // Covered files still read only their matches.
    assert!(matches!(
        found.selections.get(&0),
        Some(Selection::IncludeRoaring(rows)) if rows.len() == 6
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ranked_hits_map_to_scanned_rows() -> VortexResult<()> {
    let (storage, manifest) = table(3).await?;
    let ivf = build_index(
        IvfIndex,
        IvfOptions { lists: 4, dim: 0 },
        "vec",
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            DIM_U32,
            Nullability::NonNullable,
        ),
        &manifest,
        &storage,
    )
    .await?;
    let index = TableIndex::bind(ivf, &manifest);
    // The stored vector of id 1234 is its own nearest neighbour.
    let target = 1234u64;
    let query: Vec<f32> = (0..DIM).map(|d| component(target, d)).collect();
    let hits = index.index().rows().search_vectors(&query, 5, 4).await?;
    let ranked = index.ranked(&hits)?;
    let ids = scan_ids(&storage, &manifest, ranked.selections.clone()).await?;
    assert_eq!(ids.len(), ranked.hits.len());
    let best = ranked.best_first()[0];
    assert_eq!(ids[best], target);
    Ok(())
}
