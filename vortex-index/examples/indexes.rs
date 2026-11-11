// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::use_debug, reason = "example output favors quick debug prints")]
#![expect(clippy::print_stdout, reason = "examples print their results")]

//! The three index kinds, used two ways:
//!
//! 1. **Inside a data file.** The writer attaches the indexes to their columns, and a scan's
//!    filters use them with no change to the query. Ranked search opens the index directly.
//! 2. **As standalone index files in a catalog.** Data files are written plain, and index files
//!    covering several of them sit beside them. A query asks the index files which files and rows
//!    to read, then scans only those.
//!
//! Run it with:
//!
//! ```text
//! cargo run -p vortex-index --example indexes
//! ```

use std::sync::Arc;

use futures::TryStreamExt;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::InvertedIndex;
use vortex_index::IvfIndex;
use vortex_index::IvfOptions;
use vortex_index::TrigramIndex;
use vortex_index::open_in_file_index;
use vortex_index::text_match;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::indexed::IndexConfig;
use vortex_layout::layouts::indexed::IndexedStrategy;
use vortex_session::VortexSession;

const DIM: u32 = 4;

/// A tiny table of documents with embeddings.
fn documents(texts: &[&str]) -> VortexResult<ArrayRef> {
    let ids: PrimitiveArray = (0..i32::try_from(texts.len())?).collect();
    // A made-up embedding: which of four topics the document mentions.
    let elements: PrimitiveArray = texts
        .iter()
        .flat_map(|t| {
            ["rust", "storage", "search", "cooking"]
                .map(|topic| if t.contains(topic) { 1.0f32 } else { 0.1 })
        })
        .collect();
    let vectors = FixedSizeListArray::new(
        elements.into_array(),
        DIM,
        Validity::NonNullable,
        texts.len(),
    );
    Ok(StructArray::from_fields(&[
        ("id", ids.into_array()),
        ("text", VarBinViewArray::from_iter_str(texts).into_array()),
        ("vec", vectors.into_array()),
    ])?
    .into_array())
}

async fn write(
    session: &VortexSession,
    array: ArrayRef,
    with_indexes: bool,
) -> VortexResult<VortexFile> {
    let mut options = session.write_options();
    if with_indexes {
        let index = |configs| {
            Arc::new(IndexedStrategy::new(
                ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                configs,
            ))
        };
        let strategy = WriteStrategyBuilder::from_session(session)
            .with_field_writer(
                FieldPath::from_name("text"),
                index(vec![
                    IndexConfig::with_defaults(TrigramIndex::new_ref()),
                    IndexConfig::with_defaults(InvertedIndex::new_ref()),
                ]),
            )
            .with_field_writer(
                FieldPath::from_name("vec"),
                index(vec![IndexConfig::with_defaults(IvfIndex::new_ref())]),
            )
            .build();
        options = options.with_strategy(strategy);
    }
    let mut bytes = ByteBufferMut::empty();
    options.write(&mut bytes, array.to_array_stream()).await?;
    session
        .open_options()
        .include_metadata()
        .open_buffer(bytes.freeze())
}

/// Ids of the rows of `file` passing `filter`.
async fn ids(file: &VortexFile, filter: Expression) -> VortexResult<Vec<i32>> {
    let chunks: Vec<ArrayRef> = file
        .scan()?
        .with_filter(filter.bind(file.dtype())?)
        .with_projection(col("id").bind(file.dtype())?)
        .into_array_stream()?
        .try_collect()
        .await?;
    let mut ctx = file.session().create_execution_ctx();
    let mut ids = Vec::new();
    for chunk in chunks {
        ids.extend_from_slice(chunk.execute::<PrimitiveArray>(&mut ctx)?.as_slice::<i32>());
    }
    Ok(ids)
}

#[tokio::main]
async fn main() -> VortexResult<()> {
    let session = VortexSession::default();
    vortex_index::initialize(&session)?;

    // 1. Indexes inside the data file.
    let file = write(
        &session,
        documents(&[
            "rust storage engines",
            "search over rust columnar storage",
            "slow cooking recipes",
            "full text search basics",
        ])?,
        true,
    )
    .await?;
    println!(
        "in-file LIKE '%colum%' ids: {:?}",
        ids(&file, like(col("text"), lit("%colum%"))).await?
    );
    println!(
        "in-file text_match 'rust search' ids: {:?}",
        ids(&file, text_match(col("text"), "rust search")).await?
    );
    let fts =
        open_in_file_index(InvertedIndex, &file, "text")?.ok_or_else(|| vortex_err!("indexed"))?;
    println!(
        "in-file BM25 'search rust': {:?}",
        fts.search_bm25("search rust", 2).await?
    );
    let ivf = open_in_file_index(IvfIndex, &file, "vec")?.ok_or_else(|| vortex_err!("indexed"))?;
    println!(
        "in-file nearest to [cooking]: {:?}",
        ivf.search_vectors(&[0.1, 0.1, 0.1, 1.0], 1, 4).await?
    );

    // 2. Plain data files plus standalone index files, as a catalog would hold them.
    let catalog = [
        (
            "docs-0.vortex",
            documents(&["rust storage", "cooking with rust pans"])?,
        ),
        (
            "docs-1.vortex",
            documents(&["vector search", "storage tiers"])?,
        ),
    ];
    let mut data_files = Vec::new();
    for (_, array) in &catalog {
        data_files.push(write(&session, array.clone(), false).await?);
    }

    let mut fts = IndexFileBuilder::new(
        InvertedIndex,
        (),
        "text",
        DType::Utf8(NonNullable),
        &session,
    )?;
    let vec_dtype = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::F32, NonNullable)),
        DIM,
        NonNullable,
    );
    let mut vectors =
        IndexFileBuilder::new(IvfIndex, IvfOptions::default(), "vec", vec_dtype, &session)?;
    for ((uri, _), file) in catalog.iter().zip(&data_files) {
        fts.add_file(*uri, file).await?;
        vectors.add_file(*uri, file).await?;
    }
    let mut fts_bytes = ByteBufferMut::empty();
    fts.write(&mut fts_bytes).await?;
    let mut vec_bytes = ByteBufferMut::empty();
    vectors.write(&mut vec_bytes).await?;
    let open = |bytes: ByteBufferMut| {
        session
            .open_options()
            .include_metadata()
            .open_buffer(bytes.freeze())
    };
    let fts = IndexFile::open(InvertedIndex, &open(fts_bytes)?)?;
    let vectors = IndexFile::open(IvfIndex, &open(vec_bytes)?)?;

    // Filter: the index says which files and rows to read; the other files are never opened.
    let (_, selections) = fts
        .locate_files(&text_match(root(), "storage"))
        .await?
        .ok_or_else(|| vortex_err!("claimed"))?;
    for selection in selections {
        let uri = &fts.files()[selection.file].uri;
        println!(
            "catalog text_match 'storage' -> {uri}: {:?}",
            selection.selection
        );
    }
    // Ranked: top-k across every covered file.
    let hits = fts.hits_by_file(&fts.rows().search_bm25("rust", 3).await?)?;
    println!("catalog BM25 'rust': {hits:?}");
    let hits = vectors.hits_by_file(
        &vectors
            .rows()
            .search_vectors(&[0.1, 0.1, 1.0, 0.1], 1, 2)
            .await?,
    )?;
    println!("catalog nearest to [search]: {hits:?}");
    Ok(())
}
