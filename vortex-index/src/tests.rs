// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    reason = "test data sizes are small constants"
)]

use std::sync::Arc;

use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::ilike;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::indexed::IndexConfig;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::IndexedStrategy;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::CoveredFile;
use crate::IndexFile;
use crate::IndexFileBuilder;
use crate::IndexFileInfo;
use crate::InvertedIndex;
use crate::IvfIndex;
use crate::IvfOptions;
use crate::TrigramIndex;
use crate::TrigramOptions;
use crate::initialize;
use crate::open_in_file_index;
use crate::rows::collect;
use crate::text_match;

const ROWS: usize = 3000;
const BLOCK_LEN: usize = 64;
const DIM: usize = 8;
const WORDS: [&str; 16] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa",
];
/// Rows holding the rare word `Needle`.
const NEEDLE_ROWS: [usize; 3] = [100, 1500, 2900];

fn session() -> VortexResult<VortexSession> {
    let session = VortexSession::default();
    initialize(&session)?;
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

struct Data {
    texts: Vec<Option<String>>,
    vectors: Vec<[f32; DIM]>,
}

impl Data {
    fn generate(seed: u64, rows: usize) -> Self {
        let mut rng = Lcg(seed);
        let texts = (0..rows)
            .map(|row| {
                if row % 37 == 5 {
                    return None;
                }
                let len = 3 + rng.next() as usize % 8;
                let mut words: Vec<&str> = (0..len)
                    .map(|_| WORDS[rng.next() as usize % WORDS.len()])
                    .collect();
                if NEEDLE_ROWS.contains(&row) {
                    words.insert(1, "Needle");
                }
                // A few repeated words, so BM25 term frequencies vary.
                if row % 11 == 0 {
                    words.push("echo");
                    words.push("echo");
                }
                Some(words.join(" "))
            })
            .collect();
        // Vectors clustered around 4 directions, so IVF lists are meaningful.
        let vectors = (0..rows)
            .map(|row| {
                let mut v = [0f32; DIM];
                v[row % 4] = 4.0;
                v.iter_mut().for_each(|x| *x += rng.unit() - 0.5);
                v
            })
            .collect();
        Self { texts, vectors }
    }

    fn array(&self) -> VortexResult<ArrayRef> {
        let ids: PrimitiveArray = (0..self.texts.len() as i32).collect();
        let texts =
            VarBinViewArray::from_iter_nullable_str(self.texts.iter().map(Option::as_deref));
        let elements: PrimitiveArray = self.vectors.iter().flatten().copied().collect();
        let vectors = FixedSizeListArray::new(
            elements.into_array(),
            DIM as u32,
            Validity::NonNullable,
            self.vectors.len(),
        );
        Ok(StructArray::from_fields(&[
            ("id", ids.into_array()),
            ("text", texts.into_array()),
            ("vec", vectors.into_array()),
        ])?
        .into_array())
    }

    fn rows_where(&self, pred: impl Fn(&str) -> bool) -> Vec<i32> {
        self.texts
            .iter()
            .enumerate()
            .filter(|(_, t)| t.as_deref().is_some_and(&pred))
            .map(|(row, _)| row as i32)
            .collect()
    }

    /// Exact cosine top-k rows.
    fn nearest(&self, query: &[f32], k: usize) -> Vec<u64> {
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut scored: Vec<(u64, f32)> = self
            .vectors
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

fn indexed(configs: Vec<IndexConfig>) -> Arc<dyn LayoutStrategy> {
    Arc::new(
        IndexedStrategy::new(
            data_strategy(),
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            configs,
        )
        .with_data_block_len(BLOCK_LEN as u64),
    )
}

/// Write `data` with indexes inside the file: trigram and full-text on `text`, IVF on `vec`.
async fn write_indexed_file(session: &VortexSession, data: &Data) -> VortexResult<VortexFile> {
    let trigram = IndexConfig::new(
        TrigramIndex::new_ref(),
        TrigramIndex.serialize_options(&TrigramOptions {
            block_len: BLOCK_LEN as u64,
        }),
    );
    let strategy = WriteStrategyBuilder::from_session(session)
        .with_row_block_size(BLOCK_LEN)
        .with_field_writer(
            FieldPath::from_name("text"),
            indexed(vec![
                trigram,
                IndexConfig::with_defaults(InvertedIndex::new_ref()),
            ]),
        )
        .with_field_writer(
            FieldPath::from_name("vec"),
            indexed(vec![IndexConfig::with_defaults(IvfIndex::new_ref())]),
        )
        .build();
    let mut bytes = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut bytes, data.array()?.to_array_stream())
        .await?;
    session.open_options().open_buffer(bytes.freeze())
}

/// Write `data` with no indexes.
async fn write_plain_file(session: &VortexSession, data: &Data) -> VortexResult<VortexFile> {
    let mut bytes = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut bytes, data.array()?.to_array_stream())
        .await?;
    session.open_options().open_buffer(bytes.freeze())
}

async fn scan_ids(
    file: &VortexFile,
    filter: Expression,
    selection: Selection,
) -> VortexResult<Vec<i32>> {
    let chunks = collect(
        ScanBuilder::new(file.session().clone(), file.scan_file())
            .with_some_filter(Some(filter.bind(file.dtype())?))
            .with_selection(selection)
            .with_projection(col("id").bind(file.dtype())?),
    )
    .await?;
    let mut ctx = file.session().create_execution_ctx();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let ids = ChunkedArray::try_new(chunks, dtype)?
        .into_array()
        .execute::<PrimitiveArray>(&mut ctx)?;
    Ok(ids.as_slice::<i32>().to_vec())
}

fn needle(text: &str) -> bool {
    text.contains("Needle")
}

#[rstest]
#[case::like(like(col("text"), lit("%Needle%")), &needle as &dyn Fn(&str) -> bool)]
#[case::ilike(ilike(col("text"), lit("%nEEdle%")), &needle)]
#[case::two_runs(like(col("text"), lit("%Needle _lph%")), &|t: &str| t.contains("Needle alph"))]
#[case::text_match(text_match(col("text"), "needle"), &needle)]
#[case::text_match_and(
    text_match(col("text"), "ECHO hotel"),
    &|t: &str| t.split(' ').any(|w| w == "echo") && t.split(' ').any(|w| w == "hotel"),
)]
#[tokio::test]
async fn in_file_filters_match_brute_force(
    #[case] filter: Expression,
    #[case] expected: &dyn Fn(&str) -> bool,
) -> VortexResult<()> {
    let session = session()?;
    let data = Data::generate(7, ROWS);
    let file = write_indexed_file(&session, &data).await?;
    assert_eq!(
        scan_ids(&file, filter, Selection::All).await?,
        data.rows_where(expected)
    );
    Ok(())
}

#[tokio::test]
async fn in_file_trigram_prunes_blocks() -> VortexResult<()> {
    let session = session()?;
    let data = Data::generate(7, ROWS);
    let file = write_indexed_file(&session, &data).await?;
    let index = open_in_file_index(TrigramIndex, &file, "text")?.expect("trigram index");
    let (exactness, locator) = index
        .locate(&like(root(), lit("%Needle%")))
        .await?
        .expect("claims LIKE");
    assert_eq!(exactness, IndexExactness::Superset);
    let RowLocator::Blocks { block_len, ids } = locator else {
        panic!("trigram index locates blocks");
    };
    assert_eq!(block_len, BLOCK_LEN as u64);
    let expected: Vec<u32> = NEEDLE_ROWS
        .iter()
        .map(|&row| (row / BLOCK_LEN) as u32)
        .collect();
    assert_eq!(ids.iter().collect::<Vec<_>>(), expected);

    // Too short to hold a trigram: no claim, the scan reads everything.
    assert!(index.locate(&like(root(), lit("%ec%"))).await?.is_none());
    Ok(())
}

#[tokio::test]
async fn in_file_bm25_ranks_rare_terms_first() -> VortexResult<()> {
    let session = session()?;
    let data = Data::generate(7, ROWS);
    let file = write_indexed_file(&session, &data).await?;
    let index = open_in_file_index(InvertedIndex, &file, "text")?.expect("inverted index");

    let hits = index.search_bm25("needle echo", 5).await?;
    assert_eq!(hits.len(), 5);
    // The rare term outweighs any count of the common one, so the needle rows rank first.
    let mut top: Vec<u64> = hits[..3].iter().map(|h| h.row).collect();
    top.sort_unstable();
    assert_eq!(top, NEEDLE_ROWS.map(|r| r as u64));
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(index.search_bm25("zulu", 5).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn in_file_ivf_finds_nearest_vectors() -> VortexResult<()> {
    let session = session()?;
    let data = Data::generate(7, ROWS);
    let file = write_indexed_file(&session, &data).await?;
    let index = open_in_file_index(IvfIndex, &file, "vec")?.expect("ivf index");
    let lists = index.options().lists as usize;
    assert!(lists > 1);

    let query = data.vectors[1234];
    let hits = index.search_vectors(&query, 10, lists).await?;
    assert_eq!(hits[0].row, 1234);
    assert!(hits[0].score > 0.99);

    // Probing every list leaves only quantization error, so recall is near perfect.
    let expected = data.nearest(&query, 10);
    let found = hits.iter().filter(|h| expected.contains(&h.row)).count();
    assert!(found >= 8, "recall {found}/10");
    Ok(())
}

#[tokio::test]
async fn standalone_indexes_cover_several_files() -> VortexResult<()> {
    let session = session()?;
    let parts = [Data::generate(1, 1000), Data::generate(2, 1200)];
    let mut files = Vec::new();
    for part in &parts {
        files.push(write_plain_file(&session, part).await?);
    }
    let text_dtype = DType::Utf8(Nullability::Nullable);

    // Trigram index over both files.
    let mut builder = IndexFileBuilder::new(
        TrigramIndex,
        TrigramOptions { block_len: 32 },
        "text",
        text_dtype.clone(),
        &session,
    )?;
    for (idx, file) in files.iter().enumerate() {
        builder.add_file(format!("part-{idx}.vortex"), file).await?;
    }
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let index_file = session
        .open_options()
        .include_metadata()
        .open_buffer(bytes.freeze())?;

    let info = IndexFileInfo::read(&index_file)?;
    assert_eq!(info.kind, "vortex.idx.trigram");
    assert_eq!(info.column, "text");
    assert_eq!(
        info.files,
        vec![
            CoveredFile::new("part-0.vortex", 1000),
            CoveredFile::new("part-1.vortex", 1200)
        ]
    );

    let trigram = IndexFile::open(TrigramIndex, &index_file)?;
    let (exactness, selections) = trigram
        .locate_files(&like(root(), lit("%Needle%")))
        .await?
        .expect("claims LIKE");
    assert_eq!(exactness, IndexExactness::Superset);
    // Both parts are shorter than 1500 rows, so each holds the needle only at row 100.
    assert_eq!(selections.len(), 2);
    for selection in selections {
        let file = &files[selection.file];
        let found = scan_ids(
            file,
            like(col("text"), lit("%Needle%")),
            selection.selection,
        )
        .await?;
        assert_eq!(found, parts[selection.file].rows_where(needle));
    }

    // A pattern no row holds rules out every file without opening any.
    let (_, none) = trigram
        .locate_files(&like(root(), lit("%zzzz%")))
        .await?
        .expect("claims LIKE");
    assert!(none.is_empty());
    Ok(())
}

#[tokio::test]
async fn standalone_search_maps_hits_to_files() -> VortexResult<()> {
    let session = session()?;
    let parts = [Data::generate(1, 1000), Data::generate(2, 1200)];
    let mut files = Vec::new();
    for part in &parts {
        files.push(write_plain_file(&session, part).await?);
    }

    // Full-text: exact filter selections and BM25 hits across files.
    let mut builder = IndexFileBuilder::new(
        InvertedIndex,
        (),
        "text",
        DType::Utf8(Nullability::Nullable),
        &session,
    )?;
    for (idx, file) in files.iter().enumerate() {
        builder.add_file(format!("part-{idx}.vortex"), file).await?;
    }
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let inverted = IndexFile::open(
        InvertedIndex,
        &session
            .open_options()
            .include_metadata()
            .open_buffer(bytes.freeze())?,
    )?;

    let (exactness, selections) = inverted
        .locate_files(&text_match(root(), "needle"))
        .await?
        .expect("claims text_match");
    assert_eq!(exactness, IndexExactness::Exact);
    let selected: Vec<(usize, Vec<u64>)> = selections
        .iter()
        .map(|s| match &s.selection {
            Selection::IncludeRoaring(rows) => (s.file, rows.iter().collect()),
            other => panic!("unexpected selection {other:?}"),
        })
        .collect();
    assert_eq!(selected, vec![(0, vec![100]), (1, vec![100])]);

    let hits = inverted.hits_by_file(&inverted.rows().search_bm25("needle", 2).await?)?;
    let mut located: Vec<(usize, u64)> = hits.iter().map(|h| (h.file, h.row)).collect();
    located.sort_unstable();
    assert_eq!(located, vec![(0, 100), (1, 100)]);

    // Vectors: a stored vector of the second file is its own nearest neighbour.
    let mut builder = IndexFileBuilder::new(
        IvfIndex,
        IvfOptions { lists: 8, dim: 0 },
        "vec",
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            DIM as u32,
            Nullability::NonNullable,
        ),
        &session,
    )?;
    for (idx, file) in files.iter().enumerate() {
        builder.add_file(format!("part-{idx}.vortex"), file).await?;
    }
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let ivf = IndexFile::open(
        IvfIndex,
        &session
            .open_options()
            .include_metadata()
            .open_buffer(bytes.freeze())?,
    )?;
    let query = parts[1].vectors[777];
    let hits = ivf.hits_by_file(&ivf.rows().search_vectors(&query, 3, 8).await?)?;
    assert_eq!((hits[0].file, hits[0].row), (1, 777));
    Ok(())
}

/// Every stored vector is its own nearest neighbour when every list is probed, so a lookup that
/// read the wrong lists, or a build that numbered rows out of order, shows up as a wrong row.
#[tokio::test]
async fn standalone_ivf_reads_probed_lists_by_position() -> VortexResult<()> {
    let session = session()?;
    let data = Data::generate(3, ROWS);
    let file = write_plain_file(&session, &data).await?;
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
        let hits = ivf
            .rows()
            .search_vectors(&data.vectors[row], 1, lists)
            .await?;
        assert_eq!(hits[0].row, row as u64);
    }
    Ok(())
}

/// Files of one table may differ in a column's nullability; a different type is still refused.
#[tokio::test]
async fn standalone_index_accepts_files_that_differ_in_nullability() -> VortexResult<()> {
    let session = session()?;
    let mut builder = IndexFileBuilder::new(
        InvertedIndex,
        (),
        "text",
        DType::Utf8(Nullability::NonNullable),
        &session,
    )?;
    builder.push_file(
        CoveredFile::new("a.vortex", 2),
        [VarBinViewArray::from_iter_str(["red fox", "blue jay"]).into_array()],
    )?;
    builder.push_file(
        CoveredFile::new("b.vortex", 2),
        [VarBinViewArray::from_iter_nullable_str([Some("red kite"), None]).into_array()],
    )?;
    assert!(
        builder
            .push_file(
                CoveredFile::new("c.vortex", 1),
                [VarBinViewArray::from_iter_bin([b"red"]).into_array()],
            )
            .is_err()
    );

    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let index = IndexFile::open(
        InvertedIndex,
        &session
            .open_options()
            .include_metadata()
            .open_buffer(bytes.freeze())?,
    )?;
    let (_, selections) = index
        .locate_files(&text_match(root(), "red"))
        .await?
        .expect("claims text_match");
    let files: Vec<usize> = selections.iter().map(|s| s.file).collect();
    assert_eq!(files, vec![0, 1]);
    Ok(())
}

/// Each file starts on a block boundary of the shared index, so a match in one file never selects
/// rows of its neighbour, even when the files' lengths are not multiples of the block length.
#[tokio::test]
async fn trigram_blocks_do_not_straddle_files() -> VortexResult<()> {
    let session = session()?;
    let rows = 10_000;
    let file = |needle_row: Option<usize>| {
        VarBinViewArray::from_iter_str((0..rows).map(|row| {
            if Some(row) == needle_row {
                "needle".to_string()
            } else {
                format!("hay {}", row % 7)
            }
        }))
        .into_array()
    };
    let mut builder = IndexFileBuilder::new(
        TrigramIndex,
        TrigramOptions { block_len: 8192 },
        "text",
        DType::Utf8(Nullability::NonNullable),
        &session,
    )?;
    builder.push_file(CoveredFile::new("a.vortex", rows as u64), [file(None)])?;
    builder.push_file(CoveredFile::new("b.vortex", rows as u64), [file(Some(100))])?;
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let index = IndexFile::open(
        TrigramIndex,
        &session
            .open_options()
            .include_metadata()
            .open_buffer(bytes.freeze())?,
    )?;

    let (_, selections) = index
        .locate_files(&like(root(), lit("%needle%")))
        .await?
        .expect("claims like");
    let selected: Vec<(usize, Vec<u64>)> = selections
        .iter()
        .map(|s| match &s.selection {
            Selection::IncludeRoaring(rows) => (s.file, vec![rows.min().unwrap_or(0), rows.len()]),
            other => panic!("unexpected selection {other:?}"),
        })
        .collect();
    // Only file b's first block: its rows 0..8192.
    assert_eq!(selected, vec![(1, vec![0, 8192])]);
    Ok(())
}

#[tokio::test]
async fn opening_with_the_wrong_kind_fails() -> VortexResult<()> {
    let session = session()?;
    let file = write_plain_file(&session, &Data::generate(3, 100)).await?;
    let mut builder = IndexFileBuilder::new(
        InvertedIndex,
        (),
        "text",
        DType::Utf8(Nullability::Nullable),
        &session,
    )?;
    builder.add_file("part.vortex", &file).await?;
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let index_file = session
        .open_options()
        .include_metadata()
        .open_buffer(bytes.freeze())?;
    assert!(IndexFile::open(TrigramIndex, &index_file).is_err());
    assert!(IndexFileInfo::read(&file).is_err());
    Ok(())
}

#[tokio::test]
async fn covered_files_keep_catalog_identity() -> VortexResult<()> {
    let session = session()?;
    let file = write_plain_file(&session, &Data::generate(4, 50)).await?;
    let covered = CoveredFile {
        uri: "data/part-0.vortex".to_string(),
        row_count: 50,
        fingerprint: 0x1234_5678_9abc_def0u64.to_le_bytes().to_vec(),
    };
    let mut builder = IndexFileBuilder::new(
        TrigramIndex,
        TrigramOptions::default(),
        "text",
        DType::Utf8(Nullability::Nullable),
        &session,
    )?;
    builder.add_covered_file(covered.clone(), &file).await?;
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let index_file = session
        .open_options()
        .include_metadata()
        .open_buffer(bytes.freeze())?;

    let info = IndexFileInfo::read(&index_file)?;
    assert_eq!(info.files, vec![covered.clone()]);
    assert!(covered.matches(50, &covered.fingerprint));
    assert!(!covered.matches(50, &[0; 8]));
    assert!(!covered.matches(51, &covered.fingerprint));
    Ok(())
}
