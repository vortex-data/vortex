// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared index machinery, exercised through a test-only kind: a min/max range per block of
//! rows over an `i32` column.

#![allow(clippy::unwrap_used, reason = "test code")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::LazyLock;

use futures::TryStreamExt;
use roaring::RoaringBitmap;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt_eq;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::v2::ScanDriver;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::indexed::IndexBuilder;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexId;
use vortex_layout::layouts::indexed::IndexQueryPlan;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::CoveredFile;
use crate::FileMatch;
use crate::FileRows;
use crate::IndexFile;
use crate::IndexFileBuilder;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| VortexSession::default().with_tokio());

const BLOCK_LEN: u64 = 64;

fn index_fields() -> StructFields {
    StructFields::new(
        FieldNames::from(["block", "min", "max"]),
        vec![
            DType::Primitive(PType::U32, Nullability::NonNullable),
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Primitive(PType::I32, Nullability::NonNullable),
        ],
    )
}

/// Each block of `block_len` rows keeps the min and max of its non-null values, and answers
/// `value == x` with the blocks whose range holds `x`. A build that saw no values declines.
#[derive(Debug)]
struct BlockRangeIndex;

/// [`BlockRangeIndex`] under another id, to open an index file as the wrong kind.
#[derive(Debug)]
struct OtherIndex;

/// One chunk of index rows: `(block, min, max)`, ascending by block.
struct Blocks(Vec<(u32, i32, i32)>);

struct Builder {
    block_len: u64,
    blocks: BTreeMap<u32, (i32, i32)>,
}

impl IndexBuilder for Builder {
    type Options = u64;
    type Chunk = Blocks;

    fn push(
        &mut self,
        chunk: &ArrayRef,
        row_offset: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let valid = chunk.validity()?.execute_mask(chunk.len(), ctx)?;
        let values = chunk.clone().execute::<PrimitiveArray>(ctx)?;
        for (idx, &value) in values.as_slice::<i32>().iter().enumerate() {
            if !valid.value(idx) {
                continue;
            }
            let block = u32::try_from((row_offset + idx as u64) / self.block_len)?;
            let range = self.blocks.entry(block).or_insert((value, value));
            range.0 = range.0.min(value);
            range.1 = range.1.max(value);
        }
        Ok(())
    }

    fn finish(self) -> VortexResult<Option<(Vec<Blocks>, u64)>> {
        if self.blocks.is_empty() {
            return Ok(None);
        }
        let rows: Vec<_> = self
            .blocks
            .into_iter()
            .map(|(b, (lo, hi))| (b, lo, hi))
            .collect();
        // Several chunks, so lookups read only some of them.
        let chunks = rows.chunks(4).map(|rows| Blocks(rows.to_vec())).collect();
        Ok(Some((chunks, self.block_len)))
    }

    fn buffered_bytes(&self) -> u64 {
        (self.blocks.len() * 12) as u64
    }
}

macro_rules! block_range_vtable {
    ($kind:ty, $id:literal) => {
        impl IndexVTable for $kind {
            type Options = u64;
            type Builder = Builder;
            type Query = i32;
            type Chunk = Blocks;

            fn id(&self) -> IndexId {
                static ID: CachedId = CachedId::new($id);
                *ID
            }

            fn serialize_options(&self, options: &u64) -> Vec<u8> {
                options.to_le_bytes().to_vec()
            }

            fn deserialize_options(&self, options: &[u8]) -> VortexResult<u64> {
                Ok(u64::from_le_bytes(
                    options.try_into().map_err(|_| vortex_err!("bad options"))?,
                ))
            }

            fn index_dtype(&self, dtype: &DType, _options: &u64) -> Option<DType> {
                matches!(dtype, DType::Primitive(PType::I32, _))
                    .then(|| DType::Struct(index_fields(), Nullability::NonNullable))
            }

            fn builder(
                &self,
                _dtype: &DType,
                options: &u64,
                _data_block_len: Option<u64>,
                _session: &VortexSession,
            ) -> VortexResult<Builder> {
                Ok(Builder {
                    block_len: *options,
                    blocks: BTreeMap::new(),
                })
            }

            fn plan(
                &self,
                expr: &BoundExpression,
                _dtype: &DType,
                index_dtype: &DType,
                _options: &u64,
            ) -> VortexResult<Option<IndexQueryPlan<i32>>> {
                plan_eq(expr, index_dtype)
            }

            fn encode(
                &self,
                chunk: &Blocks,
                _options: &u64,
                _ctx: &mut ExecutionCtx,
            ) -> VortexResult<ArrayRef> {
                encode_blocks(chunk)
            }

            fn decode(
                &self,
                chunk: ArrayRef,
                _options: &u64,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Blocks> {
                decode_blocks(chunk, ctx)
            }

            fn resolve(
                &self,
                query: &i32,
                chunks: &[Arc<Blocks>],
                _data_row_count: u64,
                options: &u64,
                _ctx: &mut ExecutionCtx,
            ) -> VortexResult<RowLocator> {
                let ids = chunks
                    .iter()
                    .flat_map(|chunk| chunk.0.iter())
                    .filter(|(_, lo, hi)| lo <= query && query <= hi)
                    .map(|(block, ..)| *block)
                    .collect::<RoaringBitmap>();
                Ok(RowLocator::Blocks {
                    block_len: *options,
                    ids,
                })
            }
        }
    };
}

block_range_vtable!(BlockRangeIndex, "test.idx.block_range");
block_range_vtable!(OtherIndex, "test.idx.other");

/// Claims only `root() == <i32 literal>`.
fn plan_eq(
    expr: &BoundExpression,
    index_dtype: &DType,
) -> VortexResult<Option<IndexQueryPlan<i32>>> {
    if !expr.is::<Binary>() || *expr.as_::<Binary>() != Operator::Eq {
        return Ok(None);
    }
    if !expr.child(0).is_root() || !expr.child(1).is::<Literal>() {
        return Ok(None);
    }
    let Some(value) = expr
        .child(1)
        .as_::<Literal>()
        .as_primitive()
        .typed_value::<i32>()
    else {
        return Ok(None);
    };
    Ok(Some(IndexQueryPlan {
        exactness: IndexExactness::Superset,
        filter: and(lt_eq(col("min"), lit(value)), gt_eq(col("max"), lit(value)))
            .bind(index_dtype)?,
        query: value,
    }))
}

fn encode_blocks(chunk: &Blocks) -> VortexResult<ArrayRef> {
    let blocks = PrimitiveArray::from_iter(chunk.0.iter().map(|r| r.0)).into_array();
    let mins = PrimitiveArray::from_iter(chunk.0.iter().map(|r| r.1)).into_array();
    let maxs = PrimitiveArray::from_iter(chunk.0.iter().map(|r| r.2)).into_array();
    Ok(StructArray::try_new_with_dtype(
        vec![blocks, mins, maxs],
        index_fields(),
        chunk.0.len(),
        Validity::NonNullable,
    )?
    .into_array())
}

fn decode_blocks(chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Blocks> {
    let rows = chunk.execute::<StructArray>(ctx)?;
    let field = |name: &str, ctx: &mut ExecutionCtx| -> VortexResult<PrimitiveArray> {
        rows.unmasked_field_by_name(name)?
            .clone()
            .execute::<PrimitiveArray>(ctx)
    };
    let (blocks, mins, maxs) = (field("block", ctx)?, field("min", ctx)?, field("max", ctx)?);
    Ok(Blocks(
        blocks
            .as_slice::<u32>()
            .iter()
            .zip(mins.as_slice::<i32>())
            .zip(maxs.as_slice::<i32>())
            .map(|((&b, &lo), &hi)| (b, lo, hi))
            .collect(),
    ))
}

/// Run a scan to its end in row order, as a caller's driver would.
async fn scan(builder: ScanBuilder) -> VortexResult<Vec<ArrayRef>> {
    ScanDriver::new(SESSION.handle(), 1, true)
        .run(builder)
        .try_collect()
        .await
}

/// Look `expr` up in `index`, running its scans as a caller's driver would.
async fn lookup(
    index: &IndexFile<BlockRangeIndex>,
    expr: &Expression,
) -> VortexResult<Option<Vec<FileMatch>>> {
    let Some(mut lookup) = index.lookup(expr)? else {
        return Ok(None);
    };
    let mut batches = Vec::new();
    for builder in lookup.scans() {
        batches.push(scan(builder).await?);
    }
    lookup.finish(batches).map(Some)
}

/// A data file of one nullable `i32` column, `value`.
async fn write_data(values: Vec<Option<i32>>) -> VortexResult<VortexFile> {
    let len = values.len();
    let array = StructArray::try_new(
        FieldNames::from(["value"]),
        vec![PrimitiveArray::from_option_iter(values).into_array()],
        len,
        Validity::NonNullable,
    )?
    .into_array();
    let mut bytes = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut bytes, array.to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(bytes.freeze())
}

/// File `k` holds `k * 10_000 + row` at every row, so each value is in exactly one file and row.
async fn data_files(row_counts: &[u64]) -> VortexResult<Vec<VortexFile>> {
    let mut files = Vec::new();
    for (k, &rows) in row_counts.iter().enumerate() {
        let base = i32::try_from(k)? * 10_000;
        files.push(
            write_data(
                (0..i32::try_from(rows)?)
                    .map(|row| Some(base + row))
                    .collect(),
            )
            .await?,
        );
    }
    Ok(files)
}

async fn build_index(vtable_rows: FileRows, files: &[VortexFile]) -> VortexResult<VortexFile> {
    let mut builder = IndexFileBuilder::new(
        BlockRangeIndex,
        BLOCK_LEN,
        "value",
        DType::Primitive(PType::I32, Nullability::Nullable),
        vtable_rows,
        &SESSION,
    )?;
    for (idx, file) in files.iter().enumerate() {
        let chunks = scan(builder.column_scan(file)?).await?;
        builder.push_file(
            CoveredFile::new(format!("part-{idx}.vortex"), file.row_count()),
            chunks,
        )?;
    }
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    SESSION
        .open_options()
        .include_metadata()
        .open_buffer(bytes.freeze())
}

fn block_of(row: u64, row_count: u64) -> RoaringBitmap {
    let start = row / BLOCK_LEN * BLOCK_LEN;
    let mut rows = RoaringBitmap::new();
    rows.insert_range(
        u32::try_from(start).unwrap()..u32::try_from((start + BLOCK_LEN).min(row_count)).unwrap(),
    );
    rows
}

fn included(selection: &Selection) -> Vec<u64> {
    match selection {
        Selection::IncludeRoaring(rows) => rows.iter().collect(),
        other => panic!("expected included rows, got {other:?}"),
    }
}

/// Both layouts answer the same: each value is found in its own file's block only, even where a
/// file's row count is not a multiple of the block length, so a shared block could straddle two
/// files.
#[rstest]
#[case::shared(FileRows::Shared)]
#[case::per_file(FileRows::PerFile)]
#[tokio::test]
async fn lookups_select_rows_local_to_each_file(#[case] file_rows: FileRows) -> VortexResult<()> {
    let row_counts = [1000, 130, 777];
    let files = data_files(&row_counts).await?;
    let index = IndexFile::open(BlockRangeIndex, &build_index(file_rows, &files).await?)?;
    assert_eq!(index.file_rows(), file_rows);
    assert_eq!(index.files().len(), 3);

    for (file, row) in [(0usize, 5u64), (1, 129), (2, 0), (2, 776)] {
        let value = i32::try_from(file)? * 10_000 + i32::try_from(row)?;
        let matches = lookup(&index, &eq(root(), lit(value)))
            .await?
            .expect("claims equality");
        assert_eq!(matches.len(), 1, "value {value}");
        assert_eq!(matches[0].file, file);
        assert_eq!(matches[0].exactness, IndexExactness::Superset);
        let expected: Vec<u64> = block_of(row, row_counts[file])
            .iter()
            .map(u64::from)
            .collect();
        assert_eq!(included(&matches[0].selection), expected);
    }

    let none = lookup(&index, &eq(root(), lit(-1i32)))
        .await?
        .expect("claims equality");
    assert!(none.is_empty());
    assert!(lookup(&index, &gt(root(), lit(3i32))).await?.is_none());
    Ok(())
}

/// A per-file build that declines makes no claim on its file, which is then selected whole.
#[tokio::test]
async fn per_file_declined_files_are_selected_whole() -> VortexResult<()> {
    let files = vec![
        data_files(&[200]).await?.remove(0),
        write_data(vec![None; 50]).await?,
    ];
    let index = IndexFile::open(
        BlockRangeIndex,
        &build_index(FileRows::PerFile, &files).await?,
    )?;
    let matches = lookup(&index, &eq(root(), lit(70i32)))
        .await?
        .expect("claims equality");
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].file, 0);
    assert_eq!(
        included(&matches[0].selection),
        (64..128).collect::<Vec<_>>()
    );
    assert_eq!(matches[1].file, 1);
    assert!(matches!(matches[1].selection, Selection::All));
    Ok(())
}

#[rstest]
#[case::shared(FileRows::Shared)]
#[case::per_file(FileRows::PerFile)]
#[tokio::test]
async fn index_files_keep_their_covered_files(#[case] file_rows: FileRows) -> VortexResult<()> {
    let data = data_files(&[50]).await?;
    let covered = CoveredFile {
        uri: "data/part-0.vortex".to_string(),
        row_count: 50,
        fingerprint: 0x1234_5678_9abc_def0u64.to_le_bytes().to_vec(),
    };
    let mut builder = IndexFileBuilder::new(
        BlockRangeIndex,
        BLOCK_LEN,
        "value",
        DType::Primitive(PType::I32, Nullability::Nullable),
        file_rows,
        &SESSION,
    )?;
    let chunks = scan(builder.column_scan(&data[0])?).await?;
    builder.push_file(covered.clone(), chunks)?;
    let mut bytes = ByteBufferMut::empty();
    builder.write(&mut bytes).await?;
    let index_file = SESSION
        .open_options()
        .include_metadata()
        .open_buffer(bytes.freeze())?;

    assert_eq!(
        IndexFile::open(BlockRangeIndex, &index_file)?.files(),
        [covered]
    );
    assert!(IndexFile::open(OtherIndex, &index_file).is_err());
    assert!(IndexFile::open(BlockRangeIndex, &data[0]).is_err());
    Ok(())
}
