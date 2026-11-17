// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Index files: an index stored outside the data, covering an explicit list of data files.
//!
//! An index file is an ordinary Vortex file. Its rows are the index content, encoded by the
//! index kind, and a metadata segment under [`INDEX_FILE_METADATA_KEY`] records the kind, its
//! options, the indexed column, and the covered files. A file's position in that list is its id.
//! Open index files with `include_metadata()` so the description is loaded.
//!
//! How the covered files' rows are indexed is the builder's [`FileRows`]:
//!
//! - [`FileRows::Shared`] builds one index over a combined row space. Each file starts at the
//!   next multiple of [`FILE_ROW_ALIGN`], so block-granular indexes never put two files' rows in
//!   one block. A lookup is one scan of the index file, and the whole space is capped at
//!   `u32::MAX` rows.
//! - [`FileRows::PerFile`] builds one index per file, in the file's own row space, and stores
//!   them one after another. A lookup is one scan per file, but nothing is padded.
//!
//! Nothing here reads on its own. A [`Lookup`] hands back the scans of the index file to run,
//! and turns their batches into the rows of each data file that may match.

use std::ops::Range;
use std::sync::Arc;

use futures::stream;
use prost::Message;
use roaring::RoaringTreemap;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_file::WriteSummary;
use vortex_io::VortexWrite;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::indexed::BuiltIndex;
use vortex_layout::layouts::indexed::IndexBuilder;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::scan::v2::ScanCache;
use vortex_layout::scan::v2::ScanFile;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

/// Key of the metadata segment describing an index file.
pub const INDEX_FILE_METADATA_KEY: &str = "vortex.index";

const INDEX_FILE_VERSION: u32 = 3;

/// Where each covered file starts in a [`FileRows::Shared`] row space: a multiple of this many
/// rows, which is a multiple of every power-of-two block length up to it.
pub const FILE_ROW_ALIGN: u64 = 1 << 16;

/// Index rows per chunk of the index file by default. Every lookup reads whole chunks, so they
/// are kept much smaller than data chunks.
pub const DEFAULT_INDEX_CHUNK_ROWS: usize = 256;

/// How an index file lays out the rows of the files it covers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FileRows {
    /// One index over a combined row space, each file starting on a [`FILE_ROW_ALIGN`] boundary.
    #[default]
    Shared,
    /// One index per file, each in the file's own row space.
    PerFile,
}

#[derive(Clone, PartialEq, Message)]
struct IndexFileMetadata {
    #[prost(uint32, tag = "1")]
    version: u32,
    #[prost(string, tag = "2")]
    kind: String,
    /// The index's options; for a per-file index, the options each file's build started from.
    #[prost(bytes = "vec", tag = "3")]
    options: Vec<u8>,
    #[prost(string, tag = "4")]
    column: String,
    #[prost(message, optional, tag = "5")]
    source_dtype: Option<pb::DType>,
    #[prost(message, repeated, tag = "6")]
    files: Vec<CoveredFileProto>,
    /// 0 for [`FileRows::Shared`], 1 for [`FileRows::PerFile`].
    #[prost(int32, tag = "7")]
    file_rows: i32,
}

#[derive(Clone, PartialEq, Message)]
struct CoveredFileProto {
    #[prost(string, tag = "1")]
    uri: String,
    #[prost(uint64, tag = "2")]
    row_count: u64,
    #[prost(bytes = "vec", tag = "3")]
    fingerprint: Vec<u8>,
    /// Per-file only: the exclusive end of this file's rows in the index file.
    #[prost(uint64, tag = "4")]
    index_end: u64,
    /// Per-file only: the options this file's build finished with, unless it declined.
    #[prost(bytes = "vec", optional, tag = "5")]
    options: Option<Vec<u8>>,
}

/// A data file an index covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoveredFile {
    /// Where the catalog finds the data file.
    pub uri: String,
    /// Rows in the data file when the index was built.
    pub row_count: u64,
    /// Optional catalog-defined identity, such as a hash of the data file's footer, so a reader
    /// can tell that the file was replaced since the index was built. Empty when not set.
    pub fingerprint: Vec<u8>,
}

impl CoveredFile {
    /// A covered file without a fingerprint.
    pub fn new(uri: impl Into<String>, row_count: u64) -> Self {
        Self {
            uri: uri.into(),
            row_count,
            fingerprint: Vec::new(),
        }
    }
}

/// Builds an index file over a column of one or more data files.
///
/// Add the files in order with [`push_file`](Self::push_file), reading each one's column with
/// the scan [`column_scan`](Self::column_scan) returns.
pub struct IndexFileBuilder<V: IndexVTable> {
    vtable: V,
    options: V::Options,
    column: String,
    source_dtype: DType,
    build: Build<V::Builder>,
    files: Vec<CoveredFile>,
    session: VortexSession,
    strategy: Option<Arc<dyn LayoutStrategy>>,
}

enum Build<B: IndexBuilder> {
    Shared {
        builder: B,
        /// The end of the shared row space so far.
        rows: u64,
    },
    /// Each file's finished build, `None` where it declined.
    PerFile(Vec<Option<BuiltIndex<B>>>),
}

impl<V: IndexVTable> IndexFileBuilder<V> {
    /// Start an index of kind `vtable` over `column`, whose values have `source_dtype`.
    pub fn new(
        vtable: V,
        options: V::Options,
        column: impl Into<String>,
        source_dtype: DType,
        file_rows: FileRows,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            vtable.index_dtype(&source_dtype, &options).is_some(),
            "Index {} cannot index {source_dtype}",
            vtable.id()
        );
        let build = match file_rows {
            FileRows::Shared => Build::Shared {
                builder: vtable.builder(&source_dtype, &options, None, session)?,
                rows: 0,
            },
            FileRows::PerFile => Build::PerFile(Vec::new()),
        };
        Ok(Self {
            vtable,
            options,
            column: column.into(),
            source_dtype,
            build,
            files: Vec::new(),
            session: session.clone(),
            strategy: None,
        })
    }

    /// Write the index rows with `strategy` instead of the default.
    pub fn with_strategy(mut self, strategy: Arc<dyn LayoutStrategy>) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// The scan of `file`'s indexed column, for the caller to run and pass to
    /// [`push_file`](Self::push_file).
    pub fn column_scan(&self, file: &VortexFile) -> VortexResult<ScanBuilder> {
        let projection = col(self.column.as_str()).bind(file.dtype())?;
        Ok(ScanBuilder::new(file.session().clone(), file.scan_file()).with_projection(projection))
    }

    /// Add the next covered file, from its indexed column's values in row order.
    pub fn push_file(
        &mut self,
        file: CoveredFile,
        chunks: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<()> {
        let mut ctx = self.session.create_execution_ctx();
        let source_dtype = &self.source_dtype;
        let push = |builder: &mut V::Builder, start: u64| -> VortexResult<u64> {
            let mut row = start;
            for chunk in chunks {
                // Files of one table may differ in the column's nullability.
                vortex_ensure!(
                    chunk.dtype().eq_ignore_nullability(source_dtype),
                    "Chunk dtype {} does not match the indexed column's {source_dtype}",
                    chunk.dtype()
                );
                builder.push(&chunk, row, &mut ctx)?;
                row += chunk.len() as u64;
            }
            vortex_ensure_eq!(
                row - start,
                file.row_count,
                "Pushed rows do not match the covered file's row count"
            );
            u32::try_from(row)
                .map_err(|_| vortex_err!("An index covers at most u32::MAX rows, got {row}"))?;
            Ok(row)
        };
        match &mut self.build {
            Build::Shared { builder, rows } => {
                *rows = push(builder, rows.next_multiple_of(FILE_ROW_ALIGN))?;
            }
            Build::PerFile(built) => {
                let mut builder =
                    self.vtable
                        .builder(&self.source_dtype, &self.options, None, &self.session)?;
                push(&mut builder, 0)?;
                built.push(builder.finish()?);
            }
        }
        self.files.push(file);
        Ok(())
    }

    /// Finish the index and write it as a Vortex file.
    pub async fn write<W: VortexWrite + Unpin>(self, write: W) -> VortexResult<WriteSummary> {
        let vtable = &self.vtable;
        let initial_options = vtable.serialize_options(&self.options);
        let index_dtype = vtable
            .index_dtype(&self.source_dtype, &self.options)
            .ok_or_else(|| vortex_err!("Index {} cannot index the column", vtable.id()))?;
        let mut ctx = self.session.create_execution_ctx();
        let mut encode = |built: BuiltIndex<V::Builder>| -> VortexResult<(Vec<ArrayRef>, Vec<u8>)> {
            let (chunks, options) = built;
            vortex_ensure!(
                vtable.index_dtype(&self.source_dtype, &options).as_ref() == Some(&index_dtype),
                "A build changed the index's row dtype"
            );
            let arrays = chunks
                .iter()
                .map(|chunk| vtable.encode(chunk, &options, &mut ctx))
                .collect::<VortexResult<_>>()?;
            Ok((arrays, vtable.serialize_options(&options)))
        };

        let mut arrays = Vec::new();
        let mut files: Vec<CoveredFileProto> = self
            .files
            .into_iter()
            .map(|f| CoveredFileProto {
                uri: f.uri,
                row_count: f.row_count,
                fingerprint: f.fingerprint,
                index_end: 0,
                options: None,
            })
            .collect();
        let (file_rows, options) = match self.build {
            Build::Shared { builder, .. } => match builder.finish()? {
                Some(built) => {
                    let (encoded, options) = encode(built)?;
                    arrays = encoded;
                    (0, options)
                }
                // A declined build writes an index with no rows: every lookup then misses.
                None => (0, initial_options),
            },
            Build::PerFile(built) => {
                let mut end = 0u64;
                for (file, built) in files.iter_mut().zip(built) {
                    if let Some(built) = built {
                        let (encoded, options) = encode(built)?;
                        end += encoded.iter().map(|a| a.len() as u64).sum::<u64>();
                        arrays.extend(encoded);
                        file.options = Some(options);
                    }
                    file.index_end = end;
                }
                (1, initial_options)
            }
        };

        let metadata = IndexFileMetadata {
            version: INDEX_FILE_VERSION,
            kind: vtable.id().to_string(),
            options,
            column: self.column,
            source_dtype: Some((&self.source_dtype).try_into()?),
            files,
            file_rows,
        };
        let strategy = self.strategy.unwrap_or_else(|| {
            WriteStrategyBuilder::from_session(&self.session)
                .with_row_block_size(DEFAULT_INDEX_CHUNK_ROWS)
                .build()
        });
        self.session
            .write_options()
            .with_strategy(strategy)
            .with_metadata_segment(INDEX_FILE_METADATA_KEY, metadata.encode_to_vec())
            .write(
                write,
                ArrayStreamAdapter::new(index_dtype, stream::iter(arrays.into_iter().map(Ok))),
            )
            .await
    }
}

/// An opened index file.
pub struct IndexFile<V: IndexVTable> {
    vtable: Arc<V>,
    source_dtype: DType,
    index_dtype: DType,
    files: Vec<CoveredFile>,
    file_rows: FileRows,
    parts: Vec<Part<V::Options>>,
    /// Covered files whose per-file build declined, so the index knows nothing of them.
    declined: Vec<usize>,
    scan_file: ScanFile,
    session: VortexSession,
}

/// One index of the file: every covered file of a shared index, or one file of a per-file one.
struct Part<O> {
    options: Arc<O>,
    /// The index file rows holding this index.
    index_rows: Range<u64>,
    /// Rows of data this index covers.
    data_row_count: u64,
    /// The covered files it indexes, and where each one's rows are in its row space.
    files: Vec<(usize, Range<u64>)>,
}

impl<V: IndexVTable> IndexFile<V> {
    /// Open an index file written by [`IndexFileBuilder`] for kind `vtable`.
    pub fn open(vtable: V, file: &VortexFile) -> VortexResult<Self> {
        let Some(bytes) = file.metadata_segment(INDEX_FILE_METADATA_KEY) else {
            vortex_bail!(
                "No {INDEX_FILE_METADATA_KEY} metadata: not an index file, or not opened with `include_metadata()`"
            );
        };
        let metadata = IndexFileMetadata::decode(bytes.as_slice())
            .map_err(|err| vortex_err!("Invalid index file metadata: {err}"))?;
        vortex_ensure_eq!(
            metadata.version,
            INDEX_FILE_VERSION,
            "Unsupported index file version"
        );
        vortex_ensure!(
            metadata.kind == vtable.id().as_ref(),
            "Index file holds a {} index, not {}",
            metadata.kind,
            vtable.id()
        );
        let source_dtype = DType::from_proto(
            metadata
                .source_dtype
                .as_ref()
                .ok_or_else(|| vortex_err!("Index file metadata has no source dtype"))?,
            file.session(),
        )?;
        let index_dtype = vtable
            .index_dtype(
                &source_dtype,
                &vtable.deserialize_options(&metadata.options)?,
            )
            .ok_or_else(|| {
                vortex_err!("Index {} no longer indexes {source_dtype}", metadata.kind)
            })?;
        vortex_ensure_eq!(
            file.dtype(),
            &index_dtype,
            "Index file rows do not match the dtype the index kind declares"
        );

        let files: Vec<CoveredFile> = metadata
            .files
            .iter()
            .map(|f| CoveredFile {
                uri: f.uri.clone(),
                row_count: f.row_count,
                fingerprint: f.fingerprint.clone(),
            })
            .collect();
        let mut parts = Vec::new();
        let mut declined = Vec::new();
        let file_rows = match metadata.file_rows {
            0 => {
                let mut end = 0u64;
                let ranges = files
                    .iter()
                    .enumerate()
                    .map(|(idx, f)| {
                        let start = end.next_multiple_of(FILE_ROW_ALIGN);
                        end = start + f.row_count;
                        (idx, start..end)
                    })
                    .collect();
                parts.push(Part {
                    options: Arc::new(vtable.deserialize_options(&metadata.options)?),
                    index_rows: 0..file.row_count(),
                    data_row_count: end,
                    files: ranges,
                });
                FileRows::Shared
            }
            1 => {
                let mut start = 0;
                for (idx, f) in metadata.files.iter().enumerate() {
                    vortex_ensure!(
                        start <= f.index_end && f.index_end <= file.row_count(),
                        "Index rows of {} are out of order",
                        f.uri
                    );
                    match &f.options {
                        Some(options) => parts.push(Part {
                            options: Arc::new(vtable.deserialize_options(options)?),
                            index_rows: start..f.index_end,
                            data_row_count: f.row_count,
                            files: vec![(idx, 0..f.row_count)],
                        }),
                        None => declined.push(idx),
                    }
                    start = f.index_end;
                }
                FileRows::PerFile
            }
            other => vortex_bail!("Unknown index file row layout {other}"),
        };

        // Lookups lower the same layout every time, so keep the plans, but not the bytes.
        let scan_file = file.scan_file();
        let scan_file = ScanFile {
            cache: scan_file
                .cache
                .clone()
                .or_else(|| Some(ScanCache::with_retained_bytes(0))),
            ..scan_file
        };
        Ok(Self {
            vtable: Arc::new(vtable),
            source_dtype,
            index_dtype,
            files,
            file_rows,
            parts,
            declined,
            scan_file,
            session: file.session().clone(),
        })
    }

    /// The covered data files, in order. A file's position is its id.
    pub fn files(&self) -> &[CoveredFile] {
        &self.files
    }

    /// How the covered files' rows are laid out.
    pub fn file_rows(&self) -> FileRows {
        self.file_rows
    }

    /// Plan a lookup of `expr`, a predicate over the indexed column where `root()` is the
    /// value, or `None` when the index cannot answer it for any file.
    pub fn lookup(&self, expr: &Expression) -> VortexResult<Option<Lookup<V>>> {
        let bound = expr.bind(&self.source_dtype)?;
        let mut scans = Vec::new();
        let mut plans = Vec::new();
        let mut matches: Vec<FileMatch> = self
            .declined
            .iter()
            .map(|&file| FileMatch::whole(file))
            .collect();
        for part in &self.parts {
            match self
                .vtable
                .plan(&bound, &self.source_dtype, &self.index_dtype, &part.options)?
            {
                Some(plan) => {
                    scans.push(
                        ScanBuilder::new(self.session.clone(), self.scan_file.clone())
                            .with_row_range(part.index_rows.clone())
                            .with_some_filter(Some(plan.filter)),
                    );
                    plans.push((part, plan.exactness, plan.query));
                }
                None => matches.extend(part.files.iter().map(|&(file, _)| FileMatch::whole(file))),
            }
        }
        if plans.is_empty() {
            return Ok(None);
        }
        let plans = plans
            .into_iter()
            .map(|(part, exactness, query)| PlannedPart {
                options: Arc::clone(&part.options),
                data_row_count: part.data_row_count,
                files: part.files.clone(),
                exactness,
                query,
            })
            .collect();
        Ok(Some(Lookup {
            vtable: Arc::clone(&self.vtable),
            session: self.session.clone(),
            scans,
            plans,
            matches,
        }))
    }
}

/// The rows of one covered data file that may match a lookup.
#[derive(Clone, Debug)]
pub struct FileMatch {
    /// Position of the file in [`IndexFile::files`].
    pub file: usize,
    /// Rows local to the file. Pass this to the file's scan.
    pub selection: Selection,
    /// Whether the rows are exactly the matches, or a superset to recheck.
    pub exactness: IndexExactness,
}

impl FileMatch {
    /// A file the index makes no claim on: every row may match.
    fn whole(file: usize) -> Self {
        Self {
            file,
            selection: Selection::All,
            exactness: IndexExactness::Superset,
        }
    }
}

/// A planned lookup: the scans of the index file to run, then [`finish`](Self::finish).
pub struct Lookup<V: IndexVTable> {
    vtable: Arc<V>,
    session: VortexSession,
    scans: Vec<ScanBuilder>,
    plans: Vec<PlannedPart<V>>,
    /// Files answered without reading: those the index makes no claim on.
    matches: Vec<FileMatch>,
}

struct PlannedPart<V: IndexVTable> {
    options: Arc<V::Options>,
    data_row_count: u64,
    files: Vec<(usize, Range<u64>)>,
    exactness: IndexExactness,
    query: V::Query,
}

impl<V: IndexVTable> Lookup<V> {
    /// The scans of the index file this lookup reads: one for a shared index, one per indexed
    /// file otherwise. Run each, in row order.
    pub fn scans(&mut self) -> Vec<ScanBuilder> {
        std::mem::take(&mut self.scans)
    }

    /// Answer the lookup from the batches of each scan, in the order [`scans`](Self::scans)
    /// returned them. Files with no possible match are left out, so they need not be opened.
    pub fn finish(self, batches: Vec<Vec<ArrayRef>>) -> VortexResult<Vec<FileMatch>> {
        vortex_ensure_eq!(batches.len(), self.plans.len(), "One batch list per scan");
        let mut ctx = self.session.create_execution_ctx();
        let mut matches = self.matches;
        for (plan, batches) in self.plans.into_iter().zip(batches) {
            let chunks = batches
                .into_iter()
                .filter(|array| !array.is_empty())
                .map(|array| {
                    self.vtable
                        .decode(array, &plan.options, &mut ctx)
                        .map(Arc::new)
                })
                .collect::<VortexResult<Vec<_>>>()?;
            let locator = self.vtable.resolve(
                &plan.query,
                &chunks,
                plan.data_row_count,
                &plan.options,
                &mut ctx,
            )?;
            for (file, range) in plan.files {
                let rows = local_rows(&locator, range)?;
                if !rows.is_empty() {
                    matches.push(FileMatch {
                        file,
                        selection: Selection::IncludeRoaring(rows),
                        exactness: plan.exactness,
                    });
                }
            }
        }
        matches.sort_by_key(|m| m.file);
        Ok(matches)
    }
}

/// The rows of `locator` within `range`, shifted to start at zero.
fn local_rows(locator: &RowLocator, range: Range<u64>) -> VortexResult<RoaringTreemap> {
    let mut rows = RoaringTreemap::new();
    match locator {
        RowLocator::Rows(bitmap) => {
            for row in bitmap.range(u32::try_from(range.start)?..u32::try_from(range.end)?) {
                rows.insert(u64::from(row) - range.start);
            }
        }
        RowLocator::Blocks { block_len, ids } => {
            let first = u32::try_from(range.start / block_len)?;
            let last = u32::try_from(range.end.div_ceil(*block_len))?;
            for block in ids.range(first..last) {
                let start = (u64::from(block) * block_len).max(range.start);
                let end = ((u64::from(block) + 1) * block_len).min(range.end);
                if start < end {
                    rows.insert_range(start - range.start..end - range.start);
                }
            }
        }
    }
    Ok(rows)
}
