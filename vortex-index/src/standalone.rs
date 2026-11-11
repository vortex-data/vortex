// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Standalone index files: an index stored outside the data, covering one or more data files.
//!
//! An index file is an ordinary Vortex file. Its rows are the index content, written through the
//! default strategy so they are chunked, compressed and zone-mapped like any data, and a metadata
//! segment under [`INDEX_FILE_METADATA_KEY`] records the index kind, its options, the indexed
//! column, and the data files it covers. A catalog lists index files next to data files and
//! hands them to [`IndexFile::open`]; [`IndexFileInfo::read`] tells it what an index file is
//! without knowing its kind. Open index files with `include_metadata()` so the description is
//! loaded.
//!
//! The covered files are concatenated in the order they were added, and the index works in that
//! combined row space. [`IndexFile::locate_files`] and [`IndexFile::hits_by_file`] map results
//! back to each file's own rows, ready to pass to a scan as a row selection.

use std::sync::Arc;
use std::sync::OnceLock;

use futures::stream;
use prost::Message;
use roaring::RoaringTreemap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
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
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::indexed::IndexBuilder;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::rows::Hit;
use crate::rows::IndexRows;
use crate::rows::collect;

/// Key of the metadata segment describing a standalone index file.
pub const INDEX_FILE_METADATA_KEY: &str = "vortex.index";

const INDEX_FILE_VERSION: u32 = 1;

#[derive(Clone, PartialEq, Message)]
struct IndexFileMetadata {
    #[prost(uint32, tag = "1")]
    version: u32,
    #[prost(string, tag = "2")]
    kind: String,
    #[prost(bytes = "vec", tag = "3")]
    options: Vec<u8>,
    #[prost(string, tag = "4")]
    column: String,
    #[prost(message, optional, tag = "5")]
    source_dtype: Option<pb::DType>,
    #[prost(message, repeated, tag = "6")]
    files: Vec<CoveredFileProto>,
}

#[derive(Clone, PartialEq, Message)]
struct CoveredFileProto {
    #[prost(string, tag = "1")]
    uri: String,
    #[prost(uint64, tag = "2")]
    row_count: u64,
    #[prost(bytes = "vec", tag = "3")]
    fingerprint: Vec<u8>,
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

    /// Whether `file` still looks like the file this index was built over.
    pub fn matches(&self, row_count: u64, fingerprint: &[u8]) -> bool {
        self.row_count == row_count
            && (self.fingerprint.is_empty() || self.fingerprint == fingerprint)
    }
}

/// What an index file holds, read without knowing its kind.
#[derive(Clone, Debug)]
pub struct IndexFileInfo {
    /// The index kind's registry id, such as `vortex.idx.inverted`.
    pub kind: String,
    /// The indexed column of the covered data files.
    pub column: String,
    /// The dtype of the indexed column.
    pub source_dtype: DType,
    /// The covered data files, in row-space order.
    pub files: Vec<CoveredFile>,
    options: Vec<u8>,
}

impl IndexFileInfo {
    /// Read the description of an index file.
    pub fn read(file: &VortexFile) -> VortexResult<Self> {
        let Some(bytes) = file.metadata_segment(INDEX_FILE_METADATA_KEY) else {
            if file
                .footer()
                .metadata_segment(INDEX_FILE_METADATA_KEY)
                .is_some()
            {
                vortex_bail!(
                    "Index file metadata was not loaded: open it with `include_metadata()`"
                );
            }
            vortex_bail!("Not a Vortex index file: no {INDEX_FILE_METADATA_KEY} segment");
        };
        let metadata = IndexFileMetadata::decode(bytes.as_slice())
            .map_err(|err| vortex_err!("Invalid index file metadata: {err}"))?;
        vortex_ensure_eq!(
            metadata.version,
            INDEX_FILE_VERSION,
            "Unsupported index file version"
        );
        let source_dtype = DType::from_proto(
            metadata
                .source_dtype
                .as_ref()
                .ok_or_else(|| vortex_err!("Index file metadata has no source dtype"))?,
            file.session(),
        )?;
        Ok(Self {
            kind: metadata.kind,
            column: metadata.column,
            source_dtype,
            files: metadata
                .files
                .into_iter()
                .map(|f| CoveredFile {
                    uri: f.uri,
                    row_count: f.row_count,
                    fingerprint: f.fingerprint,
                })
                .collect(),
            options: metadata.options,
        })
    }
}

/// Builds a standalone index file over a column of one or more data files.
pub struct IndexFileBuilder<V: IndexVTable> {
    vtable: Arc<V>,
    initial_options: Vec<u8>,
    column: String,
    source_dtype: DType,
    builder: V::Builder,
    files: Vec<CoveredFile>,
    rows: u64,
    session: VortexSession,
    strategy: Option<Arc<dyn LayoutStrategy>>,
}

/// Index rows per chunk by default.
///
/// Every lookup reads whole chunks of the index, so they are kept much smaller than data chunks
/// (which coalesce to about 1 MiB): a probe for a handful of keys then reads a handful of small
/// segments, found through the zone map on the index's sorted key column.
pub const DEFAULT_INDEX_CHUNK_ROWS: usize = 256;

impl<V: IndexVTable> IndexFileBuilder<V> {
    /// Start an index of kind `vtable` over `column`, whose values have `source_dtype`.
    pub fn new(
        vtable: V,
        options: V::Options,
        column: impl Into<String>,
        source_dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            vtable.index_dtype(&source_dtype, &options).is_some(),
            "Index {} cannot index {source_dtype}",
            vtable.id()
        );
        let builder = vtable.builder(&source_dtype, &options, None, session)?;
        Ok(Self {
            initial_options: vtable.serialize_options(&options),
            vtable: Arc::new(vtable),
            column: column.into(),
            source_dtype,
            builder,
            files: Vec::new(),
            rows: 0,
            session: session.clone(),
            strategy: None,
        })
    }

    /// Write the index rows with `strategy` rather than the default small-chunk strategy.
    pub fn with_strategy(mut self, strategy: Arc<dyn LayoutStrategy>) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Index the column values of one data file, given as chunks in row order.
    pub fn push_file(
        &mut self,
        file: CoveredFile,
        chunks: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<()> {
        let mut ctx = self.session.create_execution_ctx();
        let start = self.rows;
        for chunk in chunks {
            vortex_ensure!(
                chunk.dtype() == &self.source_dtype,
                "Chunk dtype {} does not match the indexed column's {}",
                chunk.dtype(),
                self.source_dtype
            );
            self.builder.push(&chunk, self.rows, &mut ctx)?;
            self.rows += chunk.len() as u64;
        }
        vortex_ensure_eq!(
            self.rows - start,
            file.row_count,
            "Pushed rows do not match the covered file's row count"
        );
        u32::try_from(self.rows).map_err(|_| {
            vortex_err!(
                "An index file covers at most u32::MAX rows, got {}",
                self.rows
            )
        })?;
        self.files.push(file);
        Ok(())
    }

    /// Read the indexed column of a data file and index it, recording it by `uri` alone.
    pub async fn add_file(
        &mut self,
        uri: impl Into<String>,
        file: &VortexFile,
    ) -> VortexResult<()> {
        self.add_covered_file(CoveredFile::new(uri, file.row_count()), file)
            .await
    }

    /// Read the indexed column of a data file and index it, recording it as `covered`.
    ///
    /// Use this to bind the index to a catalog's own identity for the file, such as a path
    /// relative to the table and a hash of the file's footer as the fingerprint.
    pub async fn add_covered_file(
        &mut self,
        covered: CoveredFile,
        file: &VortexFile,
    ) -> VortexResult<()> {
        let projection = col(self.column.as_str()).bind(file.dtype())?;
        let chunks = collect(
            ScanBuilder::new(file.session().clone(), file.scan_file()).with_projection(projection),
        )
        .await?;
        self.push_file(covered, chunks)
    }

    /// Finish the index and write it as a Vortex file.
    pub async fn write<W: VortexWrite + Unpin>(self, write: W) -> VortexResult<WriteSummary> {
        let Self {
            vtable,
            initial_options,
            column,
            source_dtype,
            builder,
            files,
            session,
            strategy,
            ..
        } = self;

        let (chunks, options_bytes) = match builder.finish()? {
            Some((chunks, options)) => {
                let bytes = vtable.serialize_options(&options);
                (chunks, bytes)
            }
            // A declined build writes an index with no rows: every lookup then misses.
            None => (Vec::new(), initial_options),
        };
        let options = vtable.deserialize_options(&options_bytes)?;
        let index_dtype = vtable
            .index_dtype(&source_dtype, &options)
            .ok_or_else(|| vortex_err!("Index {} cannot index {source_dtype}", vtable.id()))?;

        let mut ctx = session.create_execution_ctx();
        // Hand the writer small slices, so each index chunk, and so each lookup's read, stays
        // small even when a kind emits its whole content as one chunk.
        let mut arrays = Vec::new();
        for chunk in &chunks {
            let encoded = vtable.encode(chunk, &options, &mut ctx)?;
            let len = encoded.len();
            for start in (0..len).step_by(DEFAULT_INDEX_CHUNK_ROWS) {
                let slice = encoded.slice(start..(start + DEFAULT_INDEX_CHUNK_ROWS).min(len))?;
                arrays.push(compact(slice, &mut ctx)?);
            }
        }

        let metadata = IndexFileMetadata {
            version: INDEX_FILE_VERSION,
            kind: vtable.id().to_string(),
            options: options_bytes,
            column,
            source_dtype: Some((&source_dtype).try_into()?),
            files: files
                .into_iter()
                .map(|f| CoveredFileProto {
                    uri: f.uri,
                    row_count: f.row_count,
                    fingerprint: f.fingerprint,
                })
                .collect(),
        };

        session
            .write_options()
            .with_strategy(
                strategy.unwrap_or_else(|| default_index_strategy(&session, &index_dtype)),
            )
            .with_metadata_segment(INDEX_FILE_METADATA_KEY, metadata.encode_to_vec())
            .write(
                write,
                ArrayStreamAdapter::new(index_dtype, stream::iter(arrays.into_iter().map(Ok))),
            )
            .await
    }
}

/// Copy the variable-length values a slice uses out of its parent's buffers, so writing the
/// slice writes only its own bytes.
fn compact(slice: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let DType::Struct(fields, _) = slice.dtype().clone() else {
        return Ok(slice);
    };
    let rows = slice.execute::<StructArray>(ctx)?;
    let columns = (0..fields.nfields())
        .map(|idx| {
            let column = rows.unmasked_field(idx).clone();
            if matches!(column.dtype(), DType::Binary(_) | DType::Utf8(_)) {
                Ok(column
                    .execute::<VarBinViewArray>(ctx)?
                    .compact_buffers(ctx)?
                    .into_array())
            } else {
                Ok(column)
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(
        StructArray::try_new_with_dtype(columns, fields, rows.len(), rows.validity()?)?
            .into_array(),
    )
}

/// The default writer for index rows: small chunks, with binary payload columns (postings,
/// bitmaps, codes) written as-is in chunks of the same size.
///
/// Payloads are already compact, and leaving them out of dictionary encoding keeps each chunk's
/// payload in its own small segment, so a lookup reads only the payloads of the keys it matched.
fn default_index_strategy(session: &VortexSession, index_dtype: &DType) -> Arc<dyn LayoutStrategy> {
    let mut builder = WriteStrategyBuilder::from_session(session)
        .with_row_block_size(DEFAULT_INDEX_CHUNK_ROWS)
        .with_data_block_target_bytes(None);
    if let Some(fields) = index_dtype.as_struct_fields_opt() {
        for (name, dtype) in fields.names().iter().zip(fields.fields()) {
            if matches!(dtype, DType::Binary(_)) {
                let payload = RepartitionStrategy::new(
                    ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                    RepartitionWriterOptions {
                        block_size_minimum: 0,
                        block_len_multiple: DEFAULT_INDEX_CHUNK_ROWS,
                        block_size_target: None,
                        canonicalize: true,
                    },
                );
                builder = builder
                    .with_field_writer(FieldPath::from_name(name.clone()), Arc::new(payload));
            }
        }
    }
    builder.build()
}

/// Rows of one covered data file selected by an index lookup.
#[derive(Clone, Debug)]
pub struct FileSelection {
    /// Position of the file in [`IndexFile::files`].
    pub file: usize,
    /// Rows to read, local to the file. Pass this to the file's scan.
    pub selection: Selection,
}

/// A ranked search hit in a covered data file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FileHit {
    /// Position of the file in [`IndexFile::files`].
    pub file: usize,
    /// Row within that file.
    pub row: u64,
    /// Higher is better.
    pub score: f32,
}

/// An opened standalone index file.
pub struct IndexFile<V: IndexVTable> {
    info: IndexFileInfo,
    rows: IndexRows<V>,
    /// Start of each covered file in the combined row space, plus the total at the end.
    offsets: Vec<u64>,
}

impl<V: IndexVTable> IndexFile<V> {
    /// Open an index file written by [`IndexFileBuilder`] for kind `vtable`.
    pub fn open(vtable: V, file: &VortexFile) -> VortexResult<Self> {
        let info = IndexFileInfo::read(file)?;
        if info.kind != vtable.id().as_ref() {
            vortex_bail!(
                "Index file holds a {} index, not {}",
                info.kind,
                vtable.id()
            );
        }
        let options = vtable.deserialize_options(&info.options)?;
        let index_dtype = vtable
            .index_dtype(&info.source_dtype, &options)
            .ok_or_else(|| {
                vortex_err!(
                    "Index {} no longer indexes {}",
                    info.kind,
                    info.source_dtype
                )
            })?;
        vortex_ensure_eq!(
            file.dtype(),
            &index_dtype,
            "Index file rows do not match the dtype the index kind declares"
        );

        let mut offsets = Vec::with_capacity(info.files.len() + 1);
        let mut total = 0u64;
        offsets.push(0);
        for covered in &info.files {
            total += covered.row_count;
            offsets.push(total);
        }

        let rows = IndexRows {
            vtable: Arc::new(vtable),
            options,
            source_dtype: info.source_dtype.clone(),
            index_dtype,
            data_row_count: total,
            file: file.scan_file(),
            session: file.session().clone(),
            memo: OnceLock::new(),
        };
        Ok(Self {
            info,
            rows,
            offsets,
        })
    }

    /// The index content, for lookups and ranked search in the combined row space.
    pub fn rows(&self) -> &IndexRows<V> {
        &self.rows
    }

    /// The indexed column of the covered data files.
    pub fn column(&self) -> &str {
        &self.info.column
    }

    /// The covered data files, in row-space order.
    pub fn files(&self) -> &[CoveredFile] {
        &self.info.files
    }

    /// Answer a predicate over the indexed column, as a row selection per covered file.
    ///
    /// Files with no possible match are left out, so a caller never opens them. Returns `None`
    /// when the index makes no claim on `expr`; scan every file then.
    pub async fn locate_files(
        &self,
        expr: &Expression,
    ) -> VortexResult<Option<(IndexExactness, Vec<FileSelection>)>> {
        let Some((exactness, locator)) = self.rows.locate(expr).await? else {
            return Ok(None);
        };
        let mut selections = Vec::new();
        for (file, window) in self.offsets.windows(2).enumerate() {
            let rows = local_rows(&locator, window[0]..window[1])?;
            if !rows.is_empty() {
                selections.push(FileSelection {
                    file,
                    selection: Selection::IncludeRoaring(rows),
                });
            }
        }
        Ok(Some((exactness, selections)))
    }

    /// Map hits in the combined row space to their data files.
    pub fn hits_by_file(&self, hits: &[Hit]) -> VortexResult<Vec<FileHit>> {
        hits.iter()
            .map(|hit| {
                let file = self.offsets.partition_point(|&start| start <= hit.row) - 1;
                vortex_ensure!(
                    file < self.info.files.len(),
                    "Hit row {} is past the covered files",
                    hit.row
                );
                Ok(FileHit {
                    file,
                    row: hit.row - self.offsets[file],
                    score: hit.score,
                })
            })
            .collect()
    }
}

/// The rows of `locator` within `range`, shifted to start at zero.
fn local_rows(locator: &RowLocator, range: std::ops::Range<u64>) -> VortexResult<RoaringTreemap> {
    let mut rows = RoaringTreemap::new();
    match locator {
        RowLocator::Rows(bitmap) => {
            let start = u32::try_from(range.start)?;
            let end = u32::try_from(range.end)?;
            for row in bitmap.range(start..end) {
                rows.insert(u64::from(row) - range.start);
            }
        }
        RowLocator::Blocks { block_len, ids } => {
            let first = u32::try_from(range.start / block_len)?;
            let last = u32::try_from(range.end.div_ceil(*block_len))?;
            for block in ids.range(first..last) {
                let block_start = (u64::from(block) * block_len).max(range.start);
                let block_end = ((u64::from(block) + 1) * block_len).min(range.end);
                if block_start < block_end {
                    rows.insert_range(block_start - range.start..block_end - range.start);
                }
            }
        }
    }
    Ok(rows)
}
