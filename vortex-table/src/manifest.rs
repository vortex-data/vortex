// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The manifest: one row per data file, itself stored as a Vortex file.
//!
//! | column          | type | meaning                                                   |
//! |-----------------|------|-----------------------------------------------------------|
//! | `path`          | utf8 | the file, relative to the table                           |
//! | `byte_size`     | u64  | the file's length                                         |
//! | `row_count`     | u64  | rows in the file                                          |
//! | `footer_offset` | u64  | where the file's footer region starts; it runs to the end |
//! | `stats`         | struct, optional | each column's file statistics, see [`TableStats`] |
//!
//! The footer holds the file's dtype, layout, segment map, and column statistics. Knowing exactly
//! where it is, a reader fetches it in one read instead of guessing a tail size and reading
//! again, without the manifest growing with the footers, which are large for wide files.
//!
//! When the manifest has stats, it records the files' row dtype in its
//! [`DTYPE_METADATA_KEY`] metadata segment, and a scan drops files on their stats alone.

use std::sync::Arc;

use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::flatbuffers::FlatBuffer;
use vortex_array::flatbuffers::WriteFlatBufferExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::v2::ScanDriver;
use vortex_io::VortexReadAt;
use vortex_io::VortexWrite;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_session::VortexSession;

use crate::TableStats;

const ENTRY_FIELDS: [&str; 4] = ["path", "byte_size", "row_count", "footer_offset"];

/// The manifest metadata segment that holds the files' row dtype, written when it has stats.
pub const DTYPE_METADATA_KEY: &str = "vortex.table.dtype";

/// One data file of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestEntry {
    /// The file, relative to the table.
    pub path: String,
    /// The file's length in bytes.
    pub byte_size: u64,
    /// Rows in the file.
    pub row_count: u64,
    /// Where the file's footer region starts: the end of its segments. It runs to the end of
    /// the file.
    pub footer_offset: u64,
}

impl ManifestEntry {
    /// The entry for the file at `path`, whose bytes `read` serves.
    ///
    /// The footer is found the way any open finds it. Its region starts where the file's
    /// segments end.
    pub async fn read(
        session: &VortexSession,
        path: impl Into<String>,
        read: Arc<dyn VortexReadAt>,
    ) -> VortexResult<Self> {
        Ok(Self::read_file(session, path, read).await?.0)
    }

    /// The entry for the file at `path`, and the opened file.
    async fn read_file(
        session: &VortexSession,
        path: impl Into<String>,
        read: Arc<dyn VortexReadAt>,
    ) -> VortexResult<(Self, VortexFile)> {
        let byte_size = read.size().await?;
        let file = session
            .open_options()
            .with_file_size(byte_size)
            .open(Arc::clone(&read))
            .await?;
        let footer_offset = file
            .footer()
            .segments_end()
            .ok_or_else(|| vortex_err!("the file's segment ranges overflow"))?;
        let entry = Self {
            path: path.into(),
            byte_size,
            row_count: file.row_count(),
            footer_offset,
        };
        Ok((entry, file))
    }

    /// Opens the file, fetching its footer in one read: the reader is told exactly how much of
    /// the tail to read, so it never reads a guessed tail and then again.
    pub async fn open(
        &self,
        session: &VortexSession,
        read: Arc<dyn VortexReadAt>,
    ) -> VortexResult<VortexFile> {
        session
            .open_options()
            .with_file_size(self.byte_size)
            .with_initial_read_size(usize::try_from(
                self.byte_size
                    .checked_sub(self.footer_offset)
                    .ok_or_else(|| vortex_err!("{}: footer offset is past the end", self.path))?,
            )?)
            .open(read)
            .await
    }
}

/// The files of a table, in table order. A file's ordinal is its index in `entries`.
#[derive(Clone, Debug, Default)]
pub struct Manifest {
    /// The table's files.
    pub entries: Vec<ManifestEntry>,
    /// The files' statistics, one row per entry.
    pub stats: Option<TableStats>,
}

impl Manifest {
    /// The manifest of `files`, each a path and the reader of its bytes, with their statistics.
    /// Every file must have the same row dtype.
    pub async fn build(
        session: &VortexSession,
        files: Vec<(String, Arc<dyn VortexReadAt>)>,
    ) -> VortexResult<Self> {
        let read: Vec<(ManifestEntry, VortexFile)> = stream::iter(files)
            .map(|(path, read)| ManifestEntry::read_file(session, path, read))
            .buffered(crate::scan::OPEN_CONCURRENCY)
            .try_collect()
            .await?;
        let Some((_, first)) = read.first() else {
            return Ok(Self::default());
        };
        let dtype = first.dtype().clone();
        for (entry, file) in &read {
            vortex_ensure_eq!(file.dtype(), &dtype, "{} has another dtype", entry.path);
        }
        let file_stats: Vec<_> = read.iter().map(|(_, f)| f.footer().statistics()).collect();
        let stats = TableStats::from_files(dtype, &file_stats)?;
        Ok(Self {
            entries: read.into_iter().map(|(entry, _)| entry).collect(),
            stats: Some(stats),
        })
    }

    /// The manifest as a struct array, one row per file.
    pub fn to_array(&self) -> VortexResult<ArrayRef> {
        let u64s = |f: fn(&ManifestEntry) -> u64| {
            PrimitiveArray::from_iter(self.entries.iter().map(f)).into_array()
        };
        let paths: Vec<&str> = self.entries.iter().map(|e| e.path.as_str()).collect();
        let mut fields = vec![
            ("path", VarBinArray::from(paths).into_array()),
            ("byte_size", u64s(|e| e.byte_size)),
            ("row_count", u64s(|e| e.row_count)),
            ("footer_offset", u64s(|e| e.footer_offset)),
        ];
        if let Some(stats) = &self.stats {
            vortex_ensure_eq!(stats.array().len(), self.entries.len());
            fields.push(("stats", stats.array().clone().into_array()));
        }
        Ok(StructArray::from_fields(&fields)?.into_array())
    }

    /// Appends the entries of a struct array that [`to_array`](Self::to_array) made, and returns
    /// its stats column, if any.
    fn extend_from_array(
        &mut self,
        array: ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut ctx = session.create_execution_ctx();
        let array = array.execute::<StructArray>(&mut ctx)?;
        let stats = array.unmasked_field_by_name_opt("stats").cloned();
        let entries = match stats {
            Some(_) => array.project(&ENTRY_FIELDS.map(FieldName::from))?,
            None => array,
        };
        vortex_ensure_eq!(entries.dtype(), Self::default().to_array()?.dtype());
        let array = entries;
        let field = |name: &str| array.unmasked_field_by_name(name).cloned();
        let byte_size = field("byte_size")?.execute::<PrimitiveArray>(&mut ctx)?;
        let row_count = field("row_count")?.execute::<PrimitiveArray>(&mut ctx)?;
        let paths = field("path")?.execute::<VarBinViewArray>(&mut ctx)?;
        let footer_offset = field("footer_offset")?.execute::<PrimitiveArray>(&mut ctx)?;
        for i in 0..array.len() {
            self.entries.push(ManifestEntry {
                path: String::from_utf8(paths.bytes_at(i).to_vec())
                    .map_err(|e| vortex_err!("manifest path {i} is not utf8: {e}"))?,
                byte_size: byte_size.as_slice::<u64>()[i],
                row_count: row_count.as_slice::<u64>()[i],
                footer_offset: footer_offset.as_slice::<u64>()[i],
            });
        }
        Ok(stats)
    }

    /// Writes the manifest as a Vortex file.
    pub async fn write<W: VortexWrite + Unpin>(
        &self,
        session: &VortexSession,
        write: W,
    ) -> VortexResult<()> {
        let mut options = session.write_options();
        if let Some(stats) = &self.stats {
            let dtype = stats.dtype().write_flatbuffer_bytes()?.into_inner();
            options = options.with_metadata_segment(DTYPE_METADATA_KEY, dtype);
        }
        options
            .write(write, self.to_array()?.to_array_stream())
            .await?;
        Ok(())
    }

    /// Reads a manifest from a Vortex file.
    pub async fn read(session: &VortexSession, read: Arc<dyn VortexReadAt>) -> VortexResult<Self> {
        let file = session.open_options().include_metadata().open(read).await?;
        // One ordered stream: the manifest is small, and its rows are the table's file order.
        let chunks: Vec<ArrayRef> = ScanDriver::new(session.handle(), 1, true)
            .run(ScanBuilder::new(session.clone(), file.scan_file()))
            .try_collect()
            .await?;
        let mut manifest = Self::default();
        let mut stats = Vec::new();
        for chunk in chunks {
            stats.extend(manifest.extend_from_array(chunk, session)?);
        }
        if let Some(dtype) = file.metadata_segment(DTYPE_METADATA_KEY) {
            let dtype = DType::from_flatbuffer(FlatBuffer::align_from(dtype.clone()), session)?;
            let array = match stats.first() {
                Some(first) => ChunkedArray::try_new(stats.clone(), first.dtype().clone())?
                    .into_array()
                    .execute::<StructArray>(&mut session.create_execution_ctx())?,
                None => TableStats::from_files(dtype.clone(), &[])?.array().clone(),
            };
            manifest.stats = Some(TableStats::new(dtype, array));
        }
        Ok(manifest)
    }
}
