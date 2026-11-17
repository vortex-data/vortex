// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The manifest: one row per data file, itself stored as a Vortex file.
//!
//! | column          | type   | meaning                                                         |
//! |-----------------|--------|-----------------------------------------------------------------|
//! | `path`          | utf8   | the file, relative to the table                                 |
//! | `byte_size`     | u64    | the file's length                                               |
//! | `row_count`     | u64    | rows in the file                                                |
//! | `footer_offset` | u64    | first byte of the file's footer region                          |
//! | `footer_len`    | u64    | length of the footer region, through the end of the file        |
//! | `footer_hash`   | u64    | [`footer_hash`] of the region                                   |
//! | `footer`        | binary | the region itself, or empty when the manifest does not inline it |
//!
//! The footer holds the file's dtype, layout, segment map, and column statistics, so a planner
//! prunes a file on its statistics and opens it from the manifest alone: with the footer inlined
//! that costs no IO at all, and without it one read of a known range. The hash identifies the
//! exact bytes of a file, which is what out-of-file indexes bind to.

use std::sync::Arc;

use futures::TryStreamExt;
use twox_hash::XxHash3_64;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::v2::ScanDriver;
use vortex_io::VortexReadAt;
use vortex_io::VortexWrite;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_session::VortexSession;

/// One data file of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestEntry {
    /// The file, relative to the table.
    pub path: String,
    /// The file's length in bytes.
    pub byte_size: u64,
    /// Rows in the file.
    pub row_count: u64,
    /// First byte of the file's footer region.
    pub footer_offset: u64,
    /// Length of the footer region, through the end of the file.
    pub footer_len: u64,
    /// Hash of the footer region.
    pub footer_hash: u64,
    /// The footer region, when the manifest inlines it.
    pub footer: Option<ByteBuffer>,
}

/// The hash a manifest records for a footer region: XXH3-64 over the region's bytes. A rewritten
/// file, even one with the same path and size, hashes differently.
pub fn footer_hash(region: &[u8]) -> u64 {
    XxHash3_64::oneshot(region)
}

/// The files of a table, in table order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    entries: Vec<ManifestEntry>,
}

impl Manifest {
    /// A manifest over `entries`, in table order.
    pub fn new(entries: Vec<ManifestEntry>) -> Self {
        Self { entries }
    }

    /// The table's files, in table order. A file's ordinal is its index here.
    pub fn entries(&self) -> &[ManifestEntry] {
        &self.entries
    }

    /// Rows in the table.
    pub fn row_count(&self) -> u64 {
        self.entries.iter().map(|entry| entry.row_count).sum()
    }

    /// The manifest as a struct array, one row per file.
    pub fn to_array(&self) -> VortexResult<ArrayRef> {
        let u64s = |f: fn(&ManifestEntry) -> u64| {
            PrimitiveArray::from_iter(self.entries.iter().map(f)).into_array()
        };
        let paths: Vec<&str> = self.entries.iter().map(|e| e.path.as_str()).collect();
        let footers: Vec<&[u8]> = self
            .entries
            .iter()
            .map(|e| {
                e.footer
                    .as_ref()
                    .map_or(&[][..], |footer| footer.as_slice())
            })
            .collect();
        Ok(StructArray::from_fields(&[
            ("path", VarBinArray::from(paths).into_array()),
            ("byte_size", u64s(|e| e.byte_size)),
            ("row_count", u64s(|e| e.row_count)),
            ("footer_offset", u64s(|e| e.footer_offset)),
            ("footer_len", u64s(|e| e.footer_len)),
            ("footer_hash", u64s(|e| e.footer_hash)),
            ("footer", VarBinArray::from(footers).into_array()),
        ])?
        .into_array())
    }

    /// Reads a manifest back from the struct array [`to_array`](Self::to_array) made.
    pub fn from_array(array: ArrayRef, session: &VortexSession) -> VortexResult<Self> {
        let mut ctx = session.create_execution_ctx();
        let array = array.execute::<StructArray>(&mut ctx)?;
        let field = |name: &str| array.unmasked_field_by_name(name).cloned();
        let mut u64s = |name: &str| -> VortexResult<Vec<u64>> {
            let column = field(name)?.execute::<PrimitiveArray>(&mut ctx)?;
            Ok(column.as_slice::<u64>().to_vec())
        };
        let byte_size = u64s("byte_size")?;
        let row_count = u64s("row_count")?;
        let footer_offset = u64s("footer_offset")?;
        let footer_len = u64s("footer_len")?;
        let hash = u64s("footer_hash")?;
        let paths = field("path")?.execute::<VarBinViewArray>(&mut ctx)?;
        let footers = field("footer")?.execute::<VarBinViewArray>(&mut ctx)?;
        vortex_ensure_eq!(paths.len(), byte_size.len());
        let entries = (0..paths.len())
            .map(|i| {
                let path = String::from_utf8(paths.bytes_at(i).to_vec())
                    .map_err(|e| vortex_err!("manifest path {i} is not utf8: {e}"))?;
                let footer = footers.bytes_at(i);
                Ok(ManifestEntry {
                    path,
                    byte_size: byte_size[i],
                    row_count: row_count[i],
                    footer_offset: footer_offset[i],
                    footer_len: footer_len[i],
                    footer_hash: hash[i],
                    footer: (!footer.is_empty()).then_some(footer),
                })
            })
            .collect::<VortexResult<_>>()?;
        Ok(Self { entries })
    }

    /// Writes the manifest as a Vortex file.
    pub async fn write<W: VortexWrite + Unpin>(
        &self,
        session: &VortexSession,
        write: W,
    ) -> VortexResult<()> {
        session
            .write_options()
            .write(write, self.to_array()?.to_array_stream())
            .await?;
        Ok(())
    }

    /// Reads a manifest from a Vortex file.
    pub async fn read(session: &VortexSession, read: Arc<dyn VortexReadAt>) -> VortexResult<Self> {
        let file = session.open_options().open(read).await?;
        // One ordered stream: the manifest is small, and its rows are the table's file order.
        let chunks: Vec<ArrayRef> = ScanDriver::new(session.handle(), 1, true)
            .run(ScanBuilder::new(session.clone(), file.scan_file()))
            .try_collect()
            .await?;
        let mut entries = Vec::new();
        for chunk in chunks {
            entries.extend(Self::from_array(chunk, session)?.entries);
        }
        Ok(Self { entries })
    }
}

/// Builds a manifest one data file at a time, reading each file's footer region.
pub struct ManifestBuilder {
    session: VortexSession,
    inline_footers: bool,
    entries: Vec<ManifestEntry>,
}

impl ManifestBuilder {
    /// A builder that inlines every file's footer region into the manifest, so scans open files
    /// without reading their footers.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            inline_footers: true,
            entries: Vec::new(),
        }
    }

    /// Whether to inline footer regions. Without, a scan reads each file's region once, at the
    /// offset and length the manifest records.
    pub fn with_inline_footers(mut self, inline: bool) -> Self {
        self.inline_footers = inline;
        self
    }

    /// Appends the file at `path`, whose bytes `read` serves, as the table's next file.
    ///
    /// The footer is found the way any open finds it; its region is everything from the end of
    /// the file's segments to the end of the file.
    pub async fn add(
        &mut self,
        path: impl Into<String>,
        read: Arc<dyn VortexReadAt>,
    ) -> VortexResult<()> {
        let byte_size = read.size().await?;
        let file = self
            .session
            .open_options()
            .with_file_size(byte_size)
            .open(Arc::clone(&read))
            .await?;
        let footer = file.footer();
        let footer_offset = footer
            .segments_end()
            .ok_or_else(|| vortex_err!("the file's segment ranges overflow"))?;
        let region = read
            .read_at(
                footer_offset,
                usize::try_from(byte_size - footer_offset)?,
                Alignment::none(),
            )
            .await?
            .try_into_host()?
            .await?;
        self.entries.push(ManifestEntry {
            path: path.into(),
            byte_size,
            row_count: footer.row_count(),
            footer_offset,
            footer_len: region.len() as u64,
            footer_hash: footer_hash(&region),
            footer: self.inline_footers.then_some(region),
        });
        Ok(())
    }

    /// The manifest of the files added, in the order they were added.
    pub fn finish(self) -> Manifest {
        Manifest::new(self.entries)
    }
}
