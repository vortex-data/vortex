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
//! | `footer_hash`   | u64    | [`footer_hash`](crate::footer::footer_hash) of the region       |
//! | `footer`        | binary | the region itself, or empty when the manifest does not inline it |
//!
//! The footer holds the file's dtype, layout, segment map, and column statistics, so a planner
//! prunes a file on its statistics and opens it from the manifest alone: with the footer inlined
//! that costs no IO at all, and without it one read of a known range. The hash identifies the
//! exact bytes of a file, which is what out-of-file indexes bind to.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::v2::collect;
use vortex_io::VortexReadAt;
use vortex_io::VortexWrite;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_session::VortexSession;

use crate::footer::FooterRegion;
use crate::footer::footer_hash;

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

impl ManifestEntry {
    /// The entry for a file of `byte_size` bytes at `path`, from its footer region.
    pub fn new(
        path: impl Into<String>,
        byte_size: u64,
        region: &FooterRegion,
        inline: bool,
    ) -> Self {
        Self {
            path: path.into(),
            byte_size,
            row_count: region.footer.row_count(),
            footer_offset: region.offset,
            footer_len: region.bytes.len() as u64,
            footer_hash: footer_hash(&region.bytes),
            footer: inline.then(|| region.bytes.clone()),
        }
    }
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
        let chunks = collect(ScanBuilder::new(session.clone(), file.scan_file())).await?;
        let mut entries = Vec::new();
        for chunk in chunks {
            entries.extend(Self::from_array(chunk, session)?.entries);
        }
        Ok(Self { entries })
    }
}
