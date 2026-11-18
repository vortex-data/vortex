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
//!
//! The footer holds the file's dtype, layout, segment map, and column statistics. Knowing exactly
//! where it is, a reader fetches it in one read instead of guessing a tail size and reading
//! again, without the manifest growing with the footers, which are large for wide files.

use std::sync::Arc;

use futures::TryStreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
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
        Ok(Self {
            path: path.into(),
            byte_size,
            row_count: file.row_count(),
            footer_offset,
        })
    }
}

/// The files of a table, in table order. A file's ordinal is its index in `entries`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    /// The table's files.
    pub entries: Vec<ManifestEntry>,
}

impl Manifest {
    /// The manifest as a struct array, one row per file.
    pub fn to_array(&self) -> VortexResult<ArrayRef> {
        let u64s = |f: fn(&ManifestEntry) -> u64| {
            PrimitiveArray::from_iter(self.entries.iter().map(f)).into_array()
        };
        let paths: Vec<&str> = self.entries.iter().map(|e| e.path.as_str()).collect();
        Ok(StructArray::from_fields(&[
            ("path", VarBinArray::from(paths).into_array()),
            ("byte_size", u64s(|e| e.byte_size)),
            ("row_count", u64s(|e| e.row_count)),
            ("footer_offset", u64s(|e| e.footer_offset)),
        ])?
        .into_array())
    }

    /// Appends the rows of a struct array that [`to_array`](Self::to_array) made.
    fn extend_from_array(&mut self, array: ArrayRef, session: &VortexSession) -> VortexResult<()> {
        let mut ctx = session.create_execution_ctx();
        vortex_ensure_eq!(array.dtype(), Self::default().to_array()?.dtype());
        let array = array.execute::<StructArray>(&mut ctx)?;
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
        Ok(())
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
        let mut manifest = Self::default();
        for chunk in chunks {
            manifest.extend_from_array(chunk, session)?;
        }
        Ok(manifest)
    }
}
