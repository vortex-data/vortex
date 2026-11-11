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
//! | `zone_segments` | list   | the byte ranges of the file's zone maps, by top-level column     |
//! | `stats`         | struct | the file's column statistics, when the manifest records them    |
//!
//! `stats` is a [`TableStats`] table, `struct { <column>: struct { <aggregate>: value? } }`, so
//! a planner prunes every file in one evaluation without parsing any footer. Its field names
//! are for display: the aggregate behind each field, and the files' dtype, are kept in the
//! manifest's `vortex.table.stats` metadata segment.
//!
//! The footer holds the file's dtype, layout, segment map, and column statistics, so a planner
//! prunes a file on its statistics and opens it from the manifest alone: with the footer inlined
//! that costs no IO at all, and without it one read of a known range. The hash identifies the
//! exact bytes of a file, which is what out-of-file indexes bind to.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::validity::Validity;
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
use crate::stats::TableStats;
use crate::stats::spec::STATS_SPEC_KEY;
use crate::stats::spec::read_spec;
use crate::stats::spec::write_spec;
use crate::zone_segments::ZoneSegment;
use crate::zone_segments::zone_segments;

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
    /// The segments of the file's zone maps, so a planner can fetch them without walking the
    /// layout.
    pub zone_segments: Vec<ZoneSegment>,
}

impl ManifestEntry {
    /// The entry for a file of `byte_size` bytes at `path`, from its footer region.
    pub fn try_new(
        path: impl Into<String>,
        byte_size: u64,
        region: &FooterRegion,
        inline: bool,
    ) -> VortexResult<Self> {
        Ok(Self {
            path: path.into(),
            byte_size,
            row_count: region.footer.row_count(),
            footer_offset: region.offset,
            footer_len: region.bytes.len() as u64,
            footer_hash: footer_hash(&region.bytes),
            footer: inline.then(|| region.bytes.clone()),
            zone_segments: zone_segments(&region.footer)?,
        })
    }
}

/// The files of a table, in table order.
#[derive(Clone, Debug, Default)]
pub struct Manifest {
    entries: Vec<ManifestEntry>,
    stats: Option<Arc<TableStats>>,
}

impl Manifest {
    /// A manifest over `entries`, in table order.
    pub fn new(entries: Vec<ManifestEntry>) -> Self {
        Self {
            entries,
            stats: None,
        }
    }

    /// The manifest with `stats`, one row per entry, as the column statistics of its files.
    pub fn with_stats(mut self, stats: TableStats) -> VortexResult<Self> {
        vortex_ensure_eq!(
            stats.len(),
            self.entries.len(),
            "the stats have a row per file"
        );
        self.stats = Some(Arc::new(stats));
        Ok(self)
    }

    /// The column statistics of every file, when the manifest records them.
    pub fn stats(&self) -> Option<&Arc<TableStats>> {
        self.stats.as_ref()
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
    pub fn to_array(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
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
        let mut fields = vec![
            ("path", VarBinArray::from(paths).into_array()),
            ("byte_size", u64s(|e| e.byte_size)),
            ("row_count", u64s(|e| e.row_count)),
            ("footer_offset", u64s(|e| e.footer_offset)),
            ("footer_len", u64s(|e| e.footer_len)),
            ("footer_hash", u64s(|e| e.footer_hash)),
            ("footer", VarBinArray::from(footers).into_array()),
            ("zone_segments", self.zone_segments_array(session)?),
        ];
        if let Some(stats) = &self.stats {
            fields.push(("stats", stats.table().clone()));
        }
        Ok(StructArray::from_fields(&fields)?.into_array())
    }

    /// `list<struct { column, segment, offset, length, alignment }>`, one list per file.
    fn zone_segments_array(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        let segments: Vec<&ZoneSegment> = self
            .entries
            .iter()
            .flat_map(|e| e.zone_segments.iter())
            .collect();
        let columns: Vec<&str> = segments.iter().map(|s| s.column.as_str()).collect();
        let elements = StructArray::from_fields(&[
            ("column", VarBinArray::from(columns).into_array()),
            (
                "segment",
                PrimitiveArray::from_iter(segments.iter().map(|s| s.segment)).into_array(),
            ),
            (
                "offset",
                PrimitiveArray::from_iter(segments.iter().map(|s| s.offset)).into_array(),
            ),
            (
                "length",
                PrimitiveArray::from_iter(segments.iter().map(|s| s.length)).into_array(),
            ),
            (
                "alignment",
                PrimitiveArray::from_iter(segments.iter().map(|s| s.alignment)).into_array(),
            ),
        ])?
        .into_array();
        let mut offsets = Vec::with_capacity(self.entries.len() + 1);
        offsets.push(0u64);
        let mut end = 0u64;
        for entry in &self.entries {
            end += entry.zone_segments.len() as u64;
            offsets.push(end);
        }
        let mut ctx = session.create_execution_ctx();
        Ok(ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array())
    }

    /// Reads a manifest's entries back from the struct array [`to_array`](Self::to_array) made.
    /// Its statistics need the schema [`write`](Self::write) keeps beside the array, so only
    /// [`read`](Self::read) restores them.
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
        let zone_segments = match array.struct_fields().find("zone_segments") {
            Some(_) => read_zone_segments(field("zone_segments")?, &mut ctx)?,
            None => vec![Vec::new(); byte_size.len()],
        };
        let paths = field("path")?.execute::<VarBinViewArray>(&mut ctx)?;
        let footers = field("footer")?.execute::<VarBinViewArray>(&mut ctx)?;
        vortex_ensure_eq!(paths.len(), byte_size.len());
        vortex_ensure_eq!(zone_segments.len(), byte_size.len());
        let mut zone_segments = zone_segments.into_iter();
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
                    zone_segments: zone_segments.next().unwrap_or_default(),
                })
            })
            .collect::<VortexResult<_>>()?;
        Ok(Self::new(entries))
    }

    /// Writes the manifest as a Vortex file.
    pub async fn write<W: VortexWrite + Unpin>(
        &self,
        session: &VortexSession,
        write: W,
    ) -> VortexResult<()> {
        let mut options = session.write_options();
        if let Some(stats) = &self.stats {
            options = options.with_metadata_segment(STATS_SPEC_KEY, write_spec(stats)?);
        }
        options
            .write(write, self.to_array(session)?.to_array_stream())
            .await?;
        Ok(())
    }

    /// Reads a manifest from a Vortex file, with the statistics it records.
    pub async fn read(session: &VortexSession, read: Arc<dyn VortexReadAt>) -> VortexResult<Self> {
        let file = session.open_options().include_metadata().open(read).await?;
        let spec = file.metadata_segment(STATS_SPEC_KEY).cloned();
        let chunks = collect(ScanBuilder::new(session.clone(), file.scan_file())).await?;
        let mut ctx = session.create_execution_ctx();
        let mut entries = Vec::new();
        let mut stats_chunks = Vec::new();
        for chunk in chunks {
            if spec.is_some() {
                let chunk = chunk.clone().execute::<StructArray>(&mut ctx)?;
                stats_chunks.push(chunk.unmasked_field_by_name("stats")?.clone());
            }
            entries.extend(Self::from_array(chunk, session)?.entries);
        }
        let manifest = Self::new(entries);
        let Some(spec) = spec else {
            return Ok(manifest);
        };
        let stats_dtype = file
            .dtype()
            .as_struct_fields_opt()
            .and_then(|fields| fields.field("stats"))
            .ok_or_else(|| vortex_err!("the manifest records a stats schema but no stats"))?;
        let table = match stats_chunks.len() {
            1 => stats_chunks.remove(0),
            _ => ChunkedArray::try_new(stats_chunks, stats_dtype)?.into_array(),
        };
        let row_counts =
            PrimitiveArray::from_iter(manifest.entries.iter().map(|e| e.row_count)).into_array();
        let stats = read_spec(spec.as_slice(), table, row_counts, session)?;
        manifest.with_stats(stats)
    }
}

/// The zone segments of each file, from the `zone_segments` column.
fn read_zone_segments(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Vec<ZoneSegment>>> {
    let lists = array.execute::<ListViewArray>(ctx)?;
    let elements = lists.elements().clone().execute::<StructArray>(ctx)?;
    let column = elements
        .unmasked_field_by_name("column")?
        .clone()
        .execute::<VarBinViewArray>(ctx)?;
    let mut prim = |name: &str| {
        elements
            .unmasked_field_by_name(name)?
            .clone()
            .execute::<PrimitiveArray>(ctx)
    };
    let segment = prim("segment")?;
    let offset = prim("offset")?;
    let length = prim("length")?;
    let alignment = prim("alignment")?;
    let (segment, offset, length, alignment) = (
        segment.as_slice::<u32>(),
        offset.as_slice::<u64>(),
        length.as_slice::<u32>(),
        alignment.as_slice::<u8>(),
    );
    (0..lists.len())
        .map(|i| {
            let start = lists.offset_at(i, ctx);
            let end = start + lists.size_at(i, ctx);
            (start..end)
                .map(|j| {
                    Ok(ZoneSegment {
                        column: String::from_utf8(column.bytes_at(j).to_vec())
                            .map_err(|e| vortex_err!("zone segment column is not utf8: {e}"))?,
                        segment: segment[j],
                        offset: offset[j],
                        length: length[j],
                        alignment: alignment[j],
                    })
                })
                .collect()
        })
        .collect()
}
