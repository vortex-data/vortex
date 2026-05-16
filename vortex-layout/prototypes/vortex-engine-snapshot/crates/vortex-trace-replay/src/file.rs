use std::fs::File;
use std::path::Path;

use vortex_trace_format::framing::{FORMAT_VERSION, MAGIC, decode_record_header};
use vortex_trace_format::header::TraceHeader;

use crate::error::ReplayError;
use crate::index::{TimelineSummary, TraceIndex};

/// Owned bytes for a trace file.
///
/// On native we mmap the file via `memmap2::Mmap` (read-only, the
/// file is closed once mapped). On wasm we hold a `Vec<u8>` that
/// the JS side transferred to us.
pub enum TraceBytes {
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl TraceBytes {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped(m) => m.as_ref(),
            Self::Owned(v) => v.as_slice(),
        }
    }
}

pub struct TraceFile {
    bytes: TraceBytes,
    header: TraceHeader,
    index: TraceIndex,
    summary: TimelineSummary,
}

impl TraceFile {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: &Path) -> Result<Self, ReplayError> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_trace_bytes(TraceBytes::Mapped(mmap))
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, ReplayError> {
        Self::from_trace_bytes(TraceBytes::Owned(bytes))
    }

    fn from_trace_bytes(bytes: TraceBytes) -> Result<Self, ReplayError> {
        let (header, header_end_offset) = parse_file_header(bytes.as_slice())?;
        let (index, summary) =
            TraceIndex::build(bytes.as_slice(), header_end_offset, header.operators.len())?;
        Ok(Self {
            bytes,
            header,
            index,
            summary,
        })
    }

    pub fn header(&self) -> &TraceHeader {
        &self.header
    }

    pub fn turns(&self) -> u32 {
        self.index.events_per_turn.len() as u32
    }

    pub fn events_in_turn(&self, turn: u32) -> u32 {
        self.index
            .events_per_turn
            .get(turn as usize)
            .copied()
            .unwrap_or(0)
    }

    pub fn timeline_summary(&self) -> &TimelineSummary {
        &self.summary
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    pub(crate) fn index(&self) -> &TraceIndex {
        &self.index
    }
}

fn parse_file_header(bytes: &[u8]) -> Result<(TraceHeader, u64), ReplayError> {
    if bytes.len() < 12 {
        return Err(ReplayError::Truncated { offset: 0 });
    }
    let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != MAGIC {
        return Err(ReplayError::BadMagic);
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != FORMAT_VERSION {
        return Err(ReplayError::UnsupportedVersion {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let header_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let header_start = 12usize;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or(ReplayError::Truncated { offset: 12 })?;
    if bytes.len() < header_end {
        return Err(ReplayError::Truncated {
            offset: header_end as u64,
        });
    }
    let header_bytes = &bytes[header_start..header_end];
    let header: TraceHeader = postcard::from_bytes(header_bytes)?;
    Ok((header, header_end as u64))
}

/// Public for the index module too.
pub(crate) fn read_record_header(
    bytes: &[u8],
    offset: u64,
) -> Result<(vortex_trace_format::framing::RecordHeader, u64), ReplayError> {
    let off = offset as usize;
    if bytes.len() < off + 5 {
        return Err(ReplayError::Truncated { offset });
    }
    let header_bytes: [u8; 5] = bytes[off..off + 5].try_into().unwrap();
    let header = decode_record_header(&header_bytes).ok_or_else(|| {
        ReplayError::InvalidRecordKind(header_bytes[4])
    })?;
    Ok((header, offset + 5))
}
