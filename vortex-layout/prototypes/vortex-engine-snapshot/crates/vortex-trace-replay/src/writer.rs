//! Helpers for writing `*.vtrx` files.
//!
//! Used by the recorder (in `vortex-engine`) and by tests/fixtures.
//! The on-disk layout matches `trace-recording.md`:
//!
//! ```text
//! [u8; 4]  MAGIC = b"VTRX"
//! u32      FORMAT_VERSION (LE)
//! u32      header_len (LE)
//! [u8; header_len]  postcard-encoded TraceHeader
//!
//! repeat:
//!   u32  record_len (LE)        -- covers kind byte + payload
//!   u8   record_kind            -- 0=event, 1=snapshot
//!   [u8; record_len-1]  postcard-encoded payload
//! ```

use std::io::{self, Write};

use vortex_trace_format::framing::{FORMAT_VERSION, MAGIC, RecordKind, encode_record_header};
use vortex_trace_format::header::TraceHeader;
use vortex_trace_format::record::TraceRecord;
use vortex_trace_format::snapshot::TurnSnapshot;

/// Write the file header (magic + version + length-prefixed
/// postcard-encoded TraceHeader). Call once at the start of a file.
pub fn write_file_header<W: Write>(w: &mut W, header: &TraceHeader) -> io::Result<()> {
    let header_bytes = postcard::to_allocvec(header)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    w.write_all(&MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    let header_len = u32::try_from(header_bytes.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "header too large"))?;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header_bytes)?;
    Ok(())
}

pub fn write_event<W: Write>(w: &mut W, record: &TraceRecord) -> io::Result<()> {
    let payload = postcard::to_allocvec(record)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    write_payload(w, RecordKind::Event, &payload)
}

pub fn write_snapshot<W: Write>(w: &mut W, snapshot: &TurnSnapshot) -> io::Result<()> {
    let payload = postcard::to_allocvec(snapshot)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    write_payload(w, RecordKind::Snapshot, &payload)
}

fn write_payload<W: Write>(w: &mut W, kind: RecordKind, payload: &[u8]) -> io::Result<()> {
    let record_len = u32::try_from(payload.len() + 1)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "record too large"))?;
    let header = encode_record_header(record_len, kind);
    w.write_all(&header)?;
    w.write_all(payload)?;
    Ok(())
}

/// A small writer wrapper used in tests and the fixture generator.
pub struct TraceWriter<W: Write> {
    inner: W,
}

impl<W: Write> TraceWriter<W> {
    pub fn new(mut inner: W, header: &TraceHeader) -> io::Result<Self> {
        write_file_header(&mut inner, header)?;
        Ok(Self { inner })
    }

    pub fn write_event(&mut self, record: &TraceRecord) -> io::Result<()> {
        write_event(&mut self.inner, record)
    }

    pub fn write_snapshot(&mut self, snapshot: &TurnSnapshot) -> io::Result<()> {
        write_snapshot(&mut self.inner, snapshot)
    }

    pub fn into_inner(self) -> W {
        self.inner
    }

    pub fn inner_mut(&mut self) -> &mut W {
        &mut self.inner
    }
}
