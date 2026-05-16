//! On-disk framing for `*.vtrx` files.

/// Magic bytes prefixing every `*.vtrx` file.
pub const MAGIC: [u8; 4] = *b"VTRX";

/// Schema version bumped on any incompatible change.
pub const FORMAT_VERSION: u32 = 1;

/// One byte preceding every record payload, distinguishing event
/// records from per-turn snapshots.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordKind {
    Event = 0,
    Snapshot = 1,
}

impl RecordKind {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Event),
            1 => Some(Self::Snapshot),
            _ => None,
        }
    }
}

/// Record framing: a u32 little-endian length followed by a kind
/// byte and the postcard-encoded payload.
///
/// The reported `record_len` is `1 + payload_len` so that
/// `record_len` covers both the kind byte and the payload, and the
/// reader can advance by `record_len` from the cursor *after* it has
/// read `record_len` itself.
#[derive(Copy, Clone, Debug)]
pub struct RecordHeader {
    pub record_len: u32,
    pub kind: RecordKind,
}

/// Encode a record header to its 5-byte on-disk representation:
/// 4-byte little-endian length followed by the kind byte.
pub fn encode_record_header(record_len: u32, kind: RecordKind) -> [u8; 5] {
    let mut out = [0u8; 5];
    out[0..4].copy_from_slice(&record_len.to_le_bytes());
    out[4] = kind as u8;
    out
}

/// Decode the 5-byte record header. Returns `None` if the kind
/// byte is unrecognized.
pub fn decode_record_header(bytes: &[u8; 5]) -> Option<RecordHeader> {
    let record_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let kind = RecordKind::from_byte(bytes[4])?;
    Some(RecordHeader { record_len, kind })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let bytes = encode_record_header(123, RecordKind::Snapshot);
        let decoded = decode_record_header(&bytes).unwrap();
        assert_eq!(decoded.record_len, 123);
        assert_eq!(decoded.kind, RecordKind::Snapshot);
    }

    #[test]
    fn invalid_kind() {
        let bad = [0, 0, 0, 0, 99];
        assert!(decode_record_header(&bad).is_none());
    }
}
