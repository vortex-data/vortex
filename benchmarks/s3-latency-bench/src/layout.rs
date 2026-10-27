// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic "Vortex-like" file layouts.
//!
//! The bytes are random, but the shape matches a Vortex file written by the default strategy:
//! per-row-group column chunks (with optional validity buffers), per-column zone maps after the
//! data, then a flatbuffer footer, a postscript and the 8-byte `EOF` (version, postscript length,
//! `VTXF` magic). The scan simulation reads these files exactly the way `vortex-file` would.

use std::ops::Range;

use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::LogNormal;
use serde::Deserialize;
use serde::Serialize;

/// Bytes `vortex-file` reads from the end of the file before it knows the footer size
/// (`MAX_POSTSCRIPT_SIZE + EOF_SIZE`).
pub const INITIAL_READ_SIZE: u64 = 65527 + 8;
pub const EOF_SIZE: u64 = 8;
pub const MAGIC: &[u8; 4] = b"VTXF";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentKind {
    Data,
    Validity,
    ZoneMap,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub offset: u64,
    pub len: u64,
    pub column: u32,
    /// Row group for data and validity segments; `u32::MAX` for zone maps.
    pub row_group: u32,
    pub kind: SegmentKind,
}

impl Segment {
    pub fn range(&self) -> Range<u64> {
        self.offset..self.offset + self.len
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileLayout {
    pub name: String,
    pub size: u64,
    pub columns: u32,
    pub row_groups: u32,
    pub footer_offset: u64,
    pub footer_len: u64,
    pub postscript_len: u64,
    pub segments: Vec<Segment>,
}

impl FileLayout {
    /// Generate a layout whose total size is close to `target_size`.
    pub fn generate(name: String, target_size: u64, seed: u64) -> anyhow::Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let columns: u32 = rng.random_range(6..=16);

        // Per-column average chunk size, log-uniform in [2 KiB, 1 MiB]: from tiny boolean and
        // dictionary-coded columns up to wide string columns.
        let col_mean: Vec<f64> = (0..columns)
            .map(|_| 2f64.powf(rng.random_range(11.0..20.0)))
            .collect();
        let nullable: Vec<bool> = (0..columns).map(|_| rng.random_bool(0.3)).collect();
        let jitter = LogNormal::new(0.0, 0.35)?;

        // Roughly what a Vortex footer costs: a fixed header, ~40 bytes per segment spec and
        // layout node, and some per-column dtype/statistics metadata.
        let footer_estimate = |segments: u64| 4096 + 40 * segments + 256 * u64::from(columns);
        let postscript_len = 512;

        let mut segments = Vec::new();
        let mut offset = 0u64;
        let mut row_groups = 0u32;
        loop {
            let mut group = Vec::new();
            let mut group_offset = offset;
            for col in 0..columns {
                let len = ((col_mean[col as usize] * jitter.sample(&mut rng)) as u64).max(64);
                group.push(Segment {
                    offset: group_offset,
                    len,
                    column: col,
                    row_group: row_groups,
                    kind: SegmentKind::Data,
                });
                group_offset += len;
                if nullable[col as usize] {
                    let len = (len / 32).max(64);
                    group.push(Segment {
                        offset: group_offset,
                        len,
                        column: col,
                        row_group: row_groups,
                        kind: SegmentKind::Validity,
                    });
                    group_offset += len;
                }
            }
            let nsegs = (segments.len() + group.len()) as u64 + u64::from(columns);
            let zone_bytes = u64::from(columns) * 64 * u64::from(row_groups + 1);
            let projected =
                group_offset + zone_bytes + footer_estimate(nsegs) + postscript_len + EOF_SIZE;
            if projected > target_size && row_groups > 0 {
                break;
            }
            segments.extend(group);
            offset = group_offset;
            row_groups += 1;
        }

        for col in 0..columns {
            let len = 64 * u64::from(row_groups) + 128;
            segments.push(Segment {
                offset,
                len,
                column: col,
                row_group: u32::MAX,
                kind: SegmentKind::ZoneMap,
            });
            offset += len;
        }

        let footer_len = footer_estimate(segments.len() as u64);
        let footer_offset = offset;
        let size = footer_offset + footer_len + postscript_len + EOF_SIZE;
        Ok(Self {
            name,
            size,
            columns,
            row_groups,
            footer_offset,
            footer_len,
            postscript_len,
            segments,
        })
    }

    /// Materialize the file bytes: random payload followed by a valid Vortex `EOF` marker.
    pub fn bytes(&self, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed_f11e);
        let mut buf = vec![0u8; self.size as usize];
        rng.fill_bytes(&mut buf);
        let eof = buf.len() - EOF_SIZE as usize;
        buf[eof..eof + 2].copy_from_slice(&1u16.to_le_bytes());
        buf[eof + 2..eof + 4].copy_from_slice(&(self.postscript_len as u16).to_le_bytes());
        buf[eof + 4..].copy_from_slice(MAGIC);
        buf
    }

    /// The reads `vortex-file` issues to open this file: the initial suffix read, plus a second
    /// read if the footer did not fit into it.
    pub fn open_reads(&self) -> (Range<u64>, Option<Range<u64>>) {
        let initial = self.size.saturating_sub(INITIAL_READ_SIZE)..self.size;
        let rest =
            (self.footer_offset < initial.start).then_some(self.footer_offset..initial.start);
        (initial, rest)
    }

    pub fn data_segments(&self, column: u32, row_group: u32) -> impl Iterator<Item = &Segment> {
        self.segments.iter().filter(move |s| {
            s.column == column && s.row_group == row_group && s.kind != SegmentKind::ZoneMap
        })
    }

    pub fn zone_map(&self, column: u32) -> Option<&Segment> {
        self.segments
            .iter()
            .find(|s| s.column == column && s.kind == SegmentKind::ZoneMap)
    }
}

/// A physical read after coalescing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoalescedRead {
    pub range: Range<u64>,
    /// Number of logical segment requests served by this read.
    pub segments: u32,
    /// Bytes requested by the segments themselves (excludes gap bytes read to coalesce).
    pub useful_bytes: u64,
}

/// Coalesce segment ranges the way `vortex-file`'s segment source does with a `CoalesceConfig`:
/// sort by offset and merge while the gap is at most `distance` and the merged read stays within
/// `max_size`. Overlapping and duplicate ranges merge too.
pub fn coalesce(mut ranges: Vec<Range<u64>>, config: Option<(u64, u64)>) -> Vec<CoalescedRead> {
    ranges.sort_by_key(|r| (r.start, r.end));
    let mut out: Vec<CoalescedRead> = Vec::new();
    for r in ranges {
        let len = r.end - r.start;
        if let (Some(last), Some((distance, max_size))) = (out.last_mut(), config) {
            let start = last.range.start;
            let end = last.range.end.max(r.end);
            if r.start <= last.range.end + distance && end - start <= max_size {
                last.range.end = end;
                last.segments += 1;
                last.useful_bytes += len;
                continue;
            }
        }
        out.push(CoalescedRead {
            range: r,
            segments: 1,
            useful_bytes: len,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_contiguous_and_sized() -> anyhow::Result<()> {
        let target = 20 << 20;
        let layout = FileLayout::generate("f".into(), target, 7)?;
        let mut expected = 0;
        for s in &layout.segments {
            assert_eq!(s.offset, expected);
            expected += s.len;
        }
        assert_eq!(expected, layout.footer_offset);
        assert!(layout.size <= target);
        assert!(layout.size > target - (4 << 20));

        let bytes = layout.bytes(7);
        assert_eq!(bytes.len() as u64, layout.size);
        assert_eq!(&bytes[bytes.len() - 4..], MAGIC);
        Ok(())
    }

    #[test]
    fn open_reads_cover_footer() -> anyhow::Result<()> {
        let layout = FileLayout::generate("f".into(), 20 << 20, 1)?;
        let (initial, rest) = layout.open_reads();
        assert_eq!(initial.end, layout.size);
        let start = rest.map_or(initial.start, |r| r.start);
        assert!(start <= layout.footer_offset);
        Ok(())
    }

    #[test]
    fn coalesce_merges_within_distance() {
        let reads = coalesce(vec![20..30, 0..10, 12..15, 100..110], Some((5, 50)));
        assert_eq!(
            reads,
            vec![
                CoalescedRead {
                    range: 0..30,
                    segments: 3,
                    useful_bytes: 23
                },
                CoalescedRead {
                    range: 100..110,
                    segments: 1,
                    useful_bytes: 10
                },
            ]
        );
    }

    #[test]
    fn coalesce_respects_max_size_and_overlap() {
        let reads = coalesce(vec![0..10, 5..15, 15..40], Some((0, 20)));
        assert_eq!(reads.len(), 2);
        assert_eq!(reads[0].range, 0..15);
        assert_eq!(reads[1].range, 15..40);
        assert_eq!(coalesce(vec![0..10, 10..20], None).len(), 2);
    }
}
