// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_buffer::Alignment;

/// The block size assumed by [`SegmentPadding::block_aligned`], matching the direct-I/O and page
/// granularity of every mainstream filesystem we target.
pub const DEFAULT_BLOCK_SIZE: Alignment = Alignment::new(4096);

/// The default overhead budget for [`SegmentPadding::proportional`], bounding the padding the
/// writer may add to at most `1/64` (1.6%) of the segment bytes written.
pub const DEFAULT_MAX_OVERHEAD_RATIO: u32 = 64;

/// Environment variable selecting the default [`SegmentPadding`] policy.
pub const SEGMENT_PADDING_ENV_VAR: &str = "VORTEX_SEGMENT_PADDING";

static PADDING_FROM_ENV: LazyLock<SegmentPadding> = LazyLock::new(|| {
    let Ok(value) = std::env::var(SEGMENT_PADDING_ENV_VAR) else {
        return SegmentPadding::None;
    };
    // An empty value reads as unset, so a workflow can template the variable in unconditionally.
    if value.trim().is_empty() {
        return SegmentPadding::None;
    }
    SegmentPadding::parse(&value).unwrap_or_else(|| {
        tracing::warn!(
            "ignoring unrecognised {SEGMENT_PADDING_ENV_VAR}={value}, \
             expected none, always, grouped, or proportional[:ratio]"
        );
        SegmentPadding::None
    })
});

/// How the file writer positions segments relative to storage block boundaries.
///
/// A segment is always padded enough to satisfy its own memory alignment, so that a reader can
/// hand the bytes straight to a typed array without copying. This policy controls whether the
/// writer pads *further*, up to a storage block boundary.
///
/// ## Why this is a trade-off
///
/// Block alignment does not make direct I/O possible — a reader can always widen an unaligned
/// request out to the enclosing blocks and slice the result — it only removes the widening. That
/// saves at most one block of over-read per *physical* read, while padding costs up to one block
/// per *segment*. Because reads are coalesced across many segments, aligning every segment
/// usually costs far more bytes than it saves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SegmentPadding {
    /// Pack segments contiguously, padding only as far as each segment's own alignment requires.
    #[default]
    None,
    /// Start every segment on a `block` boundary.
    ///
    /// Costs on average half a block per segment, regardless of how small the segment is.
    Always {
        /// The storage block size to align to.
        block: Alignment,
    },
    /// Keep every sub-block segment inside a single `block`, packing runs of them together.
    ///
    /// A segment smaller than `block` stays where it is when it fits in what is left of the
    /// current block, and moves to the next block only when it would straddle one. Runs of small
    /// consecutive segments therefore share a block instead of each burning a whole one, and each
    /// still reads in the single block [`Self::Always`] would have given it.
    ///
    /// A segment at least a block long is never moved. It already spans several blocks, so the
    /// one extra block a straddle can cost it is a small fraction of its read, and far cheaper
    /// than the up-to-a-block of padding that aligning it would add to the file.
    Grouped {
        /// The storage block size to align to.
        block: Alignment,
    },
    /// Start a segment on a `block` boundary only when the padding is worth it.
    ///
    /// A segment is aligned when its padding is at most `1/max_overhead_ratio` of the segment's
    /// own length, which bounds the padding added across the whole file to `1/max_overhead_ratio`
    /// of the segment bytes written. Large segments — the ones that dominate bytes read — are
    /// nearly always aligned, while small segments are packed contiguously as before.
    Proportional {
        /// The storage block size to align to.
        block: Alignment,
        /// The reciprocal of the padding budget. Larger values pad less.
        max_overhead_ratio: u32,
    },
}

impl SegmentPadding {
    /// Start every segment on a [`DEFAULT_BLOCK_SIZE`] boundary.
    pub const fn block_aligned() -> Self {
        Self::Always {
            block: DEFAULT_BLOCK_SIZE,
        }
    }

    /// Keep every segment below [`DEFAULT_BLOCK_SIZE`] within a single block of that size.
    pub const fn grouped() -> Self {
        Self::Grouped {
            block: DEFAULT_BLOCK_SIZE,
        }
    }

    /// Block-align segments within the default [`DEFAULT_MAX_OVERHEAD_RATIO`] padding budget.
    pub const fn proportional() -> Self {
        Self::Proportional {
            block: DEFAULT_BLOCK_SIZE,
            max_overhead_ratio: DEFAULT_MAX_OVERHEAD_RATIO,
        }
    }

    /// The policy named by `VORTEX_SEGMENT_PADDING`, or [`Self::None`] if it is unset.
    ///
    /// Accepts `none`, `always`, `grouped`, and `proportional[:ratio]`, all at the
    /// [`DEFAULT_BLOCK_SIZE`] block size, so a deployment or benchmark can switch policies
    /// without recompiling. Other block sizes are reachable only through the variants.
    pub fn from_env() -> Self {
        *PADDING_FROM_ENV
    }

    fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        let (name, ratio) = match value.split_once(':') {
            Some((name, ratio)) => (name, Some(ratio.parse().ok()?)),
            None => (value, None),
        };
        match (name, ratio) {
            ("none", None) => Some(Self::None),
            ("always", None) => Some(Self::block_aligned()),
            ("grouped", None) => Some(Self::grouped()),
            ("proportional", None) => Some(Self::proportional()),
            // A zero budget would pad everything, which `always` already says more clearly.
            ("proportional", Some(ratio)) if ratio > 0 => Some(Self::Proportional {
                block: DEFAULT_BLOCK_SIZE,
                max_overhead_ratio: ratio,
            }),
            _ => None,
        }
    }

    /// The number of padding bytes to insert before a segment of `length` bytes requiring
    /// `alignment`, when the writer is positioned at `byte_offset`.
    pub fn padding(self, byte_offset: u64, length: u64, alignment: Alignment) -> u64 {
        let required = pad_to(byte_offset, alignment);
        match self {
            Self::None => required,
            // A segment whose own alignment exceeds the block size still has to satisfy it, so
            // align to whichever is larger. Both are powers of two, so the larger subsumes both.
            Self::Always { block } => pad_to(byte_offset, block.max(alignment)),
            // Only a sub-block segment is worth moving: straddling a boundary doubles the blocks
            // it touches, from one to two. A segment at least a block long already spans several,
            // so the extra block a straddle costs it is a small fraction, not worth a whole block
            // of padding. Within that, a segment that fits in the current block's remainder rides
            // along for free.
            Self::Grouped { block } => {
                let block_size = block.as_usize() as u64;
                let offset_in_block = (byte_offset + required) % block_size;
                if length >= block_size || offset_in_block + length <= block_size {
                    required
                } else {
                    pad_to(byte_offset, block.max(alignment))
                }
            }
            Self::Proportional {
                block,
                max_overhead_ratio,
            } => {
                let aligned = pad_to(byte_offset, block.max(alignment));
                if aligned.saturating_mul(u64::from(max_overhead_ratio)) <= length {
                    aligned
                } else {
                    required
                }
            }
        }
    }
}

fn pad_to(byte_offset: u64, alignment: Alignment) -> u64 {
    byte_offset.next_multiple_of(alignment.as_usize() as u64) - byte_offset
}

#[cfg(test)]
mod tests;
