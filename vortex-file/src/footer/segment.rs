// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_flatbuffers::footer as fb;

/// The location of a segment within a Vortex file.
///
/// A segment is a contiguous block of bytes in a file that contains a part of the file's data.
/// The `SegmentSpec` struct specifies the location and properties of a segment.
#[derive(Clone, Copy, Debug)]
pub struct SegmentSpec {
    /// The byte offset of the segment from the start of the file.
    pub offset: u64,
    /// The length of the segment in bytes.
    pub length: u32,
    /// The memory alignment requirement of the segment.
    pub alignment: Alignment,
}

impl SegmentSpec {
    /// Returns the byte range of the segment within the file.
    ///
    /// The range starts at the segment's offset and extends for its length.
    pub fn byte_range(&self) -> Range<u64> {
        self.offset..self.offset + u64::from(self.length)
    }
}

impl From<&SegmentSpec> for fb::SegmentSpec {
    fn from(value: &SegmentSpec) -> Self {
        fb::SegmentSpec::new(value.offset, value.length, value.alignment.exponent(), 0, 0)
    }
}

impl TryFrom<&fb::SegmentSpec> for SegmentSpec {
    type Error = VortexError;

    fn try_from(value: &fb::SegmentSpec) -> Result<Self, Self::Error> {
        Ok(Self {
            offset: value.offset(),
            length: value.length(),
            // The alignment exponent comes from the file and may be corrupt, so validate it rather
            // than panicking on a too-large shift (see issue #8819).
            alignment: Alignment::try_from_untrusted_exponent(value.alignment_exponent())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_out_of_range_alignment_exponent() {
        // A fuzzed segment spec can declare an alignment exponent that would overflow a usize
        // shift. Parsing it must return an error rather than panicking (see issue #8819).
        let fb_spec = fb::SegmentSpec::new(0, 0, u8::MAX, 0, 0);
        let err = SegmentSpec::try_from(&fb_spec).unwrap_err();
        assert!(err.to_string().contains("too large"), "{err}");
    }

    #[test]
    fn rejects_excessive_alignment_exponent() {
        // A representable alignment can still cause an unreasonable allocation when a segment is
        // copied to satisfy it.
        let fb_spec = fb::SegmentSpec::new(0, 0, 17, 0, 0);
        let err = SegmentSpec::try_from(&fb_spec).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
    }
}
