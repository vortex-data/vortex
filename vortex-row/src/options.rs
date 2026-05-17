// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use smallvec::SmallVec;

/// Per-column options for the row-oriented byte encoder.
///
/// These options control how a single column is encoded into row bytes:
/// - `descending`: if true, the encoded value bytes are bit-inverted so that
///   lexicographic byte comparison reflects the reverse of the natural ordering.
///   The null sentinel byte is NOT inverted, so nulls keep their requested
///   position relative to non-nulls.
/// - `nulls_first`: if true, nulls sort before non-nulls. If false, nulls sort
///   after non-nulls. Implemented via the sentinel byte that precedes every
///   value's encoded bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SortField {
    /// If true, encoded value bytes are bit-inverted so lexicographic byte
    /// comparison reflects the reverse of the natural ordering.
    pub descending: bool,
    /// If true, nulls sort before non-null values; otherwise nulls sort after.
    pub nulls_first: bool,
}

impl Default for SortField {
    fn default() -> Self {
        Self {
            descending: false,
            nulls_first: true,
        }
    }
}

impl Display for SortField {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "descending={}, nulls_first={}",
            self.descending, self.nulls_first
        )
    }
}

impl SortField {
    /// Construct a new `SortField` with explicit options.
    pub fn new(descending: bool, nulls_first: bool) -> Self {
        Self {
            descending,
            nulls_first,
        }
    }

    /// Returns the sentinel byte to write for a non-null value.
    #[inline]
    pub fn non_null_sentinel(&self) -> u8 {
        // Non-null is always 0x01. Null choices are < or > 0x01.
        0x01
    }

    /// Returns the sentinel byte to write for a null value.
    #[inline]
    pub fn null_sentinel(&self) -> u8 {
        if self.nulls_first {
            // Nulls before non-nulls (smaller byte sorts first).
            0x00
        } else {
            // Nulls after non-nulls (larger byte sorts later).
            0x02
        }
    }
}

/// Inline capacity for [`RowEncodeOptions::fields`]. Up to this many [`SortField`]s
/// are held inline without a heap allocation; beyond, the storage spills.
pub const FIELDS_INLINE: usize = 4;

/// Options for the variadic [`RowSize`] and [`RowEncode`] scalar functions:
/// one [`SortField`] per input column.
///
/// Stored in a [`SmallVec`] so that typical 1–4 column keys avoid a heap
/// allocation; longer field lists spill to the heap transparently.
///
/// [`RowSize`]: super::size::RowSize
/// [`RowEncode`]: super::encode::RowEncode
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowEncodeOptions {
    /// Per-column sort fields, in left-to-right column order.
    pub fields: SmallVec<[SortField; FIELDS_INLINE]>,
}

impl RowEncodeOptions {
    /// Construct a new `RowEncodeOptions` from any iterator of [`SortField`]s.
    pub fn new(fields: impl IntoIterator<Item = SortField>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }
}

impl Display for RowEncodeOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, field) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", field)?;
        }
        write!(f, "]")
    }
}

/// Serialize a [`RowEncodeOptions`] to a compact byte vector: 4-byte LE length followed by
/// `2 * len` bytes (descending + nulls_first booleans for each field).
pub(crate) fn serialize_row_encode_options(opts: &RowEncodeOptions) -> Vec<u8> {
    use vortex_error::VortexExpect;
    let n =
        u32::try_from(opts.fields.len()).vortex_expect("RowEncodeOptions length must fit in u32");
    let mut out = Vec::with_capacity(4 + 2 * opts.fields.len());
    out.extend_from_slice(&n.to_le_bytes());
    for f in &opts.fields {
        out.push(u8::from(f.descending));
        out.push(u8::from(f.nulls_first));
    }
    out
}

/// Deserialize a [`RowEncodeOptions`] produced by [`serialize_row_encode_options`].
pub(crate) fn deserialize_row_encode_options(
    bytes: &[u8],
) -> vortex_error::VortexResult<RowEncodeOptions> {
    if bytes.len() < 4 {
        vortex_error::vortex_bail!("RowEncodeOptions metadata must contain a 4-byte length prefix");
    }
    let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let expected = 4 + 2 * n;
    if bytes.len() != expected {
        vortex_error::vortex_bail!(
            "RowEncodeOptions metadata wrong size: got {}, expected {}",
            bytes.len(),
            expected
        );
    }
    let mut fields: SmallVec<[SortField; FIELDS_INLINE]> = SmallVec::with_capacity(n);
    let mut i = 4;
    for _ in 0..n {
        fields.push(SortField {
            descending: bytes[i] != 0,
            nulls_first: bytes[i + 1] != 0,
        });
        i += 2;
    }
    Ok(RowEncodeOptions { fields })
}
