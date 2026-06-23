// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use smallvec::SmallVec;

/// Per-column ordering options for row-oriented encoding.
///
/// A `RowSortFieldOptions` describes how one input column contributes to a row key. Descending order
/// reverses the encoded value bytes for that column. Null placement is controlled separately,
/// so nulls keep the requested position relative to non-null values in either direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowSortFieldOptions {
    /// If true, this column sorts in descending order.
    pub descending: bool,
    /// If true, nulls sort before non-null values.
    pub nulls_first: bool,
}

impl Default for RowSortFieldOptions {
    fn default() -> Self {
        Self::ascending()
    }
}

impl Display for RowSortFieldOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "descending={}, nulls_first={}",
            self.descending, self.nulls_first
        )
    }
}

impl RowSortFieldOptions {
    /// Construct a new `RowSortFieldOptions` with explicit options.
    pub const fn new(descending: bool, nulls_first: bool) -> Self {
        Self {
            descending,
            nulls_first,
        }
    }

    /// Construct an ascending field with nulls first.
    pub const fn ascending() -> Self {
        Self::new(false, true)
    }

    /// Construct a descending field with nulls first.
    pub const fn descending() -> Self {
        Self::new(true, true)
    }

    /// Return this field with nulls ordered before non-null values.
    pub const fn nulls_first(mut self) -> Self {
        self.nulls_first = true;
        self
    }

    /// Return this field with nulls ordered after non-null values.
    pub const fn nulls_last(mut self) -> Self {
        self.nulls_first = false;
        self
    }
}

const FIELDS_INLINE: usize = 4;

/// Ordering options for row-oriented encoding.
///
/// The options contain one [`RowSortFieldOptions`] per input column, in the same order as the columns
/// passed to [`convert_columns`](crate::convert_columns),
/// [`compute_row_sizes`](crate::compute_row_sizes), [`RowSize`](crate::RowSize), or
/// [`RowEncode`](crate::RowEncode).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowEncodingOptions {
    pub(crate) fields: SmallVec<[RowSortFieldOptions; FIELDS_INLINE]>,
}

impl RowEncodingOptions {
    /// Construct a new `RowEncodingOptions` from any iterator of [`RowSortFieldOptions`]s.
    pub fn new(fields: impl IntoIterator<Item = RowSortFieldOptions>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    /// Construct default ascending, nulls-first options for `column_count` input columns.
    pub fn default_for_columns(column_count: usize) -> Self {
        Self::new(std::iter::repeat_n(
            RowSortFieldOptions::default(),
            column_count,
        ))
    }

    /// Borrow the per-column sort fields.
    pub fn fields(&self) -> &[RowSortFieldOptions] {
        &self.fields
    }

    /// Return the number of input columns described by these options.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Return true when the options do not describe any input columns.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl FromIterator<RowSortFieldOptions> for RowEncodingOptions {
    fn from_iter<T: IntoIterator<Item = RowSortFieldOptions>>(iter: T) -> Self {
        Self::new(iter)
    }
}

impl Display for RowEncodingOptions {
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

/// Serialize a [`RowEncodingOptions`] to a compact byte vector: 4-byte LE length followed by
/// `2 * len` bytes (descending + nulls_first booleans for each field).
pub(crate) fn serialize_row_encoding_options(opts: &RowEncodingOptions) -> Vec<u8> {
    use vortex_error::VortexExpect;
    let n =
        u32::try_from(opts.fields.len()).vortex_expect("RowEncodingOptions length must fit in u32");
    let mut out = Vec::with_capacity(4 + 2 * opts.fields.len());
    out.extend_from_slice(&n.to_le_bytes());
    for f in &opts.fields {
        out.push(u8::from(f.descending));
        out.push(u8::from(f.nulls_first));
    }
    out
}

/// Deserialize a [`RowEncodingOptions`] produced by [`serialize_row_encoding_options`].
pub(crate) fn deserialize_row_encoding_options(
    bytes: &[u8],
) -> vortex_error::VortexResult<RowEncodingOptions> {
    if bytes.len() < 4 {
        vortex_error::vortex_bail!(
            "RowEncodingOptions metadata must contain a 4-byte length prefix"
        );
    }
    let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let expected = 4 + 2 * n;
    if bytes.len() != expected {
        vortex_error::vortex_bail!(
            "RowEncodingOptions metadata wrong size: got {}, expected {}",
            bytes.len(),
            expected
        );
    }
    let mut fields: SmallVec<[RowSortFieldOptions; FIELDS_INLINE]> = SmallVec::with_capacity(n);
    let mut i = 4;
    for _ in 0..n {
        fields.push(RowSortFieldOptions {
            descending: bytes[i] != 0,
            nulls_first: bytes[i + 1] != 0,
        });
        i += 2;
    }
    Ok(RowEncodingOptions { fields })
}
