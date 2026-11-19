// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Column builders that the decoder can shorten again.
//!
//! A later occurrence of a field replaces the value of an earlier one, a oneof member clears the
//! other members, and a message that fails to decode must leave no values behind. So besides
//! appending, each builder can [`truncate`](Scalars::truncate) back to fewer rows. A builder
//! appends absent rows lazily: [`fill`](Scalars::fill) closes a gap of any length with a single
//! call, so the work is proportional to the values present rather than to the rows.

use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views_from_offsets;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::wire::ClosedEnum;
use crate::wire::ScalarKind;
use crate::wire::read_fixed32;
use crate::wire::read_fixed64;
use crate::wire::read_length_delimited;
use crate::wire::read_varint;
use crate::wire::varint_to_i32;
use crate::wire::varint_to_u32;
use crate::wire::zigzag_decode32;
use crate::wire::zigzag_decode64;

/// The builder of a column of scalar values, by storage type.
pub(super) enum Scalars {
    /// `f64` values.
    F64(Values<f64>),
    /// `f32` values.
    F32(Values<f32>),
    /// `i32` values.
    I32(Values<i32>),
    /// `i64` values.
    I64(Values<i64>),
    /// `u32` values.
    U32(Values<u32>),
    /// `u64` values.
    U64(Values<u64>),
    /// `bool` values.
    Bool(Bools),
    /// `utf8` or `binary` values.
    Binary(Binary),
}

impl Scalars {
    /// Creates the builder of a column of `kind` values.
    ///
    /// Absent rows of a non-nullable column hold the zero value of their type, except in an enum
    /// column, where they hold `enum_default`.
    pub(super) fn new(
        kind: ScalarKind,
        nullability: Nullability,
        enum_default: i32,
        capacity: usize,
    ) -> Self {
        match kind {
            ScalarKind::Double => Self::F64(Values::new(nullability, 0.0, capacity)),
            ScalarKind::Float => Self::F32(Values::new(nullability, 0.0, capacity)),
            ScalarKind::Int32 | ScalarKind::Sint32 | ScalarKind::Sfixed32 => {
                Self::I32(Values::new(nullability, 0, capacity))
            }
            ScalarKind::Enum => Self::I32(Values::new(nullability, enum_default, capacity)),
            ScalarKind::Int64 | ScalarKind::Sint64 | ScalarKind::Sfixed64 => {
                Self::I64(Values::new(nullability, 0, capacity))
            }
            ScalarKind::Uint32 | ScalarKind::Fixed32 => {
                Self::U32(Values::new(nullability, 0, capacity))
            }
            ScalarKind::Uint64 | ScalarKind::Fixed64 => {
                Self::U64(Values::new(nullability, 0, capacity))
            }
            ScalarKind::Bool => Self::Bool(Bools::new(nullability, capacity)),
            ScalarKind::String => Self::Binary(Binary::new(DType::Utf8(nullability), capacity)),
            ScalarKind::Bytes => Self::Binary(Binary::new(DType::Binary(nullability), capacity)),
        }
    }

    /// The number of rows.
    pub(super) fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::Binary(values) => values.len(),
        }
    }

    /// Appends absent rows until the column holds `len` rows.
    pub(super) fn fill(&mut self, len: usize) {
        match self {
            Self::F64(values) => values.fill(len),
            Self::F32(values) => values.fill(len),
            Self::I32(values) => values.fill(len),
            Self::I64(values) => values.fill(len),
            Self::U32(values) => values.fill(len),
            Self::U64(values) => values.fill(len),
            Self::Bool(values) => values.fill(len),
            Self::Binary(values) => values.fill(len),
        }
    }

    /// Removes the rows from `len` on, if the column holds more.
    pub(super) fn truncate(&mut self, len: usize) {
        match self {
            Self::F64(values) => values.truncate(len),
            Self::F32(values) => values.truncate(len),
            Self::I32(values) => values.truncate(len),
            Self::I64(values) => values.truncate(len),
            Self::U32(values) => values.truncate(len),
            Self::U64(values) => values.truncate(len),
            Self::Bool(values) => values.truncate(len),
            Self::Binary(values) => values.truncate(len),
        }
    }

    /// Fills the column to `len` rows and builds its array.
    pub(super) fn finish(self, len: usize) -> ArrayRef {
        match self {
            Self::F64(values) => values.finish(len),
            Self::F32(values) => values.finish(len),
            Self::I32(values) => values.finish(len),
            Self::I64(values) => values.finish(len),
            Self::U32(values) => values.finish(len),
            Self::U64(values) => values.finish(len),
            Self::Bool(values) => values.finish(len),
            Self::Binary(values) => values.finish(len),
        }
    }

    /// Compares the values at rows `a` and `b`, which must both be present.
    pub(super) fn compare(&self, a: usize, b: usize) -> Ordering {
        match self {
            Self::F64(values) => values.get(a).total_cmp(&values.get(b)),
            Self::F32(values) => values.get(a).total_cmp(&values.get(b)),
            Self::I32(values) => values.get(a).cmp(&values.get(b)),
            Self::I64(values) => values.get(a).cmp(&values.get(b)),
            Self::U32(values) => values.get(a).cmp(&values.get(b)),
            Self::U64(values) => values.get(a).cmp(&values.get(b)),
            Self::Bool(values) => values.get(a).cmp(&values.get(b)),
            Self::Binary(values) => values.get(a).cmp(values.get(b)),
        }
    }

    /// Reads values of `kind` from `input` and appends them.
    ///
    /// Reads one value, or with `packed` all the values until `input` is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` does not hold a valid value, or a string is not valid UTF-8.
    pub(super) fn append_from(
        &mut self,
        kind: ScalarKind,
        input: &mut &[u8],
        packed: bool,
    ) -> VortexResult<()> {
        match (self, kind) {
            (Self::F64(values), ScalarKind::Double) => values.append_from(input, packed, |input| {
                Ok(f64::from_bits(read_fixed64(input)?))
            }),
            (Self::F32(values), ScalarKind::Float) => values.append_from(input, packed, |input| {
                Ok(f32::from_bits(read_fixed32(input)?))
            }),
            (Self::I32(values), ScalarKind::Int32 | ScalarKind::Enum) => {
                values.append_from(input, packed, |input| {
                    Ok(varint_to_i32(read_varint(input)?))
                })
            }
            (Self::I32(values), ScalarKind::Sint32) => values.append_from(input, packed, |input| {
                Ok(zigzag_decode32(read_varint(input)?))
            }),
            (Self::I32(values), ScalarKind::Sfixed32) => values.append_from(
                input,
                packed,
                |input| Ok(read_fixed32(input)?.cast_signed()),
            ),
            (Self::I64(values), ScalarKind::Int64) => {
                values.append_from(input, packed, |input| Ok(read_varint(input)?.cast_signed()))
            }
            (Self::I64(values), ScalarKind::Sint64) => values.append_from(input, packed, |input| {
                Ok(zigzag_decode64(read_varint(input)?))
            }),
            (Self::I64(values), ScalarKind::Sfixed64) => values.append_from(
                input,
                packed,
                |input| Ok(read_fixed64(input)?.cast_signed()),
            ),
            (Self::U32(values), ScalarKind::Uint32) => values.append_from(input, packed, |input| {
                Ok(varint_to_u32(read_varint(input)?))
            }),
            (Self::U32(values), ScalarKind::Fixed32) => {
                values.append_from(input, packed, read_fixed32)
            }
            (Self::U64(values), ScalarKind::Uint64) => {
                values.append_from(input, packed, read_varint)
            }
            (Self::U64(values), ScalarKind::Fixed64) => {
                values.append_from(input, packed, read_fixed64)
            }
            (Self::Bool(values), ScalarKind::Bool) => values.append_from(input, packed),
            (Self::Binary(values), ScalarKind::String) => {
                values.push(read_utf8(read_length_delimited(input)?)?)
            }
            (Self::Binary(values), ScalarKind::Bytes) => values.push(read_length_delimited(input)?),
            (_, kind) => {
                vortex_panic!("A protobuf {kind:?} value does not match its storage column")
            }
        }
    }

    /// Reads a packed run of enum numbers from `input` and appends the numbers that the closed
    /// enum `closed` defines. The other numbers go to `rejected`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` does not hold valid varints.
    pub(super) fn append_closed_enums(
        &mut self,
        closed: &ClosedEnum,
        input: &mut &[u8],
        rejected: &mut Vec<i32>,
    ) -> VortexResult<()> {
        let Self::I32(values) = self else {
            vortex_panic!("Protobuf enum numbers are stored as i32");
        };
        read_values(input, true, |input| {
            let number = varint_to_i32(read_varint(input)?);
            if closed.contains(number) {
                values.push(number);
            } else {
                rejected.push(number);
            }
            Ok(())
        })
    }
}

/// Calls `read` once to read one value, or with `packed` until `input` is empty.
fn read_values(
    input: &mut &[u8],
    packed: bool,
    mut read: impl FnMut(&mut &[u8]) -> VortexResult<()>,
) -> VortexResult<()> {
    if packed {
        while !input.is_empty() {
            read(input)?;
        }
        Ok(())
    } else {
        read(input)
    }
}

/// Checks that a protobuf string is valid UTF-8, which the `utf8` storage dtype requires.
fn read_utf8(value: &[u8]) -> VortexResult<&[u8]> {
    if simdutf8::basic::from_utf8(value).is_err() {
        vortex_bail!("Invalid UTF-8 in a protobuf string field");
    }
    Ok(value)
}

/// Primitive values of one column, with a validity if the column is nullable.
pub(super) struct Values<T> {
    /// The values, which hold `default` at absent rows.
    values: BufferMut<T>,
    /// The validity, or `None` if the column is non-nullable.
    validity: Option<BitBufferMut>,
    /// The value of absent rows.
    default: T,
}

impl<T: NativePType> Values<T> {
    /// Creates an empty column whose absent rows hold `default`.
    pub(super) fn new(nullability: Nullability, default: T, capacity: usize) -> Self {
        Self {
            values: BufferMut::with_capacity(capacity),
            validity: new_validity(nullability, capacity),
            default,
        }
    }

    /// The number of rows.
    fn len(&self) -> usize {
        self.values.len()
    }

    /// The value at `index`.
    fn get(&self, index: usize) -> T {
        self.values[index]
    }

    /// Appends a present value.
    pub(super) fn push(&mut self, value: T) {
        self.values.push(value);
        if let Some(validity) = &mut self.validity {
            validity.append_true();
        }
    }

    /// Reads one value, or with `packed` all the values until `input` is empty, and appends them.
    fn append_from(
        &mut self,
        input: &mut &[u8],
        packed: bool,
        read: impl Fn(&mut &[u8]) -> VortexResult<T>,
    ) -> VortexResult<()> {
        read_values(input, packed, |input| {
            self.push(read(input)?);
            Ok(())
        })
    }

    /// Appends absent rows until the column holds `len` rows.
    pub(super) fn fill(&mut self, len: usize) {
        let missing = len.saturating_sub(self.len());
        if missing > 0 {
            self.values.push_n(self.default, missing);
            if let Some(validity) = &mut self.validity {
                validity.append_n(false, missing);
            }
        }
    }

    /// Removes the rows from `len` on, if the column holds more.
    pub(super) fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.values.truncate(len);
            if let Some(validity) = &mut self.validity {
                validity.truncate(len);
            }
        }
    }

    /// Fills the column to `len` rows and builds its array.
    pub(super) fn finish(mut self, len: usize) -> ArrayRef {
        self.fill(len);
        PrimitiveArray::new(self.values.freeze(), finish_validity(self.validity)).into_array()
    }
}

/// Boolean values of one column, with a validity if the column is nullable.
pub(super) struct Bools {
    /// The values, which are false at absent rows.
    values: BitBufferMut,
    /// The validity, or `None` if the column is non-nullable.
    validity: Option<BitBufferMut>,
}

impl Bools {
    /// Creates an empty column.
    fn new(nullability: Nullability, capacity: usize) -> Self {
        Self {
            values: BitBufferMut::with_capacity(capacity),
            validity: new_validity(nullability, capacity),
        }
    }

    /// The number of rows.
    fn len(&self) -> usize {
        self.values.len()
    }

    /// The value at `index`.
    fn get(&self, index: usize) -> bool {
        self.values.value(index)
    }

    /// Reads one varint, or with `packed` all the varints until `input` is empty, and appends
    /// them as booleans.
    fn append_from(&mut self, input: &mut &[u8], packed: bool) -> VortexResult<()> {
        read_values(input, packed, |input| {
            self.values.append(read_varint(input)? != 0);
            if let Some(validity) = &mut self.validity {
                validity.append_true();
            }
            Ok(())
        })
    }

    /// Appends absent rows until the column holds `len` rows.
    fn fill(&mut self, len: usize) {
        let missing = len.saturating_sub(self.len());
        if missing > 0 {
            self.values.append_n(false, missing);
            if let Some(validity) = &mut self.validity {
                validity.append_n(false, missing);
            }
        }
    }

    /// Removes the rows from `len` on, if the column holds more.
    fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.values.truncate(len);
            if let Some(validity) = &mut self.validity {
                validity.truncate(len);
            }
        }
    }

    /// Fills the column to `len` rows and builds its array.
    fn finish(mut self, len: usize) -> ArrayRef {
        self.fill(len);
        BoolArray::new(self.values.freeze(), finish_validity(self.validity)).into_array()
    }
}

/// Variable-length values of one `utf8` or `binary` column, with a validity if the column is
/// nullable.
pub(super) struct Binary {
    /// The storage dtype of the column.
    dtype: DType,
    /// The bytes of all values, back to back.
    bytes: BufferMut<u8>,
    /// The offset of each value into `bytes`, and the end of the last value.
    offsets: BufferMut<u64>,
    /// The validity, or `None` if the column is non-nullable.
    validity: Option<BitBufferMut>,
}

impl Binary {
    /// Creates an empty column of the `utf8` or `binary` dtype `dtype`.
    pub(super) fn new(dtype: DType, capacity: usize) -> Self {
        let mut offsets = BufferMut::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            validity: new_validity(dtype.nullability(), capacity),
            dtype,
            bytes: BufferMut::empty(),
            offsets,
        }
    }

    /// The number of rows.
    pub(super) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The value at `index`.
    pub(super) fn get(&self, index: usize) -> &[u8] {
        &self.bytes[as_usize(self.offsets[index])..as_usize(self.offsets[index + 1])]
    }

    /// Whether the row at `index` holds a value.
    pub(super) fn is_valid(&self, index: usize) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.value(index))
    }

    /// Appends `value` to the value at `row`, which becomes present, or is the next row.
    ///
    /// # Errors
    ///
    /// Returns an error if the value grows longer than a Vortex view can reference.
    pub(super) fn extend(&mut self, row: usize, value: &[u8]) -> VortexResult<()> {
        if row + 1 != self.len() {
            self.fill(row);
            return self.push(value);
        }
        debug_assert!(self.is_valid(row), "only a present value grows");
        let len = self.get(row).len() + value.len();
        if len > MAX_BUFFER_LEN {
            vortex_bail!(
                "A protobuf string or bytes value of {len} bytes exceeds the limit of {MAX_BUFFER_LEN}"
            );
        }
        self.bytes.extend_from_slice(value);
        self.offsets[row + 1] = self.bytes.len() as u64;
        Ok(())
    }

    /// Appends a present value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is longer than a Vortex view can reference.
    pub(super) fn push(&mut self, value: &[u8]) -> VortexResult<()> {
        if value.len() > MAX_BUFFER_LEN {
            vortex_bail!(
                "A protobuf string or bytes value of {} bytes exceeds the limit of {MAX_BUFFER_LEN}",
                value.len()
            );
        }
        self.bytes.extend_from_slice(value);
        self.offsets.push(self.bytes.len() as u64);
        if let Some(validity) = &mut self.validity {
            validity.append_true();
        }
        Ok(())
    }

    /// Appends absent rows, which hold empty values, until the column holds `len` rows.
    pub(super) fn fill(&mut self, len: usize) {
        let missing = len.saturating_sub(self.len());
        if missing > 0 {
            self.offsets.push_n(self.bytes.len() as u64, missing);
            if let Some(validity) = &mut self.validity {
                validity.append_n(false, missing);
            }
        }
    }

    /// Removes the rows from `len` on, if the column holds more.
    pub(super) fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.bytes.truncate(as_usize(self.offsets[len]));
            self.offsets.truncate(len + 1);
            if let Some(validity) = &mut self.validity {
                validity.truncate(len);
            }
        }
    }

    /// Fills the column to `len` rows and builds its array.
    pub(super) fn finish(mut self, len: usize) -> ArrayRef {
        self.fill(len);
        let (buffers, views) =
            build_views_from_offsets(0, MAX_BUFFER_LEN, self.bytes.freeze(), &self.offsets);
        // SAFETY: the views come from the offsets of values that were appended back to back. The
        // values of a `utf8` column are strings that `read_utf8` checked, or JSON text that the
        // decoder wrote from checked strings. The validity, if any, has one entry per value.
        unsafe {
            VarBinViewArray::new_unchecked(
                views,
                Arc::from(buffers),
                self.dtype,
                finish_validity(self.validity),
            )
        }
        .into_array()
    }
}

/// The offsets of a list column, whose last list stays open for more elements.
///
/// The elements of a list are appended while its row is decoded. A row that never receives an
/// element is an empty list.
pub(super) struct ListOffsets {
    /// The offsets of the closed lists into the elements, starting with zero.
    offsets: BufferMut<u64>,
    /// Whether the list after the closed ones is open, holding the elements after the last
    /// offset.
    open: bool,
}

impl ListOffsets {
    /// Creates the offsets of an empty list column.
    pub(super) fn new(capacity: usize) -> Self {
        let mut offsets = BufferMut::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            offsets,
            open: false,
        }
    }

    /// The number of closed lists.
    fn closed(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The number of rows, which includes the open list.
    pub(super) fn len(&self) -> usize {
        self.closed() + usize::from(self.open)
    }

    /// The offset of the first element of the open list.
    pub(super) fn open_start(&self) -> usize {
        as_usize(self.offsets[self.closed()])
    }

    /// The element range of the list at `row`, given the number of `elements` of all lists.
    pub(super) fn range(&self, row: usize, elements: usize) -> Range<usize> {
        let closed = self.closed();
        if row < closed {
            as_usize(self.offsets[row])..as_usize(self.offsets[row + 1])
        } else if row == closed && self.open {
            self.open_start()..elements
        } else {
            0..0
        }
    }

    /// Opens the list at `row` for the elements appended after the first `elements` ones.
    ///
    /// This closes the list that is open, and the rows between it and `row` become empty lists.
    /// `row` must not be before the open list.
    pub(super) fn open(&mut self, row: usize, elements: usize) {
        let closed = self.closed();
        if row > closed {
            self.offsets.push_n(elements as u64, row - closed);
        }
        self.open = true;
    }

    /// Closes all lists and appends empty lists until there are `len` rows, given the number of
    /// `elements` of all lists.
    fn fill(&mut self, len: usize, elements: usize) {
        let closed = self.closed();
        if len > closed {
            self.offsets.push_n(elements as u64, len - closed);
            self.open = false;
        }
    }

    /// Removes the rows from `len` on, if the column holds more.
    ///
    /// Returns the number of elements of the remaining lists if any rows were removed.
    pub(super) fn truncate(&mut self, len: usize) -> Option<usize> {
        if len >= self.len() {
            return None;
        }
        let elements = as_usize(self.offsets[len]);
        self.offsets.truncate(len + 1);
        self.open = false;
        Some(elements)
    }

    /// Fills the column to `len` rows and returns the offsets of its lists, given the number of
    /// `elements` of all lists.
    pub(super) fn finish(mut self, len: usize, elements: usize) -> Buffer<u64> {
        self.fill(len, elements);
        self.offsets.freeze()
    }
}

/// Creates the validity of an empty column, or `None` if the column is non-nullable.
pub(super) fn new_validity(nullability: Nullability, capacity: usize) -> Option<BitBufferMut> {
    nullability
        .is_nullable()
        .then(|| BitBufferMut::with_capacity(capacity))
}

/// Converts the validity of a finished column.
pub(super) fn finish_validity(validity: Option<BitBufferMut>) -> Validity {
    match validity {
        Some(validity) => Validity::from_bit_buffer(validity.freeze(), Nullability::Nullable),
        None => Validity::NonNullable,
    }
}

/// Converts an offset that indexes memory to `usize`.
pub(super) fn as_usize(offset: u64) -> usize {
    usize::try_from(offset).vortex_expect("offsets of values in memory fit in usize")
}
