// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The column of a well-known message type that the mapping stores in a dedicated form.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::builders::Binary;
use super::builders::Scalars;
use super::builders::Values;
use crate::well_known::WellKnown;
use crate::well_known::decode_json;
use crate::well_known::decode_timestamp;
use crate::well_known::wrapper_value;

/// The encoding of a scalar zero of any wire type: a zero varint, four or eight zero bytes, or
/// an empty length-delimited value.
const ZERO_VALUE: [u8; 8] = [0; 8];

/// The column of well-known messages.
///
/// A message field can occur several times in one message, and its occurrences merge. So the
/// column first collects the encoding of each message, its occurrences one after the other, and
/// converts it once the message that holds the field is complete. A message that cannot be
/// converted then fails as a whole.
pub(super) struct WellKnownColumn {
    /// The well-known type.
    well_known: WellKnown,
    /// The nullability of the column.
    nullability: Nullability,
    /// The encoding of the message at each row.
    encoded: Binary,
    /// The converted values of the rows up to the last converted one.
    values: Converted,
    /// A buffer for the JSON text of one value.
    text: Vec<u8>,
}

/// The converted values of a [`WellKnownColumn`].
enum Converted {
    /// The wrapped values of wrapper messages.
    Wrapper(Scalars),
    /// The nanoseconds of timestamps.
    Timestamp(Values<i64>),
    /// The JSON text of JSON messages.
    Json(Binary),
}

impl WellKnownColumn {
    /// Creates the column of `well_known` messages.
    pub(super) fn new(well_known: WellKnown, nullability: Nullability, capacity: usize) -> Self {
        let values = match well_known {
            WellKnown::Wrapper(kind) => {
                Converted::Wrapper(Scalars::new(kind, nullability, 0, capacity))
            }
            WellKnown::Timestamp => Converted::Timestamp(Values::new(nullability, 0, capacity)),
            WellKnown::Json(_) => Converted::Json(Binary::new(DType::Utf8(nullability), capacity)),
        };
        Self {
            well_known,
            nullability,
            encoded: Binary::new(DType::Binary(nullability), capacity),
            values,
            text: Vec::new(),
        }
    }

    /// The number of rows.
    pub(super) fn len(&self) -> usize {
        self.encoded.len()
    }

    /// Appends one occurrence of an encoded message to the message at `row`, which is the last
    /// row or the one after it.
    pub(super) fn extend(&mut self, row: usize, encoded: &[u8]) -> VortexResult<()> {
        self.encoded.extend(row, encoded)
    }

    /// Appends absent rows until the column holds `len` rows.
    ///
    /// The absent rows of a non-nullable column hold empty messages, which convert to the default
    /// value of the type.
    pub(super) fn fill(&mut self, len: usize) {
        self.encoded.fill(len);
    }

    /// Removes the rows from `len` on, if the column holds more.
    pub(super) fn truncate(&mut self, len: usize) {
        self.encoded.truncate(len);
        match &mut self.values {
            Converted::Wrapper(values) => values.truncate(len),
            Converted::Timestamp(values) => values.truncate(len),
            Converted::Json(values) => values.truncate(len),
        }
    }

    /// Converts the message at `row`, if it is present.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoding is invalid, or the message has no stored form.
    pub(super) fn convert(&mut self, row: usize) -> VortexResult<()> {
        if row >= self.encoded.len() || !self.encoded.is_valid(row) {
            return Ok(());
        }
        let encoded = self.encoded.get(row);
        match (&mut self.values, self.well_known) {
            (Converted::Wrapper(values), WellKnown::Wrapper(kind)) => {
                // A wrapper without a value holds the default value, which is zero.
                let mut value = wrapper_value(kind, encoded)?.unwrap_or(&ZERO_VALUE);
                values.fill(row);
                values.append_from(kind, &mut value, false)?;
            }
            (Converted::Timestamp(values), _) => {
                let nanos = decode_timestamp(encoded)?;
                values.fill(row);
                values.push(nanos);
            }
            (Converted::Json(values), WellKnown::Json(json_type)) => {
                self.text.clear();
                decode_json(json_type, encoded, &mut self.text)?;
                values.fill(row);
                values.push(&self.text)?;
            }
            (_, well_known) => {
                vortex_panic!("{well_known:?} values do not match their column")
            }
        }
        Ok(())
    }

    /// Fills the column to `len` rows and builds its array.
    pub(super) fn finish(self, len: usize) -> VortexResult<ArrayRef> {
        let storage = match self.values {
            Converted::Wrapper(values) => return Ok(values.finish(len)),
            Converted::Timestamp(values) => values.finish(len),
            Converted::Json(values) => values.finish(len),
        };
        Ok(
            ExtensionArray::try_new(self.well_known.ext_dtype(self.nullability), storage)?
                .into_array(),
        )
    }
}
