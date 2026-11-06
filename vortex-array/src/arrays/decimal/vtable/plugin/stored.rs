// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validation of a `vortex.decimal` node before the [`Decimal`] plugin constructs an array in
//! memory.
//!
//! A malformed row count or buffer layout returns an error here instead of failing an assert
//! later.
//!
//! [`Decimal`]: super::Decimal

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::DecimalMetadata;
use crate::ArrayParts;
use crate::arrays::Decimal;
use crate::arrays::decimal::DecimalData;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A `vortex.decimal` node as it is stored, with its parts checked against its dtype and length.
///
/// The wire metadata records the value width independently of the logical precision.
pub(super) struct StoredDecimal {
    /// The logical precision and scale supplied by the containing format.
    pub(super) decimal_dtype: DecimalDType,
    /// The stored width of `values`.
    pub(super) values_type: DecimalType,
    /// The row count supplied by the containing format, which `values` and `validity` match.
    len: usize,
    /// One `values_type` integer per row, aligned to its width, in host or device memory.
    pub(super) values: BufferHandle,
    /// The validity, with one entry per row and the nullability of the dtype.
    pub(super) validity: Validity,
}

impl StoredDecimal {
    /// Decodes the parts of a node with `len` rows of the decimal dtype `dtype`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dtype` is not a decimal dtype, if the metadata names no known width,
    /// if the node does not have exactly one buffer, if the buffer does not hold `len` values of
    /// the stored width or is not aligned to it, or if the validity children do not match the
    /// nullability of `dtype`.
    pub(super) fn deserialize(
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self> {
        let Some(&decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("Decimal node requires a decimal dtype, got {dtype}");
        };
        let values_type = DecimalMetadata::decode(metadata)?.get_values_type()?;
        let values = fixed_width::single_buffer(buffers)?;

        // NB: a legacy read takes the row count from the buffer, so a buffer of another length
        // must fail here. A read that returns another row count panics at the length assert of
        // the deserializer.
        let byte_len = len.checked_mul(values_type.byte_width()).ok_or_else(|| {
            vortex_err!(
                "Decimal node of {values_type} values must have a byte length that fits in usize, \
                 got length {len}"
            )
        })?;
        vortex_ensure_eq!(
            values.len(),
            byte_len,
            "Decimal buffer must have the byte length of {len} {values_type} values",
        );

        DecimalData::validate(&values, values_type)?;

        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;

        Ok(Self {
            decimal_dtype,
            values_type,
            len,
            values,
            validity,
        })
    }

    /// Returns whether this is a legacy node, whose stored width is not the canonical width of its
    /// precision.
    ///
    /// Writers used to store a decimal at any width that held its values. A decimal array must now
    /// use the canonical width of its precision, and a smaller width must be stored as a [`Widen`]
    /// over a lower-precision child.
    ///
    /// [`Widen`]: crate::arrays::Widen
    pub(super) fn is_legacy(&self) -> bool {
        self.values_type != self.decimal_dtype.canonical_values_type()
    }

    /// Returns the parts of a [`DecimalArray`](crate::arrays::DecimalArray) at the stored width,
    /// with the same buffer.
    pub(super) fn into_array_parts(self) -> VortexResult<ArrayParts<Decimal>> {
        let dtype = DType::Decimal(self.decimal_dtype, self.validity.nullability());
        let slots = DecimalData::make_slots(&self.validity, self.len);
        let data = DecimalData::try_new_handle(self.values, self.values_type, self.decimal_dtype)?;

        Ok(ArrayParts::new(Decimal, dtype, self.len, data, slots))
    }
}
