// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validation of decimal wire nodes before either reader constructs an array.
//!
//! Both decimal readers use this boundary so malformed row counts and layouts fail before array
//! construction or execution can assert on them.

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::Decimal;
use super::DecimalMetadata;
use crate::ArrayParts;
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

        // NB: every later read takes the row count from the buffer, so a buffer of another length
        // must fail here. A read that returns another row count panics at the length assert of
        // the deserializer.
        let byte_len = len.checked_mul(values_type.byte_width()).ok_or_else(|| {
            vortex_err!(
                "Decimal node of {values_type} values must have a byte length that fits in usize, \
                 got length {len}"
            )
        })?;
        vortex_ensure!(
            values.len() == byte_len,
            "Decimal buffer of {len} {values_type} values must have {byte_len} bytes, got {}",
            values.len()
        );

        DecimalData::validate(&values, values_type)?;

        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;
        vortex_ensure_eq!(
            validity.nullability(),
            dtype.nullability(),
            "Decimal validity must match the nullability of the dtype",
        );

        Ok(Self {
            decimal_dtype,
            values_type,
            values,
            validity,
        })
    }

    /// Returns the parts of a [`DecimalArray`](super::DecimalArray) at the stored width, with the
    /// same buffer.
    pub(super) fn into_array_parts(self) -> VortexResult<ArrayParts<Decimal>> {
        let dtype = DType::Decimal(self.decimal_dtype, self.validity.nullability());
        let len = self.values.len() / self.values_type.byte_width();
        let slots = DecimalData::make_slots(&self.validity, len);
        let data = DecimalData::try_new_handle(self.values, self.values_type, self.decimal_dtype)?;

        Ok(ArrayParts::new(Decimal, dtype, len, data, slots))
    }
}
