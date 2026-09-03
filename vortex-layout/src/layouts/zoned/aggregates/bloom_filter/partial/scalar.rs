// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar helpers for Split Block Bloom Filters (SBBF).
//!
//! The core Bloom filter operates on bytes. This module provides scalar-aware
//! insertion and membership helpers, including validation and conversion from
//! [`Scalar`] values to the bytes used for hashing.

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_float_ptype;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::BloomPartial;

/// The following implementation provides a simpler access for scalars.
impl BloomPartial {
    /// Returns the hash of the scalar's underlying value.
    /// Returns an error if the scalar's [DType] is unsupported.
    ///
    /// Will panic if the scalar is invalid.
    ///
    /// For example, `Scalar(Primitive(I32(54)))` is hashed as `hash(54)`.
    fn hash_scalar(&self, scalar: &Scalar) -> VortexResult<u64> {
        Ok(match scalar.dtype() {
            DType::Extension(_) => self.hash_scalar(&scalar.as_extension().to_storage_scalar())?,
            DType::Bool(_) => self.hash([u8::from(
                scalar
                    .as_bool()
                    .value()
                    .vortex_expect("non-null boolean value"),
            )]),
            DType::Primitive(ptype, _) => match ptype {
                PType::F16 | PType::F32 | PType::F64 => {
                    match_each_float_ptype!(ptype, |T| {
                        let value = scalar
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("non-null primitive value");
                        self.hash(value.to_le_bytes())
                    })
                }
                _ => match_each_integer_ptype!(ptype, |T| {
                    let value = scalar
                        .as_primitive()
                        .typed_value::<T>()
                        .vortex_expect("non-null primitive value");
                    self.hash(value.to_le_bytes())
                }),
            },
            DType::Utf8(_) => {
                let buffer = scalar
                    .as_utf8()
                    .value()
                    .vortex_expect("non-null utf8 value");
                self.hash(buffer.as_bytes())
            }
            DType::Binary(_) => {
                let buffer = scalar
                    .as_binary()
                    .value()
                    .vortex_expect("non-null binary value");
                self.hash(buffer.as_slice())
            }
            other => {
                return Err(vortex_err!(
                    "Unsupported scalar type for bloom filter: {other}"
                ));
            }
        })
    }

    /// Returns `true` if the underlying value of a [Scalar] might be present in the filter.
    ///
    /// A `false` result guarantees that the value is absent. A `true` result may
    /// be a false positive.
    ///
    /// For invalid values, it always returns `false`.
    ///
    /// Returns an error if the scalar [DType] is unsupported.
    pub(in crate::layouts::zoned) fn contains_scalar(&self, scalar: &Scalar) -> VortexResult<bool> {
        if scalar.is_null() {
            return Ok(false);
        }

        let hash = self.hash_scalar(scalar)?;
        Ok(self.find_hash(hash))
    }

    /// Inserts the underlying value of a [Scalar] if it is valid.
    /// If the value is invalid, it just skips its insertion.
    ///
    /// Returns an error if the scalar [DType] is unsupported.
    pub(in crate::layouts::zoned) fn insert_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        if scalar.is_null() {
            return Ok(());
        }

        let hash = self.hash_scalar(scalar)?;
        self.add_hash(hash);

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;

    use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;

    #[should_panic]
    #[test]
    fn hash_invalid_scalar_panics() {
        let bloom = BloomPartial::from(&BloomOptions::default());
        bloom
            .hash_scalar(&Scalar::null(DType::Bool(Nullability::Nullable)))
            .expect("to panic");
    }
}
