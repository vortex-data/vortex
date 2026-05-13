// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl Scalar {
    /// Validate that the given [`ScalarValue`] is compatible with the given [`DType`].
    pub fn validate(dtype: &DType, value: Option<&ScalarValue>) -> VortexResult<()> {
        let Some(value) = value else {
            vortex_ensure!(
                dtype.is_nullable(),
                "non-nullable dtype {dtype} cannot hold a null value",
            );
            return Ok(());
        };

        // From here onwards, we know that the value is not null.
        match dtype {
            DType::Null => {
                vortex_bail!("null dtype cannot hold a non-null value {value}");
            }
            DType::Bool(_) => {
                vortex_ensure!(
                    matches!(value, ScalarValue::Bool(_)),
                    "bool dtype expected Bool value, got {value}",
                );
            }
            DType::Primitive(ptype, _) => {
                let ScalarValue::Primitive(pvalue) = value else {
                    vortex_bail!("primitive dtype {ptype} expected Primitive value, got {value}",);
                };

                // Note that this is a backwards compatibility check for poor design in the
                // previous implementation. `f16` `ScalarValue`s used to be serialized as
                // `pb::ScalarValue::Uint64Value(v.to_bits() as u64)`, so we need to ensure
                // that we can still represent them as such.
                let f16_backcompat_still_works =
                    matches!(ptype, &PType::F16) && matches!(pvalue, PValue::U64(_));

                vortex_ensure!(
                    f16_backcompat_still_works || pvalue.ptype() == *ptype,
                    "primitive dtype {ptype} is not compatible with value {pvalue}",
                );
            }
            DType::Decimal(dec_dtype, _) => {
                let ScalarValue::Decimal(dvalue) = value else {
                    vortex_bail!("decimal dtype expected Decimal value, got {value}");
                };

                vortex_ensure!(
                    dvalue.fits_in_precision(*dec_dtype),
                    "decimal value {dvalue} does not fit in precision of {dec_dtype}",
                );
            }
            DType::Utf8(_) => {
                vortex_ensure!(
                    matches!(value, ScalarValue::Utf8(_)),
                    "utf8 dtype expected Utf8 value, got {value}",
                );
            }
            DType::Binary(_) => {
                vortex_ensure!(
                    matches!(value, ScalarValue::Binary(_)),
                    "binary dtype expected Binary value, got {value}",
                );
            }
            DType::List(elem_dtype, _) => {
                let ScalarValue::Tuple(elements) = value else {
                    vortex_bail!("list dtype expected Tuple value, got {value}");
                };

                for (i, element) in elements.iter().enumerate() {
                    Self::validate(elem_dtype.as_ref(), element.as_ref())
                        .map_err(|e| vortex_error::vortex_err!("list element at index {i}: {e}"))?;
                }
            }
            DType::FixedSizeList(elem_dtype, size, _) => {
                let ScalarValue::Tuple(elements) = value else {
                    vortex_bail!("fixed-size list dtype expected Tuple value, got {value}",);
                };

                let len = elements.len();
                vortex_ensure_eq!(
                    len,
                    *size as usize,
                    "fixed-size list dtype expected {size} elements, got {len}",
                );

                for (i, element) in elements.iter().enumerate() {
                    Self::validate(elem_dtype.as_ref(), element.as_ref()).map_err(|e| {
                        vortex_error::vortex_err!("fixed-size list element at index {i}: {e}",)
                    })?;
                }
            }
            DType::Struct(fields, _) => {
                let ScalarValue::Tuple(values) = value else {
                    vortex_bail!("struct dtype expected Tuple value, got {value}");
                };

                let nfields = fields.nfields();
                let nvalues = values.len();
                vortex_ensure_eq!(
                    nvalues,
                    nfields,
                    "struct dtype expected {nfields} fields, got {nvalues}",
                );

                for (field, field_value) in fields.fields().zip(values.iter()) {
                    Self::validate(&field, field_value.as_ref())?;
                }
            }
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::Extension(ext_dtype) => ext_dtype.validate_storage_value(value)?,
            DType::Variant(_) => {
                let ScalarValue::Variant(inner) = value else {
                    vortex_bail!("variant dtype expected Variant value, got {value}");
                };

                Self::validate(inner.dtype(), inner.value())?;
                vortex_ensure!(
                    !inner.is_null() || matches!(inner.dtype(), DType::Null),
                    "variant nulls must use a nested null scalar, got {}",
                    inner.dtype(),
                );
            }
        }

        Ok(())
    }
}
