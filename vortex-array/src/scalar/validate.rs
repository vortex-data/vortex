// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl Scalar {
    /// Validate that the given [`ScalarValue`] is compatible with the given [`DType`].
    #[inline]
    pub fn validate(dtype: &DType, value: Option<&ScalarValue>) -> VortexResult<()> {
        if Self::is_compatible(dtype, value) {
            return Ok(());
        }

        Err(Self::incompatible_error(dtype, value))
    }

    /// Returns `true` if the given [`ScalarValue`] is compatible with the given [`DType`].
    ///
    /// This is the fast path of [`Self::validate`]. It must accept exactly the values that
    /// [`Self::validate_detailed`] accepts, but it builds no error messages.
    ///
    /// Null values and leaf types are checked inline, so that callers with a known dtype fold the
    /// check away. Other types go to [`Self::is_compatible_nested`].
    #[inline]
    pub(super) fn is_compatible(dtype: &DType, value: Option<&ScalarValue>) -> bool {
        let Some(value) = value else {
            return dtype.is_nullable();
        };

        match (dtype, value) {
            (DType::Bool(_), ScalarValue::Bool(_))
            | (DType::Utf8(_), ScalarValue::Utf8(_))
            | (DType::Binary(_), ScalarValue::Binary(_)) => true,
            (DType::Primitive(ptype, _), ScalarValue::Primitive(pvalue)) => {
                // See `validate_detailed` for the `f16` backwards compatibility case.
                pvalue.ptype() == *ptype
                    || (matches!(ptype, PType::F16) && matches!(pvalue, PValue::U64(_)))
            }
            _ => Self::is_compatible_nested(dtype, value),
        }
    }

    /// The out-of-line part of [`Self::is_compatible`] for non-null values of all other types.
    fn is_compatible_nested(dtype: &DType, value: &ScalarValue) -> bool {
        match (dtype, value) {
            (DType::Decimal(dec_dtype, _), ScalarValue::Decimal(dvalue)) => {
                dvalue.fits_in_precision(*dec_dtype)
            }
            (DType::List(elem_dtype, _), ScalarValue::Tuple(elements)) => elements
                .iter()
                .all(|element| Self::is_compatible(elem_dtype, element.as_ref())),
            (DType::FixedSizeList(elem_dtype, size, _), ScalarValue::Tuple(elements)) => {
                elements.len() == *size as usize
                    && elements
                        .iter()
                        .all(|element| Self::is_compatible(elem_dtype, element.as_ref()))
            }
            (DType::Map(map, _), ScalarValue::Tuple(entries)) => {
                let key_dtype = map.key_dtype();
                let value_dtype = map.value_dtype();

                entries.iter().all(|entry| match entry {
                    Some(ScalarValue::Tuple(values)) => {
                        values.len() == 2
                            && Self::is_compatible(&key_dtype, values[0].as_ref())
                            && Self::is_compatible(&value_dtype, values[1].as_ref())
                    }
                    _ => false,
                })
            }
            (DType::Struct(fields, _), ScalarValue::Tuple(values)) => {
                values.len() == fields.nfields()
                    && fields
                        .field_dtypes()
                        .zip(values.iter())
                        .all(|(field, value)| Self::is_compatible(field, value.as_ref()))
            }
            (DType::Union(variants, _), ScalarValue::Union(union_value)) => variants
                .tag_to_child_index(union_value.type_id())
                .and_then(|child_index| variants.variant_by_index(child_index))
                .is_some_and(|child_dtype| {
                    Self::is_compatible(&child_dtype, union_value.child_value())
                }),
            (DType::Variant(_), ScalarValue::Variant(inner)) => {
                Self::is_compatible(inner.dtype(), inner.value())
                    && (!inner.is_null() || matches!(inner.dtype(), DType::Null))
            }
            (DType::Extension(ext_dtype), value) => ext_dtype.validate_storage_value(value).is_ok(),
            _ => false,
        }
    }

    /// Builds the error for a [`ScalarValue`] that is not compatible with the [`DType`].
    ///
    /// Kept out of line so that the formatting code does not count against the hot callers when
    /// the inliner sizes them.
    #[cold]
    #[inline(never)]
    pub(super) fn incompatible_error(dtype: &DType, value: Option<&ScalarValue>) -> VortexError {
        match Self::validate_detailed(dtype, value) {
            Err(err) => err,
            Ok(()) => vortex_err!(
                "scalar validation fast path rejected value {value:?} for dtype {dtype}, but the \
                 detailed check accepted it"
            ),
        }
    }

    /// Validates the [`ScalarValue`] against the [`DType`], with a detailed error message.
    fn validate_detailed(dtype: &DType, value: Option<&ScalarValue>) -> VortexResult<()> {
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
                    Self::validate_detailed(elem_dtype.as_ref(), element.as_ref())
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
                    "fixed-size list scalar has the wrong number of elements",
                );

                for (i, element) in elements.iter().enumerate() {
                    Self::validate_detailed(elem_dtype.as_ref(), element.as_ref()).map_err(
                        |e| vortex_error::vortex_err!("fixed-size list element at index {i}: {e}",),
                    )?;
                }
            }
            DType::Map(map, _) => {
                let ScalarValue::Tuple(entries) = value else {
                    vortex_bail!("map dtype expected Tuple value, got {value}");
                };
                let key_dtype = map.key_dtype();
                let value_dtype = map.value_dtype();

                for (index, entry) in entries.iter().enumerate() {
                    let entry = entry.as_ref().ok_or_else(|| {
                        vortex_error::vortex_err!("map entry at index {index} cannot be null")
                    })?;
                    let ScalarValue::Tuple(values) = entry else {
                        vortex_bail!(
                            "map entry at index {index} expected Tuple value, got {entry}"
                        );
                    };
                    vortex_ensure_eq!(
                        values.len(),
                        2,
                        "map entry at index {index} has the wrong number of values",
                    );

                    Self::validate_detailed(&key_dtype, values[0].as_ref()).map_err(|error| {
                        vortex_error::vortex_err!("map key at entry {index}: {error}")
                    })?;
                    Self::validate_detailed(&value_dtype, values[1].as_ref()).map_err(|error| {
                        vortex_error::vortex_err!("map value at entry {index}: {error}")
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
                    "struct scalar has the wrong number of fields",
                );

                for (field, field_value) in fields.field_dtypes().zip(values.iter()) {
                    Self::validate_detailed(field, field_value.as_ref())?;
                }
            }
            DType::Union(variants, _) => {
                let ScalarValue::Union(union_value) = value else {
                    vortex_bail!("union dtype expected Union value, got {value}");
                };

                let type_id = union_value.type_id();
                let Some(child_index) = variants.tag_to_child_index(type_id) else {
                    vortex_bail!(
                        "union value has unknown type ID {type_id}; expected one of {:?}",
                        variants.type_ids()
                    );
                };

                let child_dtype = variants
                    .variant_by_index(child_index)
                    .vortex_expect("resolved union child index must be valid");

                Self::validate_detailed(&child_dtype, union_value.child_value()).map_err(
                    |error| {
                        vortex_error::vortex_err!(
                            "union value for type ID {type_id} is invalid for dtype {child_dtype}: \
                         {error}"
                        )
                    },
                )?;
            }
            DType::Variant(_) => {
                let ScalarValue::Variant(inner) = value else {
                    vortex_bail!("variant dtype expected Variant value, got {value}");
                };

                Self::validate_detailed(inner.dtype(), inner.value())?;
                vortex_ensure!(
                    !inner.is_null() || matches!(inner.dtype(), DType::Null),
                    "variant nulls must use a nested null scalar, got {}",
                    inner.dtype(),
                );
            }
            DType::Extension(ext_dtype) => ext_dtype.validate_storage_value(value)?,
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldNames;
    use crate::dtype::MapDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::UnionVariants;
    use crate::scalar::DecimalValue;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::scalar::UnionValue;

    /// The fast check in `is_compatible` must accept exactly what `validate_detailed` accepts.
    #[test]
    fn fast_and_detailed_checks_agree() -> VortexResult<()> {
        use Nullability::NonNullable;
        use Nullability::Nullable;

        let i32_value = || Some(ScalarValue::Primitive(PValue::I32(1)));
        let i32_dtype = DType::Primitive(PType::I32, NonNullable);
        let struct_dtype = DType::Struct(
            StructFields::new(
                FieldNames::from(["a", "b"]),
                vec![i32_dtype.clone(), DType::Utf8(Nullable)],
            ),
            NonNullable,
        );
        let map_dtype = DType::Map(
            MapDType::try_new(i32_dtype.clone(), DType::Bool(Nullable), false)?,
            NonNullable,
        );
        let union_dtype = DType::Union(
            UnionVariants::try_new(["int"].into(), vec![i32_dtype.clone()], vec![3])?,
            NonNullable,
        );
        let list_dtype = DType::List(Arc::new(i32_dtype.clone()), NonNullable);
        let fsl_dtype = DType::FixedSizeList(Arc::new(i32_dtype.clone()), 2, NonNullable);
        let decimal_dtype = DType::Decimal(DecimalDType::new(3, 0), NonNullable);
        let tuple = |values: Vec<Option<ScalarValue>>| Some(ScalarValue::Tuple(values));

        let cases = [
            (DType::Null, None),
            (DType::Null, Some(ScalarValue::Bool(true))),
            (DType::Bool(Nullable), None),
            (DType::Bool(NonNullable), None),
            (DType::Bool(NonNullable), Some(ScalarValue::Bool(true))),
            (DType::Bool(NonNullable), i32_value()),
            (i32_dtype.clone(), i32_value()),
            (i32_dtype, Some(ScalarValue::Primitive(PValue::I64(1)))),
            (
                DType::Primitive(PType::F16, NonNullable),
                Some(ScalarValue::Primitive(PValue::U64(1))),
            ),
            (
                decimal_dtype.clone(),
                Some(ScalarValue::Decimal(DecimalValue::I32(999))),
            ),
            (
                decimal_dtype,
                Some(ScalarValue::Decimal(DecimalValue::I32(1000))),
            ),
            (
                DType::Utf8(NonNullable),
                Some(ScalarValue::Utf8("a".into())),
            ),
            (DType::Utf8(NonNullable), Some(ScalarValue::Bool(true))),
            (
                DType::Binary(NonNullable),
                Some(ScalarValue::Utf8("a".into())),
            ),
            (list_dtype.clone(), tuple(vec![i32_value(), i32_value()])),
            (list_dtype.clone(), tuple(vec![i32_value(), None])),
            (list_dtype, i32_value()),
            (fsl_dtype.clone(), tuple(vec![i32_value(), i32_value()])),
            (fsl_dtype, tuple(vec![i32_value()])),
            (
                map_dtype.clone(),
                tuple(vec![tuple(vec![i32_value(), None])]),
            ),
            (map_dtype.clone(), tuple(vec![None])),
            (map_dtype.clone(), tuple(vec![tuple(vec![None, None])])),
            (map_dtype, tuple(vec![tuple(vec![i32_value()])])),
            (
                struct_dtype.clone(),
                tuple(vec![i32_value(), Some(ScalarValue::Utf8("a".into()))]),
            ),
            (struct_dtype.clone(), tuple(vec![i32_value(), None])),
            (struct_dtype.clone(), tuple(vec![None, None])),
            (struct_dtype, tuple(vec![i32_value()])),
            (
                union_dtype.clone(),
                Some(ScalarValue::Union(UnionValue::new(3, i32_value()))),
            ),
            (
                union_dtype.clone(),
                Some(ScalarValue::Union(UnionValue::new(4, i32_value()))),
            ),
            (
                union_dtype,
                Some(ScalarValue::Union(UnionValue::new(3, None))),
            ),
            (
                DType::Variant(NonNullable),
                Some(ScalarValue::Variant(Box::new(Scalar::null(DType::Null)))),
            ),
            (
                DType::Variant(NonNullable),
                Some(ScalarValue::Variant(Box::new(Scalar::null(DType::Bool(
                    Nullable,
                ))))),
            ),
            (
                DType::Variant(NonNullable),
                Some(ScalarValue::Variant(Box::new(Scalar::from(true)))),
            ),
        ];

        for (dtype, value) in &cases {
            assert_eq!(
                Scalar::is_compatible(dtype, value.as_ref()),
                Scalar::validate_detailed(dtype, value.as_ref()).is_ok(),
                "fast and detailed checks disagree for {dtype} and {value:?}",
            );
        }

        Ok(())
    }

    #[test]
    fn union_rejects_unknown_tag_and_wrong_value() -> VortexResult<()> {
        let variants = UnionVariants::try_new(
            ["int", "string"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::Nullable),
                DType::Utf8(Nullability::NonNullable),
            ],
            vec![5, 9],
        )?;
        let dtype = DType::Union(variants, Nullability::NonNullable);

        assert!(
            Scalar::try_new(
                dtype.clone(),
                Some(ScalarValue::Union(UnionValue::new(
                    7,
                    Scalar::primitive(42_i32, Nullability::Nullable).into_value(),
                ))),
            )
            .is_err()
        );

        assert!(
            Scalar::try_new(
                dtype,
                Some(ScalarValue::Union(UnionValue::new(
                    5,
                    Scalar::utf8("wrong", Nullability::NonNullable).into_value(),
                ))),
            )
            .is_err()
        );

        Ok(())
    }
}
