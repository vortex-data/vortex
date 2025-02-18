mod to_arrow;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::struct_::StructArray;
use crate::array::StructEncoding;
use crate::compute::{
    filter, scalar_at, slice, take, try_cast, CastFn, FilterFn, MaskFn, MinMaxFn, MinMaxResult,
    ScalarAtFn, SliceFn, TakeFn, ToArrowFn,
};
use crate::variants::StructArrayTrait;
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray};

impl ComputeVTable for StructEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<Array>> {
        Some(self)
    }
}

impl CastFn<StructArray> for StructEncoding {
    fn cast(&self, array: &StructArray, dtype: &DType) -> VortexResult<Array> {
        let Some(target_sdtype) = dtype.as_struct() else {
            vortex_bail!("cannot cast {} to {}", array.dtype(), dtype);
        };

        let source_sdtype = array
            .dtype()
            .as_struct()
            .vortex_expect("struct array must have struct dtype");

        if target_sdtype.names() != source_sdtype.names() {
            vortex_bail!("cannot cast {} to {}", array.dtype(), dtype);
        }

        let validity = array.validity().cast_nullability(dtype.nullability())?;

        StructArray::try_new(
            target_sdtype.names().clone(),
            array
                .children()
                .into_iter()
                .zip_eq(target_sdtype.fields())
                .map(|(field, dtype)| try_cast(&field, &dtype))
                .try_collect()?,
            array.len(),
            validity,
        )
        .map(|a| a.into_array())
    }
}

impl ScalarAtFn<StructArray> for StructEncoding {
    fn scalar_at(&self, array: &StructArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::struct_(
            array.dtype().clone(),
            array
                .fields()
                .map(|field| scalar_at(&field, index))
                .try_collect()?,
        ))
    }
}

impl TakeFn<StructArray> for StructEncoding {
    fn take(&self, array: &StructArray, indices: &Array) -> VortexResult<Array> {
        StructArray::try_new(
            array.names().clone(),
            array
                .fields()
                .map(|field| take(&field, indices))
                .try_collect()?,
            indices.len(),
            array.validity().take(indices)?,
        )
        .map(|a| a.into_array())
    }
}

impl SliceFn<StructArray> for StructEncoding {
    fn slice(&self, array: &StructArray, start: usize, stop: usize) -> VortexResult<Array> {
        let fields = array
            .fields()
            .map(|field| slice(&field, start, stop))
            .try_collect()?;
        StructArray::try_new(
            array.names().clone(),
            fields,
            stop - start,
            array.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }
}

impl FilterFn<StructArray> for StructEncoding {
    fn filter(&self, array: &StructArray, mask: &Mask) -> VortexResult<Array> {
        let validity = array.validity().filter(mask)?;

        let fields: Vec<Array> = array
            .fields()
            .map(|field| filter(&field, mask))
            .try_collect()?;
        let length = fields
            .first()
            .map(|a| a.len())
            .unwrap_or_else(|| mask.true_count());

        StructArray::try_new(array.names().clone(), fields, length, validity)
            .map(|a| a.into_array())
    }
}

impl MaskFn<StructArray> for StructEncoding {
    fn mask(&self, array: &StructArray, filter_mask: Mask) -> VortexResult<Array> {
        let validity = array.validity().mask(&filter_mask)?;

        StructArray::try_new(
            array.names().clone(),
            array.fields().collect(),
            array.len(),
            validity,
        )
        .map(|a| a.into_array())
    }
}

impl MinMaxFn<StructArray> for StructEncoding {
    fn min_max(&self, _array: &StructArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement struct min max
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldNames, Nullability, PType, StructDType};
    use vortex_mask::Mask;

    use crate::array::{BoolArray, BooleanBuffer, PrimitiveArray, StructArray, VarBinArray};
    use crate::compute::test_harness::test_mask;
    use crate::compute::{filter, try_cast};
    use crate::validity::Validity;
    use crate::IntoArray as _;

    #[test]
    fn filter_empty_struct() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 10, Validity::NonNullable).unwrap();
        let mask = vec![
            false, true, false, true, false, true, false, true, false, true,
        ];
        let filtered = filter(struct_arr.as_ref(), &Mask::from_iter(mask)).unwrap();
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn filter_empty_struct_with_empty_filter() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 0, Validity::NonNullable).unwrap();
        let filtered = filter(struct_arr.as_ref(), &Mask::from_iter::<[bool; 0]>([])).unwrap();
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_mask_empty_struct() {
        test_mask(
            StructArray::try_new(vec![].into(), vec![], 5, Validity::NonNullable)
                .unwrap()
                .into_array(),
        );
    }

    #[test]
    fn test_mask_complex_struct() {
        let xs = buffer![0i64, 1, 2, 3, 4].into_array();
        let ys = VarBinArray::from_iter(
            [Some("a"), Some("b"), None, Some("d"), None],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let zs =
            BoolArray::from_iter([Some(true), Some(true), None, None, Some(false)]).into_array();

        test_mask(
            StructArray::try_new(
                ["xs".into(), "ys".into(), "zs".into()].into(),
                vec![
                    StructArray::try_new(
                        ["left".into(), "right".into()].into(),
                        vec![xs.clone(), xs],
                        5,
                        Validity::NonNullable,
                    )
                    .unwrap()
                    .into_array(),
                    ys,
                    zs,
                ],
                5,
                Validity::NonNullable,
            )
            .unwrap()
            .into_array(),
        );
    }

    #[test]
    fn test_cast_empty_struct() {
        let array = StructArray::try_new(vec![].into(), vec![], 5, Validity::NonNullable)
            .unwrap()
            .into_array();
        let non_nullable_dtype = DType::Struct(
            Arc::from(StructDType::new([].into(), vec![])),
            Nullability::NonNullable,
        );
        let casted = try_cast(&array, &non_nullable_dtype).unwrap();
        assert_eq!(casted.dtype(), &non_nullable_dtype);

        let nullable_dtype = DType::Struct(
            Arc::from(StructDType::new([].into(), vec![])),
            Nullability::Nullable,
        );
        let casted = try_cast(&array, &nullable_dtype).unwrap();
        assert_eq!(casted.dtype(), &nullable_dtype);
    }

    #[test]
    fn test_cast_cannot_change_name_order() {
        let array = StructArray::try_new(
            ["xs".into(), "ys".into(), "zs".into()].into(),
            vec![
                buffer![1u8].into_array(),
                buffer![1u8].into_array(),
                buffer![1u8].into_array(),
            ],
            1,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let tu8 = DType::Primitive(PType::U8, Nullability::NonNullable);

        let result = try_cast(
            array,
            &DType::Struct(
                Arc::from(StructDType::new(
                    FieldNames::from(["ys".into(), "xs".into(), "zs".into()]),
                    vec![tu8.clone(), tu8.clone(), tu8],
                )),
                Nullability::NonNullable,
            ),
        );
        assert!(
            result.as_ref().is_err_and(|err| {
                err.to_string()
                    .contains("cannot cast {xs=u8, ys=u8, zs=u8} to {ys=u8, xs=u8, zs=u8}")
            }),
            "{:?}",
            result
        );
    }

    #[test]
    fn test_cast_complex_struct() {
        let xs = PrimitiveArray::from_option_iter([Some(0i64), Some(1), Some(2), Some(3), Some(4)])
            .into_array();
        let ys = VarBinArray::from_vec(
            vec!["a", "b", "c", "d", "e"],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let zs = BoolArray::new(
            BooleanBuffer::from_iter([true, true, false, false, true]),
            Nullability::Nullable,
        )
        .into_array();
        let fully_nullable_array = StructArray::try_new(
            ["xs".into(), "ys".into(), "zs".into()].into(),
            vec![
                StructArray::try_new(
                    ["left".into(), "right".into()].into(),
                    vec![xs.clone(), xs],
                    5,
                    Validity::AllValid,
                )
                .unwrap()
                .into_array(),
                ys,
                zs,
            ],
            5,
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let top_level_non_nullable = fully_nullable_array.dtype().as_nonnullable();
        let casted = try_cast(&fully_nullable_array, &top_level_non_nullable).unwrap();
        assert_eq!(casted.dtype(), &top_level_non_nullable);

        let non_null_xs_right = DType::Struct(
            Arc::from(StructDType::new(
                ["xs".into(), "ys".into(), "zs".into()].into(),
                vec![
                    DType::Struct(
                        Arc::from(StructDType::new(
                            ["left".into(), "right".into()].into(),
                            vec![
                                DType::Primitive(PType::I64, Nullability::NonNullable),
                                DType::Primitive(PType::I64, Nullability::Nullable),
                            ],
                        )),
                        Nullability::Nullable,
                    ),
                    DType::Utf8(Nullability::Nullable),
                    DType::Bool(Nullability::Nullable),
                ],
            )),
            Nullability::Nullable,
        );
        let casted = try_cast(&fully_nullable_array, &non_null_xs_right).unwrap();
        assert_eq!(casted.dtype(), &non_null_xs_right);

        let non_null_xs = DType::Struct(
            Arc::from(StructDType::new(
                ["xs".into(), "ys".into(), "zs".into()].into(),
                vec![
                    DType::Struct(
                        Arc::from(StructDType::new(
                            ["left".into(), "right".into()].into(),
                            vec![
                                DType::Primitive(PType::I64, Nullability::Nullable),
                                DType::Primitive(PType::I64, Nullability::Nullable),
                            ],
                        )),
                        Nullability::NonNullable,
                    ),
                    DType::Utf8(Nullability::Nullable),
                    DType::Bool(Nullability::Nullable),
                ],
            )),
            Nullability::Nullable,
        );
        let casted = try_cast(&fully_nullable_array, &non_null_xs).unwrap();
        assert_eq!(casted.dtype(), &non_null_xs);
    }
}
