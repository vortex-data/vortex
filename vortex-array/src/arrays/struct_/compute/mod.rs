mod cast;
mod filter;
mod mask;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::arrays::StructVTable;
use crate::arrays::struct_::StructArray;
use crate::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, MinMaxKernel, MinMaxKernelAdapter,
    MinMaxResult, TakeKernel, TakeKernelAdapter, is_constant_opts, take,
};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, register_kernel};

impl TakeKernel for StructVTable {
    fn take(&self, array: &StructArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        StructArray::try_new_with_dtype(
            array
                .fields()
                .iter()
                .map(|field| take(field, indices))
                .try_collect()?,
            array.struct_dtype().clone(),
            indices.len(),
            array.validity().take(indices)?,
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(TakeKernelAdapter(StructVTable).lift());

impl MinMaxKernel for StructVTable {
    fn min_max(&self, _array: &StructArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement struct min max
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(StructVTable).lift());

impl IsConstantKernel for StructVTable {
    fn is_constant(
        &self,
        array: &StructArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let children = array.children();
        if children.is_empty() {
            return Ok(None);
        }

        for child in children.iter() {
            match is_constant_opts(child, opts)? {
                // Un-determined
                None => return Ok(None),
                Some(false) => return Ok(Some(false)),
                Some(true) => {}
            }
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(StructVTable).lift());

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldNames, Nullability, PType, StructDType};
    use vortex_mask::Mask;

    use crate::arrays::{BoolArray, BooleanBuffer, PrimitiveArray, StructArray, VarBinArray};
    use crate::compute::conformance::mask::test_mask;
    use crate::compute::{cast, filter};
    use crate::validity::Validity;
    use crate::{Array, IntoArray as _};

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
                .as_ref(),
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
            .as_ref(),
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
        let casted = cast(&array, &non_nullable_dtype).unwrap();
        assert_eq!(casted.dtype(), &non_nullable_dtype);

        let nullable_dtype = DType::Struct(
            Arc::from(StructDType::new([].into(), vec![])),
            Nullability::Nullable,
        );
        let casted = cast(&array, &nullable_dtype).unwrap();
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
        .unwrap();

        let tu8 = DType::Primitive(PType::U8, Nullability::NonNullable);

        let result = cast(
            array.as_ref(),
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
        let xs = PrimitiveArray::from_option_iter([Some(0i64), Some(1), Some(2), Some(3), Some(4)]);
        let ys = VarBinArray::from_vec(
            vec!["a", "b", "c", "d", "e"],
            DType::Utf8(Nullability::Nullable),
        );
        let zs = BoolArray::new(
            BooleanBuffer::from_iter([true, true, false, false, true]),
            Validity::AllValid,
        );
        let fully_nullable_array = StructArray::try_new(
            ["xs".into(), "ys".into(), "zs".into()].into(),
            vec![
                StructArray::try_new(
                    ["left".into(), "right".into()].into(),
                    vec![xs.to_array(), xs.to_array()],
                    5,
                    Validity::AllValid,
                )
                .unwrap()
                .into_array(),
                ys.into_array(),
                zs.into_array(),
            ],
            5,
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let top_level_non_nullable = fully_nullable_array.dtype().as_nonnullable();
        let casted = cast(&fully_nullable_array, &top_level_non_nullable).unwrap();
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
        let casted = cast(&fully_nullable_array, &non_null_xs_right).unwrap();
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
        let casted = cast(&fully_nullable_array, &non_null_xs).unwrap();
        assert_eq!(casted.dtype(), &non_null_xs);
    }
}
