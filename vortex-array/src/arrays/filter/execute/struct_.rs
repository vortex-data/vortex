// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::arrays::StructArray;
use crate::arrays::filter::execute::filter_validity;
use crate::arrays::struct_::StructArrayExt;

pub fn filter_struct(array: &StructArray, mask: &Arc<MaskValues>) -> StructArray {
    let filtered_validity = filter_validity(
        array
            .validity()
            .vortex_expect("struct validity should be derivable"),
        mask,
    );

    let mask_for_filter = Mask::Values(Arc::clone(mask));
    let fields: Vec<ArrayRef> = array
        .iter_unmasked_fields()
        .map(|field| {
            field
                .filter(mask_for_filter.clone())
                .vortex_expect("StructArray fields are guaranteed to support filter")
        })
        .collect();

    let length = fields
        .first()
        .map(|a| a.len())
        .unwrap_or_else(|| mask.true_count());

    StructArray::try_new_with_dtype(
        fields,
        array.struct_fields().clone(),
        length,
        filtered_validity,
    )
    .vortex_expect("filtered StructArray fields have consistent lengths")
}

#[cfg(test)]
mod test {
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability::Nullable;
    use crate::validity::Validity;

    #[test]
    fn test_filter_struct_conformance() {
        let fields = vec![
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            PrimitiveArray::from_iter([10i64, 20, 30, 40, 50]).into_array(),
        ];
        let array =
            StructArray::try_new(["a", "b"].into(), fields, 5, Validity::NonNullable).unwrap();
        test_filter_conformance(&array.into_array());
    }

    #[test]
    fn test_filter_struct_with_nulls_conformance() {
        let fields = vec![
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None])
                .into_array(),
            PrimitiveArray::from_option_iter([Some(10i64), Some(20), None, Some(40), Some(50)])
                .into_array(),
        ];
        let array =
            StructArray::try_new(["a", "b"].into(), fields, 5, Validity::NonNullable).unwrap();
        test_filter_conformance(&array.into_array());
    }

    #[test]
    fn filter_struct_selects_correct_rows() {
        let array = StructArray::try_new(
            ["x", "y"].into(),
            vec![
                PrimitiveArray::from_iter([10i32, 20, 30, 40, 50]).into_array(),
                PrimitiveArray::from_iter([1i64, 2, 3, 4, 5]).into_array(),
            ],
            5,
            Validity::NonNullable,
        )
        .unwrap();

        let mask = Mask::from_iter([true, false, true, false, true]);
        let filtered = array.filter(mask).unwrap();

        let expected = StructArray::try_new(
            ["x", "y"].into(),
            vec![
                PrimitiveArray::from_iter([10i32, 30, 50]).into_array(),
                PrimitiveArray::from_iter([1i64, 3, 5]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )
        .unwrap();

        assert_arrays_eq!(filtered, expected);
    }

    #[test]
    fn filter_empty_struct() {
        let struct_arr =
            StructArray::try_new(FieldNames::empty(), vec![], 10, Validity::NonNullable).unwrap();
        let mask = Mask::from_iter([
            false, true, false, true, false, true, false, true, false, true,
        ]);
        let filtered = struct_arr.filter(mask).unwrap();

        let expected =
            StructArray::try_new(FieldNames::empty(), vec![], 5, Validity::NonNullable).unwrap();
        assert_arrays_eq!(filtered, expected);
    }

    #[test]
    fn filter_empty_struct_with_empty_filter() {
        let struct_arr =
            StructArray::try_new(FieldNames::empty(), vec![], 0, Validity::NonNullable).unwrap();
        let filtered = struct_arr.filter(Mask::from_iter::<[bool; 0]>([])).unwrap();

        let expected =
            StructArray::try_new(FieldNames::empty(), vec![], 0, Validity::NonNullable).unwrap();
        assert_arrays_eq!(filtered, expected);
    }

    #[test]
    fn test_filter_empty_struct_conformance() {
        test_filter_conformance(
            &StructArray::try_new(FieldNames::empty(), vec![], 5, Validity::NonNullable)
                .unwrap()
                .into_array(),
        );
    }

    #[test]
    fn test_filter_complex_struct_conformance() {
        let xs = PrimitiveArray::from_iter([0i64, 1, 2, 3, 4]).into_array();
        let ys = VarBinArray::from_iter(
            [Some("a"), Some("b"), None, Some("d"), None],
            DType::Utf8(Nullable),
        )
        .into_array();
        let zs =
            BoolArray::from_iter([Some(true), Some(true), None, None, Some(false)]).into_array();

        test_filter_conformance(
            &StructArray::try_new(
                ["xs", "ys", "zs"].into(),
                vec![
                    StructArray::try_new(
                        ["left", "right"].into(),
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
}
