// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Construction, execution, and encoding contracts for decimal `Widen` arrays.

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::SESSION;
use super::stored_decimal_dtype;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyArrayData;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::WidenSlots;
use crate::arrays::widen::encode::smallest_decimal_dtype;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::validity::Validity;

fn dtype(precision: u8, scale: i8, nullability: Nullability) -> DType {
    DType::Decimal(DecimalDType::new(precision, scale), nullability)
}

#[rstest]
#[case::i32_to_i64(dtype(9, 2, NonNullable), dtype(18, 2, NonNullable), true)]
#[case::i128_to_i256(dtype(38, 0, Nullable), dtype(76, 0, Nullable), true)]
#[case::same_i64(dtype(10, 2, NonNullable), dtype(18, 2, NonNullable), false)]
#[case::same_i256(dtype(75, 0, NonNullable), dtype(76, 0, NonNullable), false)]
#[case::scale(dtype(9, 1, NonNullable), dtype(18, 2, NonNullable), false)]
#[case::nullability(dtype(9, 2, Nullable), dtype(18, 2, NonNullable), false)]
#[case::higher_precision(dtype(18, 2, NonNullable), dtype(9, 2, NonNullable), false)]
#[case::integer_child(PType::I32.into(), dtype(18, 2, NonNullable), false)]
#[case::integer_parent(dtype(9, 2, NonNullable), PType::I64.into(), false)]
fn test_try_new_decimal_dtypes(
    #[case] child: DType,
    #[case] parent: DType,
    #[case] accepted: bool,
) -> VortexResult<()> {
    let values = Canonical::empty(&child).into_array();
    let parts = ArrayParts::new(
        Widen,
        parent.clone(),
        0,
        EmptyArrayData,
        WidenSlots {
            values: values.clone(),
        }
        .into_slots(),
    );

    let constructed = WidenArray::try_new(values, parent.clone());
    let validated = WidenArray::try_from_parts(parts);

    assert_eq!(constructed.is_ok(), accepted);
    assert_eq!(validated.is_ok(), accepted);
    if !accepted {
        let error = constructed
            .expect_err("the dtype pair is invalid")
            .to_string();
        assert!(error.contains(&child.to_string()), "{error}");
        assert!(error.contains(&parent.to_string()), "{error}");
    }

    Ok(())
}

#[test]
fn test_decimal_widen_flattens() -> VortexResult<()> {
    let values = DecimalArray::from_iter([1i16, 99], DecimalDType::new(4, 0)).into_array();
    let inner = WidenArray::try_new(values.clone(), dtype(9, 0, NonNullable))?.into_array();

    let outer = WidenArray::try_new(inner, dtype(18, 0, NonNullable))?;

    assert!(ArrayRef::ptr_eq(outer.values(), &values));
    assert!(!outer.values().is::<Widen>());

    Ok(())
}

#[rstest]
#[case::canonical(
    DecimalArray::from_iter([-999i32, 0, 999], DecimalDType::new(9, 2)),
    [-999, 0, 999],
)]
#[case::nullable(
    DecimalArray::from_option_iter([Some(-999i32), None, Some(999)], DecimalDType::new(9, 2)),
    [-999, 0, 999],
)]
#[case::narrow_stored(
    DecimalArray::from_iter([-99i8, 0, 99], DecimalDType::new(9, 2)),
    [-99, 0, 99],
)]
#[case::parent_width_stored(
    DecimalArray::from_iter([-999i64, 0, 999], DecimalDType::new(9, 2)),
    [-999, 0, 999],
)]
#[case::wider_stored(
    DecimalArray::from_iter([-999i128, 0, 999], DecimalDType::new(9, 2)),
    [-999, 0, 999],
)]
#[case::null_garbage(
    DecimalArray::new(
        buffer![i128::MIN, 999, i128::MAX],
        DecimalDType::new(9, 2),
        Validity::from_iter([false, true, false]),
    ),
    [0, 999, 0],
)]
fn test_decimal_execute_widens_and_terminates(
    #[case] child: DecimalArray,
    #[case] expected: [i64; 3],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = child.validity()?;
    let parent = dtype(18, 2, validity.nullability());
    let reference = DecimalArray::new(
        expected.into_iter().collect::<Buffer<i64>>(),
        DecimalDType::new(18, 2),
        validity,
    );

    let widened = WidenArray::try_new(child.into_array(), parent.clone())?.into_array();
    let executed = widened.clone().execute::<Canonical>(&mut ctx)?.into_array();

    assert_eq!(executed.dtype(), &parent);
    let decimal = executed.as_::<Decimal>();
    assert_eq!(decimal.values_type(), DecimalType::I64);
    assert_arrays_eq!(executed, reference, &mut ctx);
    assert_arrays_eq!(widened, reference, &mut ctx);

    Ok(())
}

#[rstest]
#[case::scale_raises_precision(vec![Some(1)], 7, 7, true)]
#[case::negative_boundary(vec![Some(-128), Some(99)], 0, 3, true)]
#[case::negative_scale(vec![Some(-99), Some(99)], -3, 2, true)]
#[case::all_null(vec![None, None], 0, 1, true)]
#[case::empty(vec![], 0, 1, true)]
#[case::no_narrower_width(vec![Some(1_000_000_000)], 2, 10, false)]
fn test_encode_decimal_best_precision(
    #[case] values: Vec<Option<i64>>,
    #[case] scale: i8,
    #[case] child_precision: u8,
    #[case] is_widen: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = DecimalArray::from_option_iter(values, DecimalDType::new(18, scale));

    let encoded = WidenArray::encode_decimal(input.clone(), &mut ctx)?;

    assert_eq!(encoded.is::<Widen>(), is_widen);
    if let Some(widen) = encoded.as_opt::<Widen>() {
        assert_eq!(stored_decimal_dtype(&encoded).precision(), child_precision);
        let child = widen.values().as_::<Decimal>();
        assert_eq!(
            child.values_type(),
            child.decimal_dtype().canonical_values_type()
        );
    } else {
        assert!(ArrayRef::ptr_eq(&encoded, &input.clone().into_array()));
    }
    assert_arrays_eq!(encoded, input, &mut ctx);

    Ok(())
}

#[test]
fn test_encode_decimal_ignores_null_payloads() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = DecimalArray::new(
        buffer![i64::MIN, -128, i64::MAX],
        DecimalDType::new(18, 0),
        Validity::from_iter([false, true, false]),
    );

    let encoded = WidenArray::encode_decimal(input.clone(), &mut ctx)?;

    assert_eq!(
        encoded.as_::<Widen>().values().dtype(),
        &dtype(3, 0, Nullable)
    );
    assert_arrays_eq!(encoded, input, &mut ctx);

    Ok(())
}

#[rstest]
#[case::empty_positive_scale(None, 18, 12, Some(12))]
#[case::empty_negative_scale(None, 18, -4, Some(1))]
#[case::negative_boundary(Some((-128, 0)), 18, 0, Some(3))]
#[case::decimal_boundary(Some((-999, 999)), 18, 0, Some(3))]
#[case::outside_parent(Some((-1_000, 1)), 3, 0, None)]
#[case::reversed(Some((2, 1)), 18, 0, None)]
fn test_smallest_decimal_dtype(
    #[case] range: Option<(i64, i64)>,
    #[case] parent_precision: u8,
    #[case] scale: i8,
    #[case] precision: Option<u8>,
) -> VortexResult<()> {
    let range = range.map(|(min, max)| i256::from_i128(min.into())..=i256::from_i128(max.into()));
    let result = smallest_decimal_dtype(range, DecimalDType::new(parent_precision, scale));
    match precision {
        Some(precision) => assert_eq!(result?, DecimalDType::new(precision, scale)),
        None => assert!(result.is_err()),
    }

    Ok(())
}

#[test]
fn test_smallest_decimal_dtype_i256_extrema() -> VortexResult<()> {
    let parent = DecimalDType::new(76, 0);
    let range = Some(i256::MIN_BY_PRECISION[76]..=i256::MAX_BY_PRECISION[76]);
    assert_eq!(smallest_decimal_dtype(range, parent)?, parent);
    assert!(smallest_decimal_dtype(Some(i256::MIN..=i256::MAX), parent).is_err());

    Ok(())
}
