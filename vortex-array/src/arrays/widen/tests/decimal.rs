// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Construction, execution, and encoding contracts for decimal `Widen` arrays.

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::SESSION;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::widen::WidenArraySlotsExt;
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
#[case::same_width(dtype(10, 2, NonNullable), dtype(18, 2, NonNullable), "narrower")]
#[case::scale(dtype(9, 1, NonNullable), dtype(18, 2, NonNullable), "scale")]
#[case::higher_precision(dtype(18, 2, NonNullable), dtype(9, 2, NonNullable), "narrower")]
#[case::integer_child(
    PType::I32.into(),
    dtype(18, 2, NonNullable),
    "integer dtypes or decimal dtypes",
)]
#[case::integer_parent(
    dtype(9, 2, NonNullable),
    PType::I64.into(),
    "integer dtypes or decimal dtypes",
)]
fn test_rejects_invalid_decimal_dtypes(
    #[case] child: DType,
    #[case] parent: DType,
    #[case] requirement: &str,
) {
    let values = Canonical::empty(&child).into_array();

    let error = WidenArray::try_new(values, parent)
        .expect_err("the child and parent violate a Widen dtype requirement");
    assert!(error.to_string().contains(requirement), "{error}");
}

#[rstest]
#[case::canonical(
    DecimalArray::from_iter([-999i32, 0, 999], DecimalDType::new(9, 2)),
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
    let executed = widened.execute::<Canonical>(&mut ctx)?.into_array();

    assert_eq!(executed.dtype(), &parent);
    let decimal = executed.as_::<Decimal>();
    assert_eq!(decimal.values_type(), DecimalType::I64);
    assert_arrays_eq!(executed, reference, &mut ctx);

    Ok(())
}

#[rstest]
#[case::scale_raises_precision(vec![Some(1)],                 7, Some(7))]
#[case::negative_boundary(     vec![Some(-128), Some(99)],    0, Some(3))]
#[case::negative_scale(        vec![Some(-99), Some(99)],    -3, Some(2))]
#[case::all_null(              vec![None, None],             0, Some(1))]
#[case::empty(                 vec![],                       0, Some(1))]
#[case::empty_positive_scale(  vec![],                      12, None)]
#[case::empty_negative_scale(  vec![],                      -4, Some(1))]
#[case::no_narrower_width(     vec![Some(1_000_000_000)],     2, None)]
fn test_encode_decimal_best_precision(
    #[case] values: Vec<Option<i64>>,
    #[case] scale: i8,
    #[case] child_precision: Option<u8>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = DecimalArray::from_option_iter(values, DecimalDType::new(18, scale));

    let encoded = WidenArray::encode_decimal(input.clone(), &mut ctx)?;

    if let Some(precision) = child_precision {
        let widened = encoded.as_::<Widen>();
        let child = widened.values().as_::<Decimal>();
        assert_eq!(child.decimal_dtype(), DecimalDType::new(precision, scale));
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

#[test]
fn test_encode_decimal_rejects_value_outside_precision() {
    let input = DecimalArray::from_iter([1_000_000_000_000_000_000i64], DecimalDType::new(18, 0));

    let error = WidenArray::encode_decimal(input, &mut SESSION.create_execution_ctx())
        .expect_err("the valid value exceeds the input precision");
    assert!(
        error.to_string().contains("Decimal range must fit"),
        "{error}"
    );
}

#[test]
fn test_smallest_decimal_dtype_i256_extrema() -> VortexResult<()> {
    let parent = DecimalDType::new(76, 0);
    let range = Some(i256::MIN_BY_PRECISION[76]..=i256::MAX_BY_PRECISION[76]);
    assert_eq!(smallest_decimal_dtype(range, parent)?, parent);
    assert!(smallest_decimal_dtype(Some(i256::MIN..=i256::MAX), parent).is_err());

    Ok(())
}
