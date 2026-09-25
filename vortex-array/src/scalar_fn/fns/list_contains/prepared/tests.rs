// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::Probe;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::assert_arrays_eq;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::ListContainsSet;
use crate::validity::Validity;

fn nested_needles() -> ArrayRef {
    ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), None, Some(9)])
            .into_array(),
        buffer![0u32, 2, 2, 4, 5].into_array(),
        Validity::from(BitBuffer::from_iter([true, false, true, true])),
    )
    .unwrap()
    .into_array()
}

fn map_needles() -> ArrayRef {
    let ctx = array_session().create_execution_ctx();
    let dtype = DType::map(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        false,
        Nullability::Nullable,
    )
    .unwrap();
    let mut builder = builder_with_capacity_in(&dtype, 4, ctx.allocator());
    for key in [Some(1i32), None, Some(2), Some(9)] {
        let scalar = match key {
            Some(key) => Scalar::map(
                dtype.clone(),
                [(key.into(), Scalar::null(DType::Utf8(Nullability::Nullable)))],
            ),
            None => Scalar::null(dtype.clone()),
        };
        builder.append_scalar(&scalar).unwrap();
    }
    builder.finish()
}

fn struct_needles() -> ArrayRef {
    StructArray::from_fields(&[("list", nested_needles())])
        .unwrap()
        .into_array()
}

#[rstest]
#[case::list(nested_needles())]
#[case::map(map_needles())]
#[case::struct_of_lists(struct_needles())]
#[case::fixed_size_list(FixedSizeListArray::new(
    PrimitiveArray::from_option_iter([
        Some(1i32), None, Some(2), None, Some(9), None, Some(8), None,
    ]).into_array(),
    2, Validity::NonNullable, 4,
).into_array())]
fn test_row_set_returns_membership_bits(
    #[case] needles: ArrayRef,
    #[values(false, true)] sql_null_semantics: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = needles.dtype().as_nullable();
    // Unsorted, repeated members and a top-level null exercise normalization and SQL's
    // unknown non-match independently of any nulls nested inside the members.
    let members = [2, 0, 2]
        .map(|idx| needles.execute_scalar(idx, &mut ctx)?.cast(&dtype))
        .into_iter()
        .collect::<VortexResult<Vec<_>>>()?;
    let mut elements = members.clone();
    elements.push(Scalar::null(dtype.clone()));
    let list = ConstantArray::new(
        Scalar::list(dtype, elements, Nullability::NonNullable),
        needles.len(),
    )
    .into_array();
    let options = ListContainsOptions { sql_null_semantics };
    let set = ListContainsSet::try_new(&list, needles.dtype(), &options, &mut ctx)?
        .unwrap()
        .prepare(&mut ctx)?;
    let result = set.contains(&needles, &mut ctx)?;
    // The old fallback returned a lazy OR tree here.
    assert!(result.is::<Bool>());
    let expected = (0..needles.len())
        .map(|idx| {
            let value = needles.execute_scalar(idx, &mut ctx)?;
            Ok(if value.is_null() {
                None
            } else if members.contains(&value) {
                Some(true)
            } else if sql_null_semantics {
                None
            } else {
                Some(false)
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[rstest]
fn test_decimal_bitmap_across_storage_widths(
    #[values(2, 4, 9, 18, 38, 76)] precision: u8,
    #[values(
        DecimalType::I8, DecimalType::I16, DecimalType::I32,
        DecimalType::I64, DecimalType::I128, DecimalType::I256,
    )]
    needle_width: DecimalType,
    #[values(false, true)] sql_null_semantics: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal = DecimalDType::new(precision, 1);
    let dtype = DType::Decimal(decimal, Nullability::Nullable);
    let mut elements: Vec<_> = [1i8, -2, 1]
        .map(|value| Scalar::decimal(value.into(), decimal, Nullability::Nullable))
        .into();
    elements.push(Scalar::null(dtype.clone()));
    let list = ConstantArray::new(
        Scalar::list(dtype, elements, Nullability::NonNullable),
        4,
    )
    .into_array();
    let needles = match_each_decimal_value_type!(needle_width, |T| {
        DecimalArray::from_option_iter::<T, _>(
            [Some(-2i8), Some(0), Some(1), None].map(|value| {
                value.map(|value| DecimalValue::from(value).cast::<T>().unwrap())
            }),
            decimal,
        )
        .into_array()
    });
    let options = ListContainsOptions { sql_null_semantics };
    let set = ListContainsSet::try_new(&list, needles.dtype(), &options, &mut ctx)?
        .unwrap()
        .prepare(&mut ctx)?;
    assert!(matches!(&set.probe, Probe::DecimalBitmap { .. }));
    let non_match = (!sql_null_semantics).then_some(false);
    assert_arrays_eq!(
        set.contains(&needles, &mut ctx)?,
        BoolArray::from_iter([Some(true), non_match, Some(true), None]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::dense_i128(38, i256::from_i128(1i128 << 100), true)]
#[case::sparse_i128(38, i256::from_i128(1i128 << 100), false)]
#[case::dense_i256(76, i256::from_parts(0, 1i128 << 72), true)]
#[case::sparse_i256(76, i256::from_parts(0, 1i128 << 72), false)]
fn test_decimal_wide_values(
    #[case] precision: u8,
    #[case] base: i256,
    #[case] dense: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal = DecimalDType::new(precision, 2);
    let second = if dense { base + i256::ONE } else { -base };
    let list = ConstantArray::new(
        Scalar::list(
            DType::Decimal(decimal, Nullability::NonNullable),
            [second, base, second]
                .map(|v| Scalar::decimal(v.into(), decimal, Nullability::NonNullable))
                .into(),
            Nullability::NonNullable,
        ),
        6,
    )
    .into_array();
    // The last non-null value shares the low 64 bits of a member but must not match it.
    let needles = DecimalArray::from_option_iter::<i256, _>(
        [
            Some(base),
            Some(second),
            Some(base - i256::ONE),
            Some(i256::ZERO),
            Some(base + i256::from_i128(1i128 << 64)),
            None,
        ],
        decimal,
    )
    .into_array();
    let set = ListContainsSet::try_new(
        &list,
        needles.dtype(),
        &ListContainsOptions::default(),
        &mut ctx,
    )?
    .unwrap()
    .prepare(&mut ctx)?;
    assert_eq!(matches!(&set.probe, Probe::DecimalBitmap { .. }), dense);
    assert_eq!(matches!(&set.probe, Probe::DecimalSorted(_)), !dense);
    assert_arrays_eq!(
        set.contains(&needles, &mut ctx)?,
        BoolArray::from_iter([
            Some(true), Some(true), Some(false), Some(false), Some(false), None,
        ]),
        &mut ctx
    );
    Ok(())
}
