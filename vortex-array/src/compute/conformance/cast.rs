// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::RecursiveCanonical;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// Cast and force execution via `to_canonical`, returning the canonical array.
fn cast_and_execute(array: &ArrayRef, dtype: DType) -> VortexResult<ArrayRef> {
    Ok(array
        .cast(dtype)?
        .execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())?
        .0
        .into_array())
}

/// Test conformance of the cast compute function for an array.
///
/// This function tests various casting scenarios including:
/// - Casting between numeric types (widening and narrowing)
/// - Casting between signed and unsigned types
/// - Casting between integral and floating-point types
/// - Casting with nullability changes
/// - Casting between string types (Utf8/Binary)
/// - Edge cases like overflow behavior
pub fn test_cast_conformance(array: &ArrayRef) {
    let dtype = array.dtype();

    // Always test identity cast and nullability changes
    test_cast_identity(array);

    test_cast_to_non_nullable(array);
    test_cast_to_nullable(array);

    // Test based on the specific DType
    match dtype {
        DType::Null => test_cast_from_null(array),
        DType::Primitive(ptype, ..) => match ptype {
            PType::U8
            | PType::U16
            | PType::U32
            | PType::U64
            | PType::I8
            | PType::I16
            | PType::I32
            | PType::I64 => test_cast_to_integral_types(array),
            PType::F16 | PType::F32 | PType::F64 => test_cast_from_floating_point_types(array),
        },
        _ => {}
    }
}

fn test_cast_identity(array: &ArrayRef) {
    // Casting to the same type should be a no-op
    let result = cast_and_execute(&array.clone(), array.dtype().clone())
        .vortex_expect("cast should succeed in conformance test");
    assert_eq!(result.len(), array.len());
    assert_eq!(result.dtype(), array.dtype());

    // Verify values are unchanged
    for i in 0..array.len().min(10) {
        assert_eq!(
            array
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test"),
            result
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

fn test_cast_from_null(array: &ArrayRef) {
    // Null can be cast to itself
    let result = cast_and_execute(&array.clone(), DType::Null)
        .vortex_expect("cast should succeed in conformance test");
    assert_eq!(result.len(), array.len());
    assert_eq!(result.dtype(), &DType::Null);

    // Null can also be cast to any nullable type
    let nullable_types = vec![
        DType::Bool(Nullability::Nullable),
        DType::Primitive(PType::I32, Nullability::Nullable),
        DType::Primitive(PType::F64, Nullability::Nullable),
        DType::Utf8(Nullability::Nullable),
        DType::Binary(Nullability::Nullable),
    ];

    for dtype in nullable_types {
        let result = cast_and_execute(&array.clone(), dtype.clone())
            .vortex_expect("cast should succeed in conformance test");
        assert_eq!(result.len(), array.len());
        assert_eq!(result.dtype(), &dtype);

        // Verify all values are null
        for i in 0..array.len().min(10) {
            assert!(
                result
                    .scalar_at(i)
                    .vortex_expect("scalar_at should succeed in conformance test")
                    .is_null()
            );
        }
    }

    // Casting to non-nullable types should fail
    let non_nullable_types = vec![
        DType::Bool(Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ];

    for dtype in non_nullable_types {
        assert!(cast_and_execute(&array.clone(), dtype.clone()).is_err());
    }
}

fn test_cast_to_non_nullable(array: &ArrayRef) {
    if array
        .invalid_count()
        .vortex_expect("invalid_count should succeed in conformance test")
        == 0
    {
        let non_nullable = cast_and_execute(&array.clone(), array.dtype().as_nonnullable())
            .vortex_expect("arrays without nulls can cast to non-nullable");
        assert_eq!(non_nullable.dtype(), &array.dtype().as_nonnullable());
        assert_eq!(non_nullable.len(), array.len());

        for i in 0..array.len().min(10) {
            assert_eq!(
                array
                    .scalar_at(i)
                    .vortex_expect("scalar_at should succeed in conformance test"),
                non_nullable
                    .scalar_at(i)
                    .vortex_expect("scalar_at should succeed in conformance test")
            );
        }

        let back_to_nullable = cast_and_execute(&non_nullable, array.dtype().clone())
            .vortex_expect("non-nullable arrays can cast to nullable");
        assert_eq!(back_to_nullable.dtype(), array.dtype());
        assert_eq!(back_to_nullable.len(), array.len());

        for i in 0..array.len().min(10) {
            assert_eq!(
                array
                    .scalar_at(i)
                    .vortex_expect("scalar_at should succeed in conformance test"),
                back_to_nullable
                    .scalar_at(i)
                    .vortex_expect("scalar_at should succeed in conformance test")
            );
        }
    } else {
        if &DType::Null == array.dtype() {
            // DType::Null.as_nonnullable() (confusingly) returns DType:Null. Of course, a null
            // array can be casted to DType::Null.
            return;
        }
        cast_and_execute(&array.clone(), array.dtype().as_nonnullable())
            .err()
            .unwrap_or_else(|| {
                vortex_panic!(
                    "arrays with nulls should error when casting to non-nullable {}",
                    array,
                )
            });
    }
}

fn test_cast_to_nullable(array: &ArrayRef) {
    let nullable = cast_and_execute(&array.clone(), array.dtype().as_nullable())
        .vortex_expect("arrays without nulls can cast to nullable");
    assert_eq!(nullable.dtype(), &array.dtype().as_nullable());
    assert_eq!(nullable.len(), array.len());

    for i in 0..array.len().min(10) {
        assert_eq!(
            array
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test"),
            nullable
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }

    let back = cast_and_execute(&nullable, array.dtype().clone())
        .vortex_expect("casting to nullable and back should be a no-op");
    assert_eq!(back.dtype(), array.dtype());
    assert_eq!(back.len(), array.len());

    for i in 0..array.len().min(10) {
        assert_eq!(
            array
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test"),
            back.scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

fn test_cast_from_floating_point_types(array: &ArrayRef) {
    let ptype = array.as_primitive_typed().ptype();
    test_cast_to_primitive(array, PType::I8, false);
    test_cast_to_primitive(array, PType::U8, false);
    test_cast_to_primitive(array, PType::I16, false);
    test_cast_to_primitive(array, PType::U16, false);
    test_cast_to_primitive(array, PType::I32, false);
    test_cast_to_primitive(array, PType::U32, false);
    test_cast_to_primitive(array, PType::I64, false);
    test_cast_to_primitive(array, PType::U64, false);
    test_cast_to_primitive(array, PType::F16, matches!(ptype, PType::F16));
    test_cast_to_primitive(array, PType::F32, matches!(ptype, PType::F16 | PType::F32));
    test_cast_to_primitive(array, PType::F64, true);
}

fn test_cast_to_integral_types(array: &ArrayRef) {
    test_cast_to_primitive(array, PType::I8, true);
    test_cast_to_primitive(array, PType::U8, true);
    test_cast_to_primitive(array, PType::I16, true);
    test_cast_to_primitive(array, PType::U16, true);
    test_cast_to_primitive(array, PType::I32, true);
    test_cast_to_primitive(array, PType::U32, true);
    test_cast_to_primitive(array, PType::I64, true);
    test_cast_to_primitive(array, PType::U64, true);
}

/// Does this scalar fit in this type?
fn fits(value: &Scalar, ptype: PType) -> bool {
    let dtype = DType::Primitive(ptype, value.dtype().nullability());
    value.cast(&dtype).is_ok()
}

fn test_cast_to_primitive(array: &ArrayRef, target_ptype: PType, test_round_trip: bool) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let maybe_min_max =
        min_max(array, &mut ctx).vortex_expect("cast should succeed in conformance test");

    if let Some(MinMaxResult { min, max }) = maybe_min_max
        && (!fits(&min, target_ptype) || !fits(&max, target_ptype))
    {
        cast_and_execute(
            &array.clone(),
            DType::Primitive(target_ptype, array.dtype().nullability()),
        )
        .err()
        .unwrap_or_else(|| {
            vortex_panic!(
                "Cast must fail because some values are out of bounds. {} {:?} {:?} {} {}",
                target_ptype,
                min,
                max,
                array,
                array.display_values(),
            )
        });
        return;
    }

    // Otherwise, all values must fit.
    let casted = cast_and_execute(
        &array.clone(),
        DType::Primitive(target_ptype, array.dtype().nullability()),
    )
    .unwrap_or_else(|e| {
        vortex_panic!(
            "Cast must succeed because all values are within bounds. {} {}: {e}",
            target_ptype,
            array.display_values(),
        )
    });
    assert_eq!(
        array
            .validity_mask()
            .vortex_expect("validity_mask should succeed in conformance test"),
        casted
            .validity_mask()
            .vortex_expect("validity_mask should succeed in conformance test")
    );
    for i in 0..array.len().min(10) {
        let original = array
            .scalar_at(i)
            .vortex_expect("scalar_at should succeed in conformance test");
        let casted = casted
            .scalar_at(i)
            .vortex_expect("scalar_at should succeed in conformance test");
        assert_eq!(
            original
                .cast(casted.dtype())
                .vortex_expect("cast should succeed in conformance test"),
            casted,
            "{i} {original} {casted}"
        );
        if test_round_trip {
            assert_eq!(
                original,
                casted
                    .cast(original.dtype())
                    .vortex_expect("cast should succeed in conformance test"),
                "{i} {original} {casted}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;

    #[test]
    fn test_cast_conformance_u32() {
        let array = buffer![0u32, 100, 200, 65535, 1000000].into_array();
        test_cast_conformance(&array);
    }

    #[test]
    fn test_cast_conformance_i32() {
        let array = buffer![-100i32, -1, 0, 1, 100].into_array();
        test_cast_conformance(&array);
    }

    #[test]
    fn test_cast_conformance_f32() {
        let array = buffer![0.0f32, 1.5, -2.5, 100.0, 1e6].into_array();
        test_cast_conformance(&array);
    }

    #[test]
    fn test_cast_conformance_nullable() {
        let array = PrimitiveArray::from_option_iter([Some(1u8), None, Some(255), Some(0), None]);
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_conformance_bool() {
        let array = BoolArray::from_iter(vec![true, false, true, false]);
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_conformance_null() {
        let array = NullArray::new(5);
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_conformance_utf8() {
        let array = VarBinArray::from_iter(
            vec![Some("hello"), None, Some("world")],
            DType::Utf8(Nullability::Nullable),
        );
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_conformance_binary() {
        let array = VarBinArray::from_iter(
            vec![Some(b"data".as_slice()), None, Some(b"bytes".as_slice())],
            DType::Binary(Nullability::Nullable),
        );
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_conformance_struct() {
        let names = FieldNames::from(["a", "b"]);

        let a = buffer![1i32, 2, 3].into_array();
        let b = VarBinArray::from_iter(
            vec![Some("x"), None, Some("z")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();

        let array =
            StructArray::try_new(names, vec![a, b], 3, crate::validity::Validity::NonNullable)
                .unwrap();
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_conformance_list() {
        let data = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0i64, 2, 2, 5, 6].into_array();

        let array =
            ListArray::try_new(data, offsets, crate::validity::Validity::NonNullable).unwrap();
        test_cast_conformance(&array.into_array());
    }
}
