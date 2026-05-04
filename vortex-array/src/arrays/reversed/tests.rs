// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;

use crate::Canonical;
use crate::IntoArray as _;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute as _;
use crate::arrays::struct_::StructArrayExt as _;
use crate::arrays::{
    BoolArray, Chunked, ChunkedArray, Dict, PrimitiveArray, Reversed, StructArray,
};
use crate::assert_arrays_eq;
use crate::builders::dict::dict_encode;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

#[test]
fn test_reverse_primitive() {
    let arr = buffer![1i32, 2, 3, 4, 5].into_array();
    let reversed = arr.reverse().unwrap();
    let expected = PrimitiveArray::from_iter([5i32, 4, 3, 2, 1]);
    assert_arrays_eq!(reversed, expected);
}

#[test]
fn test_reverse_nullable_primitive() {
    let arr = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let reversed = arr.reverse().unwrap();
    let expected = PrimitiveArray::from_option_iter([Some(3i32), None, Some(1)]);
    assert_arrays_eq!(reversed, expected);
}

#[test]
fn test_reverse_empty_is_identity() {
    let arr = PrimitiveArray::from_iter([] as [i32; 0]).into_array();
    let reversed = arr.reverse().unwrap();
    assert_arrays_eq!(reversed, arr);
}

#[test]
fn test_reverse_single_is_identity() {
    let arr = buffer![42i32].into_array();
    let reversed = arr.reverse().unwrap();
    assert_arrays_eq!(reversed, arr);
}

/// Double reversal must cancel out: `x.reverse().reverse()` returns the original
/// array without any `ReversedArray` wrapper.
#[test]
fn test_double_reversal_cancels() {
    let arr = buffer![1i32, 2, 3, 4, 5].into_array();
    let double_reversed = arr.reverse().unwrap().reverse().unwrap();
    assert!(
        !double_reversed.is::<Reversed>(),
        "double reversal should eliminate both Reversed wrappers"
    );
    assert_arrays_eq!(double_reversed, arr);
}

#[test]
fn test_reverse_bool() {
    let arr = BoolArray::from_iter([true, false, true, true, false]).into_array();
    let reversed = arr.reverse().unwrap();
    let expected = BoolArray::from_iter([false, true, true, false, true]);
    assert_arrays_eq!(reversed, expected);
}

#[test]
fn test_reverse_nullable_bool() {
    let arr = BoolArray::from_iter([Some(true), None, Some(false)]).into_array();
    let reversed = arr.reverse().unwrap();
    let expected = BoolArray::from_iter([Some(false), None, Some(true)]);
    assert_arrays_eq!(reversed, expected);
}

/// Reversing a dict-encoded array must fire the `ReverseReduceAdaptor(Dict)` rule,
/// producing `Dict(Reversed(codes), values)` rather than `Reversed(Dict(...))`.
/// Only the codes array (small integers) is reversed; the values dictionary is reused.
#[test]
fn test_reverse_dict_produces_dict() {
    let arr = dict_encode(&buffer![1i32, 2, 3, 2, 1].into_array()).unwrap();
    let reversed = arr.into_array().reverse().unwrap();
    assert!(
        reversed.is::<Dict>(),
        "dict reversal should produce a Dict, not a Reversed(Dict)"
    );
    let expected = PrimitiveArray::from_iter([1i32, 2, 3, 2, 1].iter().rev().copied());
    assert_arrays_eq!(reversed, expected);
}

/// Reversing a nullable dict-encoded array also preserves the Dict encoding.
#[test]
fn test_reverse_nullable_dict_produces_dict() {
    let arr = dict_encode(
        &PrimitiveArray::from_option_iter([Some(10i32), None, Some(20), None, Some(10)])
            .into_array(),
    )
    .unwrap();
    let reversed = arr.into_array().reverse().unwrap();
    assert!(reversed.is::<Dict>());
    let expected = PrimitiveArray::from_option_iter([Some(10i32), None, Some(20), None, Some(10)]);
    assert_arrays_eq!(reversed, expected);
}

/// Reversing a struct array reverses each field independently.
#[test]
fn test_reverse_struct() {
    let arr = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            buffer![10i32, 20, 30].into_array(),
            buffer![1u64, 2, 3].into_array(),
        ],
        3,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let reversed = arr.reverse().unwrap();
    let expected = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            buffer![30i32, 20, 10].into_array(),
            buffer![3u64, 2, 1].into_array(),
        ],
        3,
        Validity::NonNullable,
    )
    .unwrap();
    assert_arrays_eq!(reversed, expected);
}

/// Dict-encoded fields inside a struct remain dict-encoded after reversal.
/// The struct's `reverse_struct` path calls `field.reverse()` on each child,
/// which in turn fires `ReverseReduceAdaptor(Dict)`.
#[test]
fn test_reverse_struct_preserves_dict_encoding() {
    let field = dict_encode(&buffer![1i32, 2, 1, 2].into_array())
        .unwrap()
        .into_array();
    let arr = StructArray::try_new(
        FieldNames::from(["x"]),
        vec![field],
        4,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let reversed = arr.reverse().unwrap();

    // Execute to get the canonical struct with its reversed fields.
    let canonical = reversed
        .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    let Canonical::Struct(s) = canonical else {
        panic!("expected Struct canonical");
    };
    // The field should still be dict-encoded (codes reversed, values intact).
    assert!(
        s.unmasked_field(0).is::<Dict>(),
        "dict field should remain Dict-encoded after struct reversal"
    );
    let expected = PrimitiveArray::from_iter([2i32, 1, 2, 1]);
    assert_arrays_eq!(s.unmasked_field(0), expected);
}

/// Reversing a `ChunkedArray` must fire the `ReverseReduceAdaptor(Chunked)` rule,
/// producing `Chunked([reverse(cn), …, reverse(c0)])` rather than `Reversed(Chunked(…))`.
/// This avoids eagerly merging all chunks before reversing.
#[test]
fn test_reverse_chunked_produces_chunked() {
    let arr = ChunkedArray::try_new(
        vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![4i32, 5].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .unwrap()
    .into_array();

    let reversed = arr.reverse().unwrap();
    assert!(
        reversed.is::<Chunked>(),
        "chunked reversal should produce Chunked, not Reversed(Chunked)"
    );
    // Values must be fully reversed across chunk boundaries.
    let expected = PrimitiveArray::from_iter([5i32, 4, 3, 2, 1]);
    assert_arrays_eq!(reversed, expected);
}

/// Each individual chunk within the reversed `ChunkedArray` must itself be reversed,
/// not just the chunk order.
#[test]
fn test_reverse_chunked_per_chunk_reversal() {
    let arr = ChunkedArray::try_new(
        vec![
            buffer![10i32, 20, 30].into_array(),
            buffer![40i32, 50].into_array(),
            buffer![60i32].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .unwrap()
    .into_array();

    // Expected: last chunk first ([60]), middle chunk reversed ([50, 40]),
    // first chunk reversed ([30, 20, 10]).
    let reversed = arr.reverse().unwrap();
    let expected = PrimitiveArray::from_iter([60i32, 50, 40, 30, 20, 10]);
    assert_arrays_eq!(reversed, expected);
}
