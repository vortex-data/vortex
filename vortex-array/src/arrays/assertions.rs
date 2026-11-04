// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use itertools::Itertools;
use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use crate::scalar::Scalar;

fn format_indices<I: IntoIterator<Item = usize>>(indices: I) -> impl Display {
    indices.into_iter().format(",")
}

/// Executes an array to recursive canonical form with the given execution context.
fn execute_to_canonical(array: ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    array
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("failed to execute array to recursive canonical form")
        .0
        .into_array()
}

/// Finds indices where two arrays differ based on `scalar_at` comparison.
#[expect(clippy::unwrap_used)]
fn find_mismatched_indices(
    left: &ArrayRef,
    right: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> Vec<usize> {
    assert_eq!(left.len(), right.len());
    (0..left.len())
        .filter(|i| {
            !left
                .execute_scalar(*i, ctx)
                .unwrap()
                .eq_nulls_equal(&right.execute_scalar(*i, ctx).unwrap())
        })
        .collect()
}

/// Asserts that the scalar at position `$n` in array `$arr` equals `$expected`.
///
/// This is a convenience macro for testing that avoids verbose scalar comparison code.
///
/// # Example
/// ```ignore
/// let arr = PrimitiveArray::from_iter([1, 2, 3]);
/// assert_nth_scalar!(arr, 0, 1, &mut ctx);
/// assert_nth_scalar!(arr, 1, 2, &mut ctx);
/// ```
#[macro_export]
macro_rules! assert_nth_scalar {
    ($arr:expr, $n:expr, $expected:expr, $ctx:expr) => {{
        use $crate::IntoArray as _;
        let arr_ref: $crate::ArrayRef = $crate::IntoArray::into_array($arr.clone());
        let expected: $crate::scalar::Scalar = $expected.try_into().unwrap();
        $crate::assert_scalars_eq!(arr_ref.execute_scalar($n, $ctx).unwrap(), expected);
    }};
}

/// Asserts that two scalars are the same value, treating two nulls of the same dtype as equal.
///
/// [`Scalar`](crate::scalar::Scalar)'s `==` follows comparison semantics and is `false` when
/// either side is null, so `assert_eq!` cannot check that a computation returned a null. This
/// macro compares with [`Scalar::eq_nulls_equal`](crate::scalar::Scalar::eq_nulls_equal)
/// instead. Each operand may be a `Scalar` or a `&Scalar`.
///
/// # Example
///
/// ```
/// use vortex_array::assert_scalars_eq;
/// use vortex_array::dtype::DType;
/// use vortex_array::dtype::Nullability;
/// use vortex_array::dtype::PType;
/// use vortex_array::scalar::Scalar;
///
/// let null = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
/// assert_scalars_eq!(null, null.clone());
/// assert_scalars_eq!(&null, null, "nulls of the same dtype are the same value");
/// ```
#[macro_export]
macro_rules! assert_scalars_eq {
    ($left:expr, $right:expr $(,)?) => {
        $crate::arrays::assert_scalars_eq_impl(&$left, &$right, ::core::option::Option::None)
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        $crate::arrays::assert_scalars_eq_impl(
            &$left,
            &$right,
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
}

/// Implementation of [`assert_scalars_eq!`].
#[doc(hidden)]
#[track_caller]
#[expect(clippy::panic)]
pub fn assert_scalars_eq_impl<L, R>(left: &L, right: &R, message: Option<std::fmt::Arguments<'_>>)
where
    L: std::borrow::Borrow<Scalar> + ?Sized,
    R: std::borrow::Borrow<Scalar> + ?Sized,
{
    let (left, right) = (left.borrow(), right.borrow());
    if !left.eq_nulls_equal(right) {
        match message {
            Some(message) => panic!(
                "assertion `left == right` failed: {message}\n  left: {left:?}\n right: {right:?}"
            ),
            None => panic!("assertion `left == right` failed\n  left: {left:?}\n right: {right:?}"),
        }
    }
}

/// Asserts that the scalar at position `$n` in array `$arr` is null.
///
/// # Example
///
/// ```ignore
/// let arr = PrimitiveArray::from_option_iter([Some(1), None, Some(3)]);
/// assert_nth_scalar_null!(arr, 1, &mut ctx);
/// ```
#[macro_export]
macro_rules! assert_nth_scalar_is_null {
    ($arr:expr, $n:expr, $ctx:expr) => {{
        let arr_ref: $crate::ArrayRef = $crate::IntoArray::into_array($arr.clone());
        let scalar = arr_ref.execute_scalar($n, $ctx).unwrap();
        assert!(
            scalar.is_null(),
            "expected scalar at index {} to be null, but was {:?}",
            $n,
            scalar
        );
    }};
}

#[macro_export]
macro_rules! assert_arrays_eq {
    ($left:expr, $right:expr, $ctx:expr) => {{
        let left: $crate::ArrayRef = $crate::IntoArray::into_array($left.clone());
        let right: $crate::ArrayRef = $crate::IntoArray::into_array($right.clone());
        if left.dtype() != right.dtype() {
            let ctx: &$crate::ExecutionCtx = &*$ctx;
            panic!(
                "assertion left == right failed: arrays differ in type: {} != {}.\n  left: {}\n right: {}",
                left.dtype(),
                right.dtype(),
                left.display_values(ctx),
                right.display_values(ctx)
            )
        }

        if left.len() != right.len() {
            let ctx: &$crate::ExecutionCtx = &*$ctx;
            panic!(
                "assertion left == right failed: arrays differ in length: {} != {}.\n  left: {}\n right: {}",
                left.len(),
                right.len(),
                left.display_values(ctx),
                right.display_values(ctx)
            );
        }

        let left = left.clone();
        let right = right.clone();
        $crate::arrays::assert_arrays_eq_impl(&left, &right, $ctx);
    }};
}

/// Implementation of `assert_arrays_eq!` — called by the macro after converting inputs to
/// `ArrayRef`.
#[track_caller]
#[expect(clippy::panic)]
pub fn assert_arrays_eq_impl(left: &ArrayRef, right: &ArrayRef, ctx: &mut ExecutionCtx) {
    let executed = execute_to_canonical(left.clone(), ctx);

    let left_right_the_same =
        all_non_distinct(left, right, ctx).vortex_expect("failed to compare left and right");
    let executed_right_the_same = all_non_distinct(&executed, right, ctx)
        .vortex_expect("failed to compare executed left and right");

    if !left_right_the_same || !executed_right_the_same {
        let left_right = find_mismatched_indices(left, right, ctx);

        let mut msg = String::new();
        if !left_right.is_empty() {
            msg.push_str(&format!(
                "\n  left != right at indices: {}",
                format_indices(left_right)
            ));
        }

        let executed_right = find_mismatched_indices(&executed, right, ctx);
        if !executed_right.is_empty() {
            msg.push_str(&format!(
                "\n  executed != right at indices: {}",
                format_indices(executed_right)
            ));
        }
        panic!(
            "assertion failed: arrays do not match:{}\n     left: {}\n    right: {}\n executed: {}",
            msg,
            left.display_values(ctx),
            right.display_values(ctx),
            executed.display_values(ctx)
        )
    }
}
