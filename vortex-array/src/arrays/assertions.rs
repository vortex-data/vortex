// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use itertools::Itertools;
use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::RecursiveCanonical;
use crate::VortexSessionExecute;

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
fn find_mismatched_indices(left: &ArrayRef, right: &ArrayRef) -> Vec<usize> {
    assert_eq!(left.len(), right.len());
    (0..left.len())
        .filter(|i| left.scalar_at(*i).unwrap() != right.scalar_at(*i).unwrap())
        .collect()
}

/// Asserts that the scalar at position `$n` in array `$arr` equals `$expected`.
///
/// This is a convenience macro for testing that avoids verbose scalar comparison code.
///
/// # Example
/// ```ignore
/// let arr = PrimitiveArray::from_iter([1, 2, 3]);
/// assert_nth_scalar!(arr, 0, 1);
/// assert_nth_scalar!(arr, 1, 2);
/// ```
#[macro_export]
macro_rules! assert_nth_scalar {
    ($arr:expr, $n:expr, $expected:expr) => {
        assert_eq!($arr.scalar_at($n).unwrap(), $expected.try_into().unwrap());
    };
}

/// Asserts that the scalar at position `$n` in array `$arr` is null.
///
/// # Example
///
/// ```ignore
/// let arr = PrimitiveArray::from_option_iter([Some(1), None, Some(3)]);
/// assert_nth_scalar_null!(arr, 1);
/// ```
#[macro_export]
macro_rules! assert_nth_scalar_is_null {
    ($arr:expr, $n:expr) => {
        assert!(
            $arr.scalar_at($n).unwrap().is_null(),
            "expected scalar at index {} to be null, but was {:?}",
            $n,
            $arr.scalar_at($n).unwrap()
        );
    };
}

#[macro_export]
macro_rules! assert_arrays_eq {
    ($left:expr, $right:expr) => {{
        let left = $left.clone();
        let right = $right.clone();
        assert_eq!(
            left.dtype(),
            right.dtype(),
            "assertion left == right failed: arrays differ in type: {} != {}.\n  left: {}\n right: {}",
            left.dtype(),
            right.dtype(),
            left.display_values(),
            right.display_values()
        );

        assert_eq!(
            left.len(),
            right.len(),
            "assertion left == right failed: arrays differ in length: {} != {}.\n  left: {}\n right: {}",
            left.len(),
            right.len(),
            left.display_values(),
            right.display_values()
        );

        #[allow(deprecated)]
        let left = left.to_array();
        #[allow(deprecated)]
        let right = right.to_array();
        $crate::arrays::assert_arrays_eq_impl(&left, &right);
    }};
}

/// Implementation of `assert_arrays_eq!` — called by the macro after converting inputs to
/// `ArrayRef`.
#[track_caller]
#[allow(clippy::panic)]
pub fn assert_arrays_eq_impl(left: &ArrayRef, right: &ArrayRef) {
    let executed = execute_to_canonical(left.clone(), &mut LEGACY_SESSION.create_execution_ctx());

    let left_right = find_mismatched_indices(left, right);
    let executed_right = find_mismatched_indices(&executed, right);

    if !left_right.is_empty() || !executed_right.is_empty() {
        let mut msg = String::new();
        if !left_right.is_empty() {
            msg.push_str(&format!(
                "\n  left != right at indices: {}",
                format_indices(left_right)
            ));
        }
        if !executed_right.is_empty() {
            msg.push_str(&format!(
                "\n  executed != right at indices: {}",
                format_indices(executed_right)
            ));
        }
        panic!(
            "assertion failed: arrays do not match:{}\n     left: {}\n    right: {}\n executed: {}",
            msg,
            left.display_values(),
            right.display_values(),
            executed.display_values()
        )
    }
}
