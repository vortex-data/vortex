// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use itertools::Itertools;

pub fn format_indices<I: IntoIterator<Item = usize>>(indices: I) -> impl Display {
    indices.into_iter().format(",")
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
        if left.dtype() != right.dtype() {
            panic!(
                "assertion left == right failed: arrays differ in type: {} != {}.\n  left: {}\n right: {}",
                left.dtype(),
                right.dtype(),
                left.display_values(),
                right.display_values()
            )
        }

        if left.len() != right.len() {
            panic!(
                "assertion left == right failed: arrays differ in length: {} != {}.\n  left: {}\n right: {}",
                left.len(),
                right.len(),
                left.display_values(),
                right.display_values()
            )
        }

        let executed = {
            use $crate::IntoArray;
            use $crate::VortexSessionExecute;
            let mut ctx = $crate::LEGACY_SESSION.create_execution_ctx();
            // Allow deprecated to_array() as it's the only method that works uniformly
            // for all input types (ArrayRef, concrete arrays, and &dyn DynArray).
            #[allow(deprecated)]
            let arr = left.to_array();
            arr.execute::<$crate::RecursiveCanonical>(&mut ctx)
                .expect("assert_arrays_eq: failed to execute left array to recursive canonical form")
                .0
                .into_array()
        };

        let n = left.len();
        let left_right_mismatched: Vec<usize> = (0..n)
            .filter(|i| left.scalar_at(*i).unwrap() != right.scalar_at(*i).unwrap())
            .collect();
        let left_executed_mismatched: Vec<usize> = (0..n)
            .filter(|i| left.scalar_at(*i).unwrap() != executed.scalar_at(*i).unwrap())
            .collect();
        let right_executed_mismatched: Vec<usize> = (0..n)
            .filter(|i| right.scalar_at(*i).unwrap() != executed.scalar_at(*i).unwrap())
            .collect();

        if !left_right_mismatched.is_empty()
            || !left_executed_mismatched.is_empty()
            || !right_executed_mismatched.is_empty()
        {
            let mut msg = String::new();
            if !left_right_mismatched.is_empty() {
                msg.push_str(&format!(
                    "\n  left != right at indices: {}",
                    $crate::arrays::format_indices(left_right_mismatched)
                ));
            }
            if !left_executed_mismatched.is_empty() {
                msg.push_str(&format!(
                    "\n  left != executed at indices: {}",
                    $crate::arrays::format_indices(left_executed_mismatched)
                ));
            }
            if !right_executed_mismatched.is_empty() {
                msg.push_str(&format!(
                    "\n  right != executed at indices: {}",
                    $crate::arrays::format_indices(right_executed_mismatched)
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
    }};
}
