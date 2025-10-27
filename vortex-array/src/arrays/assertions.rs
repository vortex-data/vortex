// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
        let n = left.len();
        let mismatched_indices = (0..n)
            .filter(|i| left.scalar_at(*i) != right.scalar_at(*i))
            .collect::<Vec<_>>();
        if mismatched_indices.len() != 0 {
            panic!(
                "assertion left == right failed: arrays do not match at indices: {}.\n  left: {}\n right: {}",
                itertools::Itertools::format(mismatched_indices.into_iter(), ", "),
                left.display_values(),
                right.display_values()
            )
        }
    }};
}
