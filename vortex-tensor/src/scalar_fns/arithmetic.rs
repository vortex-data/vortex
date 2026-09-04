// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-level arithmetic shared by tensor scalar functions.
//!
//! These kernels preserve the scalar functions' left-to-right floating-point arithmetic. Keeping
//! the arithmetic here ensures that fused and single-result operations use the same result contract.

use num_traits::Float;
use vortex_array::dtype::NativePType;

/// Computes `sqrt(sum(value_i^2))` for one row.
///
/// An empty or all-zero row produces `0.0`. Overflow, underflow, and non-finite values follow the
/// input element type's arithmetic.
pub(crate) fn l2_norm_row<T: Float + NativePType>(row: &[T]) -> T {
    let mut sum_squared = T::zero();
    for &value in row {
        sum_squared = sum_squared + value * value;
    }

    sum_squared.sqrt()
}
