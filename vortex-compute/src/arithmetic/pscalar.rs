// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativePType;
use vortex_vector::primitive::PScalar;

use crate::arithmetic::{Arithmetic, Operator};

impl<Op, T> Arithmetic<Op> for &PScalar<T>
where
    T: NativePType,
    Op: Operator<T>,
    for<'a> &'a T: Arithmetic<Op, &'a T, Output = T>,
{
    type Output = PScalar<T>;

    fn eval(self, rhs: &PScalar<T>) -> Self::Output {
        match (self.value(), rhs.value()) {
            (Some(a), Some(b)) => {
                let value = Arithmetic::<Op, _>::eval(a, b);
                PScalar::new(Some(value))
            }
            (..) => {
                // At least one side is null, so result is null
                PScalar::new(None)
            }
        }
    }
}
