// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativePType;
use vortex_vector::primitive::PScalar;

use crate::arithmetic::{Arithmetic, CheckedArithmetic, CheckedOperator, Operator};

impl<Op, T> Arithmetic<Op> for &PScalar<T>
where
    T: NativePType,
    Op: Operator<T>,
{
    type Output = PScalar<T>;

    fn eval(self, rhs: &PScalar<T>) -> Self::Output {
        match (self.value(), rhs.value()) {
            (Some(a), Some(b)) => {
                let value = Op::apply(&a, &b);
                PScalar::new(Some(value))
            }
            (..) => {
                // At least one side is null, so result is null
                PScalar::new(None)
            }
        }
    }
}

impl<Op, T> CheckedArithmetic<Op> for &PScalar<T>
where
    T: NativePType,
    Op: CheckedOperator<T>,
{
    type Output = PScalar<T>;

    fn checked_eval(self, rhs: Self) -> Option<Self::Output> {
        match (self.value(), rhs.value()) {
            (Some(a), Some(b)) => {
                let value = Op::apply(&a, &b)?;
                Some(PScalar::new(Some(value)))
            }
            (..) => {
                // At least one side is null, so result is null
                Some(PScalar::new(None))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_vector::primitive::PScalar;

    use super::*;
    use crate::arithmetic::{Add, CheckedArithmetic, WrappingSub};

    #[test]
    fn test_add() {
        let left = PScalar::new(Some(5u32));
        let right = PScalar::new(Some(3u32));

        let result = CheckedArithmetic::<Add>::checked_eval(&left, &right).unwrap();
        assert_eq!(result.value(), Some(8u32));

        let left_null = PScalar::new(None);
        let result_null = CheckedArithmetic::<Add>::checked_eval(&left_null, &right).unwrap();
        assert_eq!(result_null.value(), None);
    }

    #[test]
    fn test_subtract() {
        let left = PScalar::new(Some(10u32));
        let right = PScalar::new(Some(4u32));

        let result = Arithmetic::<WrappingSub>::eval(&left, &right);
        assert_eq!(result.value(), Some(6u32));

        let right_null = PScalar::new(None);
        let result_null = Arithmetic::<WrappingSub>::eval(&left, &right_null);
        assert_eq!(result_null.value(), None);
    }
}
