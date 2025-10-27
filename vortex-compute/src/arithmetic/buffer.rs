// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};

use crate::arithmetic::{Checked, CheckedOperator};

/// Implementation that attempts to downcast to a mutable buffer and operates in-place.
impl<Op, T> Checked<Op, &Buffer<T>> for Buffer<T>
where
    T: Copy + num_traits::Zero,
    BufferMut<T>: for<'a> Checked<Op, &'a Buffer<T>, Output = Buffer<T>>,
    for<'a> &'a Buffer<T>: Checked<Op, &'a Buffer<T>, Output = Buffer<T>>,
{
    type Output = Buffer<T>;

    fn checked_op(self, rhs: &Buffer<T>) -> Option<Self::Output> {
        match self.try_into_mut() {
            Ok(lhs) => lhs.checked_op(rhs),
            Err(lhs) => (&lhs).checked_op(rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable buffer.
impl<Op, T> Checked<Op, &Buffer<T>> for BufferMut<T>
where
    T: Copy + num_traits::Zero,
    Op: CheckedOperator<T>,
{
    type Output = Buffer<T>;

    fn checked_op(self, rhs: &Buffer<T>) -> Option<Self::Output> {
        assert_eq!(self.len(), rhs.len());

        let mut i = 0;
        let mut overflow = false;
        let buffer = self
            .map_each(|a| {
                // SAFETY: lengths are equal, so index is in bounds
                let b = unsafe { *rhs.get_unchecked(i) };
                i += 1;

                // On overflow, set flag and write zero
                // We don't abort early because this code vectorizes better without the
                // branch, and we expect overflow to be an exception rather than the norm.
                Op::apply(&a, &b).unwrap_or_else(|| {
                    overflow = true;
                    T::zero()
                })
            })
            .freeze();

        (!overflow).then_some(buffer)
    }
}

/// Implementation that allocates a new output buffer.
impl<Op, T> Checked<Op, &Buffer<T>> for &Buffer<T>
where
    T: Copy + num_traits::Zero,
    Op: CheckedOperator<T>,
{
    type Output = Buffer<T>;

    fn checked_op(self, rhs: &Buffer<T>) -> Option<Self::Output> {
        assert_eq!(self.len(), rhs.len());

        let mut overflow = false;
        let buffer =
            Buffer::<T>::from_trusted_len_iter(self.iter().zip(rhs.iter()).map(|(a, b)| {
                // On overflow, set flag and write zero
                // We don't abort early because this code vectorizes better without the
                // branch, and we expect overflow to be an exception rather than the norm.
                Op::apply(a, b).unwrap_or_else(|| {
                    overflow = true;
                    T::zero()
                })
            }));
        (!overflow).then_some(buffer)
    }
}

/// Implementation that attempts to downcast to a mutable buffer and operates in-place against
/// a scalar RHS value.
impl<Op, T> Checked<Op, &T> for Buffer<T>
where
    T: Copy + num_traits::Zero,
    BufferMut<T>: for<'a> Checked<Op, &'a T, Output = Buffer<T>>,
    for<'a> &'a Buffer<T>: Checked<Op, &'a T, Output = Buffer<T>>,
{
    type Output = Buffer<T>;

    fn checked_op(self, rhs: &T) -> Option<Self::Output> {
        match self.try_into_mut() {
            Ok(lhs) => lhs.checked_op(rhs),
            Err(lhs) => (&lhs).checked_op(rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable buffer against a scalar RHS value.
impl<Op, T> Checked<Op, &T> for BufferMut<T>
where
    T: Copy + num_traits::Zero,
    Op: CheckedOperator<T>,
{
    type Output = Buffer<T>;

    fn checked_op(self, rhs: &T) -> Option<Self::Output> {
        let mut overflow = false;
        let buffer = self
            .map_each(|a| {
                Op::apply(&a, rhs).unwrap_or_else(|| {
                    overflow = true;
                    T::zero()
                })
            })
            .freeze();

        (!overflow).then_some(buffer)
    }
}

/// Implementation that allocates a new output buffer operating against a scalar RHS value.
impl<Op, T> Checked<Op, &T> for &Buffer<T>
where
    T: Copy + num_traits::Zero,
    Op: CheckedOperator<T>,
{
    type Output = Buffer<T>;

    fn checked_op(self, rhs: &T) -> Option<Self::Output> {
        let mut overflow = false;
        let buffer = Buffer::<T>::from_trusted_len_iter(self.iter().map(|a| {
            Op::apply(a, rhs).unwrap_or_else(|| {
                overflow = true;
                T::zero()
            })
        }));

        (!overflow).then_some(buffer)
    }
}

#[cfg(test)]
mod tests {
    use crate::arithmetic::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub};
    use vortex_buffer::buffer;

    #[test]
    fn test_add_buffers() {
        let left = buffer![1u32, 2, 3, 4];
        let right = buffer![10u32, 20, 30, 40];

        let result = left.checked_add(&right).unwrap();
        assert_eq!(result, buffer![11u32, 22, 33, 44]);
    }

    #[test]
    fn test_add_scalar() {
        let buf = buffer![1u32, 2, 3, 4];
        let result = buf.checked_add(&10).unwrap();
        assert_eq!(result, buffer![11u32, 12, 13, 14]);
    }

    #[test]
    fn test_add_overflow() {
        let left = buffer![u8::MAX, 100];
        let right = buffer![1u8, 50];

        let result = left.checked_add(&right);
        assert!(result.is_none());
    }

    #[test]
    fn test_sub_buffers() {
        let left = buffer![10u32, 20, 30, 40];
        let right = buffer![1u32, 2, 3, 4];

        let result = left.checked_sub(&right).unwrap();
        assert_eq!(result, buffer![9u32, 18, 27, 36]);
    }

    #[test]
    fn test_sub_scalar() {
        let buf = buffer![10u32, 20, 30, 40];
        let result = buf.checked_sub(&5).unwrap();
        assert_eq!(result, buffer![5u32, 15, 25, 35]);
    }

    #[test]
    fn test_sub_underflow() {
        let left = buffer![5u32, 10];
        let right = buffer![10u32, 5];

        let result = left.checked_sub(&right);
        assert!(result.is_none());
    }

    #[test]
    fn test_mul_buffers() {
        let left = buffer![2u32, 3, 4, 5];
        let right = buffer![10u32, 20, 30, 40];

        let result = left.checked_mul(&right).unwrap();
        assert_eq!(result, buffer![20u32, 60, 120, 200]);
    }

    #[test]
    fn test_mul_scalar() {
        let buf = buffer![1u32, 2, 3, 4];
        let result = buf.checked_mul(&10).unwrap();
        assert_eq!(result, buffer![10u32, 20, 30, 40]);
    }

    #[test]
    fn test_mul_overflow() {
        let left = buffer![u8::MAX, 100];
        let right = buffer![2u8, 3];

        let result = left.checked_mul(&right);
        assert!(result.is_none());
    }

    #[test]
    fn test_div_buffers() {
        let left = buffer![100u32, 200, 300, 400];
        let right = buffer![10u32, 20, 30, 40];

        let result = left.checked_div(&right).unwrap();
        assert_eq!(result, buffer![10u32, 10, 10, 10]);
    }

    #[test]
    fn test_div_scalar() {
        let buf = buffer![100u32, 200, 300, 400];
        let result = buf.checked_div(&10).unwrap();
        assert_eq!(result, buffer![10u32, 20, 30, 40]);
    }

    #[test]
    fn test_div_by_zero() {
        let left = buffer![10u32, 20, 30];
        let right = buffer![2u32, 0, 3];

        let result = left.checked_div(&right);
        assert!(result.is_none());
    }

    #[test]
    fn test_div_scalar_by_zero() {
        let buf = buffer![10u32, 20, 30];
        let result = buf.checked_div(&0);
        assert!(result.is_none());
    }
}
