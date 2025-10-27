// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_vector::PVector;
use crate::arithmetic::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub};

impl<T: Copy + num_traits::CheckedAdd + num_traits::Zero> CheckedAdd<&PVector<T>> for Buffer<T> {
    type Output = Self;

    fn checked_add(self, other: &Buffer<T>) -> Option<Self::Output> {
        buffer_op_inplace(self, other, |a, b| a.checked_add(b))
    }
}

impl<T: Copy + num_traits::CheckedAdd + num_traits::Zero> CheckedAdd<&T> for Buffer<T> {
    type Output = Self;

    fn checked_add(self, other: &T) -> Option<Self::Output> {
        buffer_op_inplace_scalar(self, other, |a, b| a.checked_add(b))
    }
}

impl<T: Copy + num_traits::CheckedSub + num_traits::Zero> CheckedSub<&Buffer<T>> for Buffer<T> {
    type Output = Self;

    fn checked_sub(self, other: &Buffer<T>) -> Option<Self::Output> {
        buffer_op_inplace(self, other, |a, b| a.checked_sub(b))
    }
}

impl<T: Copy + num_traits::CheckedSub + num_traits::Zero> CheckedSub<&T> for Buffer<T> {
    type Output = Self;

    fn checked_sub(self, other: &T) -> Option<Self::Output> {
        buffer_op_inplace_scalar(self, other, |a, b| a.checked_sub(b))
    }
}

impl<T: Copy + num_traits::CheckedMul + num_traits::Zero> CheckedMul<&Buffer<T>> for Buffer<T> {
    type Output = Self;

    fn checked_mul(self, other: &Buffer<T>) -> Option<Self::Output> {
        buffer_op_inplace(self, other, |a, b| a.checked_mul(b))
    }
}

impl<T: Copy + num_traits::CheckedMul + num_traits::Zero> CheckedMul<&T> for Buffer<T> {
    type Output = Self;

    fn checked_mul(self, other: &T) -> Option<Self::Output> {
        buffer_op_inplace_scalar(self, other, |a, b| a.checked_mul(b))
    }
}

impl<T: Copy + num_traits::CheckedDiv + num_traits::Zero> CheckedDiv<&Buffer<T>> for Buffer<T> {
    type Output = Self;

    fn checked_div(self, other: &Buffer<T>) -> Option<Self::Output> {
        buffer_op_inplace(self, other, |a, b| a.checked_div(b))
    }
}

impl<T: Copy + num_traits::CheckedDiv + num_traits::Zero> CheckedDiv<&T> for Buffer<T> {
    type Output = Self;

    fn checked_div(self, other: &T) -> Option<Self::Output> {
        buffer_op_inplace_scalar(self, other, |a, b| a.checked_div(b))
    }
}

fn buffer_op_inplace<O, T>(lhs: Buffer<T>, rhs: &Buffer<T>, op: O) -> Option<Buffer<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: Copy + num_traits::Zero,
{
    match lhs.try_into_mut() {
        Ok(lhs) => buffer_op_mut(lhs, rhs, op),
        Err(lhs) => buffer_op(&lhs, rhs, op),
    }
}

fn buffer_op_mut<O, T>(lhs: BufferMut<T>, rhs: &Buffer<T>, op: O) -> Option<Buffer<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: Copy + num_traits::Zero,
{
    assert_eq!(lhs.len(), rhs.len());

    let mut i = 0;
    let mut overflow = false;
    let buffer = lhs
        .map_each(|a| {
            // SAFETY: lengths are equal, so index is in bounds
            let b = unsafe { *rhs.get_unchecked(i) };
            i += 1;

            // On overflow, set flag and write zero
            // We don't abort early because this code vectorizes better without the
            // branch, and we expect overflow to be an exception rather than the norm.
            op(&a, &b).unwrap_or_else(|| {
                overflow = true;
                T::zero()
            })
        })
        .freeze();

    (!overflow).then_some(buffer)
}

fn buffer_op<O, T>(lhs: &Buffer<T>, rhs: &Buffer<T>, op: O) -> Option<Buffer<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: Copy + num_traits::Zero,
{
    assert_eq!(lhs.len(), rhs.len());

    let mut overflow = false;
    let buffer = Buffer::<T>::from_trusted_len_iter(lhs.iter().zip(rhs.iter()).map(|(a, b)| {
        // On overflow, set flag and write zero
        // We don't abort early because this code vectorizes better without the
        // branch, and we expect overflow to be an exception rather than the norm.
        op(a, b).unwrap_or_else(|| {
            overflow = true;
            T::zero()
        })
    }));
    (!overflow).then_some(buffer)
}

fn buffer_op_inplace_scalar<O, T>(lhs: Buffer<T>, rhs: &T, op: O) -> Option<Buffer<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: Copy + num_traits::Zero,
{
    match lhs.try_into_mut() {
        Ok(lhs) => buffer_op_mut_scalar(lhs, rhs, op),
        Err(lhs) => buffer_op_scalar(&lhs, rhs, op),
    }
}

fn buffer_op_mut_scalar<O, T>(lhs: BufferMut<T>, rhs: &T, op: O) -> Option<Buffer<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: Copy + num_traits::Zero,
{
    let mut overflow = false;
    let buffer = lhs
        .map_each(|a| {
            op(&a, rhs).unwrap_or_else(|| {
                overflow = true;
                T::zero()
            })
        })
        .freeze();

    (!overflow).then_some(buffer)
}

fn buffer_op_scalar<O, T>(lhs: &Buffer<T>, rhs: &T, op: O) -> Option<Buffer<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: Copy + num_traits::Zero,
{
    let mut overflow = false;
    let buffer = Buffer::<T>::from_trusted_len_iter(lhs.iter().map(|a| {
        op(a, rhs).unwrap_or_else(|| {
            overflow = true;
            T::zero()
        })
    }));
    (!overflow).then_some(buffer)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;

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
