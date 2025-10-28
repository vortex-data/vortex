// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};

use crate::arithmetic::{Arithmetic, Operator};

/// Implementation that attempts to downcast to a mutable buffer and operates in-place.
impl<Op, T> Arithmetic<Op, &Buffer<T>> for Buffer<T>
where
    T: Copy,
    BufferMut<T>: for<'a> Arithmetic<Op, &'a Buffer<T>, Output = Buffer<T>>,
    for<'a> &'a Buffer<T>: Arithmetic<Op, &'a Buffer<T>, Output = Buffer<T>>,
{
    type Output = Buffer<T>;

    fn eval(self, rhs: &Buffer<T>) -> Self::Output {
        match self.try_into_mut() {
            Ok(lhs) => lhs.eval(rhs),
            Err(lhs) => (&lhs).eval(rhs), // (&lhs) to delegate to borrowed impl
        }
    }
}

/// Implementation that operates in-place over a mutable buffer.
impl<Op, T> Arithmetic<Op, &Buffer<T>> for BufferMut<T>
where
    T: Copy + num_traits::Zero,
    Op: Operator<T>,
{
    type Output = Buffer<T>;

    fn eval(self, rhs: &Buffer<T>) -> Self::Output {
        assert_eq!(self.len(), rhs.len());

        let mut i = 0;
        self.map_each_in_place(|a| {
            // SAFETY: lengths are equal, so index is in bounds
            let b = unsafe { *rhs.get_unchecked(i) };
            i += 1;

            Op::apply(&a, &b)
        })
        .freeze()
    }
}

/// Implementation that allocates a new output buffer.
impl<Op, T> Arithmetic<Op> for &Buffer<T>
where
    Op: Operator<T>,
{
    type Output = Buffer<T>;

    fn eval(self, rhs: &Buffer<T>) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        Buffer::<T>::from_trusted_len_iter(
            self.iter().zip(rhs.iter()).map(|(a, b)| Op::apply(a, b)),
        )
    }
}

/// Implementation that attempts to downcast to a mutable buffer and operates in-place against
/// a scalar RHS value.
impl<Op, T> Arithmetic<Op, &T> for Buffer<T>
where
    BufferMut<T>: for<'a> Arithmetic<Op, &'a T, Output = Buffer<T>>,
    for<'a> &'a Buffer<T>: Arithmetic<Op, &'a T, Output = Buffer<T>>,
{
    type Output = Buffer<T>;

    fn eval(self, rhs: &T) -> Self::Output {
        match self.try_into_mut() {
            Ok(lhs) => lhs.eval(rhs),
            Err(lhs) => (&lhs).eval(rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable buffer against a scalar RHS value.
impl<Op, T> Arithmetic<Op, &T> for BufferMut<T>
where
    T: Copy,
    Op: Operator<T>,
{
    type Output = Buffer<T>;

    fn eval(self, rhs: &T) -> Self::Output {
        self.map_each_in_place(|a| Op::apply(&a, rhs)).freeze()
    }
}

/// Implementation that allocates a new output buffer operating against a scalar RHS value.
impl<Op, T> Arithmetic<Op, &T> for &Buffer<T>
where
    Op: Operator<T>,
{
    type Output = Buffer<T>;

    fn eval(self, rhs: &T) -> Self::Output {
        Buffer::<T>::from_trusted_len_iter(self.iter().map(|a| Op::apply(a, rhs)))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::arithmetic::{Arithmetic, WrappingAdd, WrappingMul, WrappingSub};

    #[test]
    fn test_add_buffers() {
        let left = buffer![1u32, 2, 3, 4];
        let right = buffer![10u32, 20, 30, 40];

        let result = Arithmetic::<WrappingAdd, _>::eval(left, &right);
        assert_eq!(result, buffer![11u32, 22, 33, 44]);
    }

    #[test]
    fn test_add_scalar() {
        let buf = buffer![1u32, 2, 3, 4];
        let result = Arithmetic::<WrappingAdd, _>::eval(buf, &10);
        assert_eq!(result, buffer![11u32, 12, 13, 14]);
    }

    #[test]
    fn test_sub_buffers() {
        let left = buffer![10u32, 20, 30, 40];
        let right = buffer![1u32, 2, 3, 4];

        let result = Arithmetic::<WrappingSub, _>::eval(left, &right);
        assert_eq!(result, buffer![9u32, 18, 27, 36]);
    }

    #[test]
    fn test_sub_scalar() {
        let buf = buffer![10u32, 20, 30, 40];
        let result = Arithmetic::<WrappingSub, _>::eval(buf, &5);
        assert_eq!(result, buffer![5u32, 15, 25, 35]);
    }

    #[test]
    fn test_mul_buffers() {
        let left = buffer![2u32, 3, 4, 5];
        let right = buffer![10u32, 20, 30, 40];

        let result = Arithmetic::<WrappingMul, _>::eval(left, &right);
        assert_eq!(result, buffer![20u32, 60, 120, 200]);
    }

    #[test]
    fn test_mul_scalar() {
        let buf = buffer![1u32, 2, 3, 4];
        let result = Arithmetic::<WrappingMul, _>::eval(buf, &10);
        assert_eq!(result, buffer![10u32, 20, 30, 40]);
    }
}
