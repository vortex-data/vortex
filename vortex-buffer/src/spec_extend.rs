use std::iter::TrustedLen;
use std::slice;

use vortex_error::VortexExpect;

use crate::BufferMut;

impl<T> BufferMut<T> {
    fn extend_iter<I: Iterator<Item = T>>(&mut self, mut iter: I) {
        // Attempt to reserve enough memory up-front, although this is only a lower bound.
        let (lower, _) = iter.size_hint();
        self.reserve(lower);

        let remaining = self.capacity() - self.len();

        let begin: *const T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
        let mut dst: *mut T = begin.cast_mut();
        for _ in 0..remaining {
            if let Some(item) = iter.next() {
                unsafe {
                    // SAFETY: We know we have enough capacity to write the item.
                    dst.write(item);
                    // Note. we used to have dst.add(iteration).write(item), here.
                    // however this was much slower than just incrementing dst.
                    dst = dst.add(1);
                }
            } else {
                break;
            }
        }

        // TODO(joe): replace with ptr_sub when stable
        let length = self.len() + unsafe { dst.byte_offset_from(begin) as usize / size_of::<T>() };
        unsafe { self.set_len(length) };

        // Append remaining elements
        iter.for_each(|item| self.push(item));
    }

    fn extend_trusted<I: TrustedLen<Item = T>>(&mut self, iter: I) {
        // Reserve all memory upfront since it's an exact upper bound
        let (_, high) = iter.size_hint();
        self.reserve(high.vortex_expect("TrustedLen iterator didn't have valid upper bound"));

        let begin: *const T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
        let mut dst: *mut T = begin.cast_mut();
        iter.for_each(|item| {
            unsafe {
                // SAFETY: We know we have enough capacity to write the item.
                dst.write(item);
                // Note. we used to have dst.add(iteration).write(item), here.
                // however this was much slower than just incrementing dst.
                dst = dst.add(1);
            }
        });
        // TODO(joe): replace with ptr_sub when stable
        let length = self.len() + unsafe { dst.byte_offset_from(begin) as usize / size_of::<T>() };
        unsafe { self.set_len(length) };
    }
}

// Specialization trait used for BufferMut::extend
pub(super) trait SpecExtend<T, I> {
    #[track_caller]
    fn spec_extend(&mut self, iter: I);
}

impl<T, I> SpecExtend<T, I> for BufferMut<T>
where
    I: Iterator<Item = T>,
{
    #[track_caller]
    default fn spec_extend(&mut self, iter: I) {
        self.extend_iter(iter)
    }
}

impl<T, I> SpecExtend<T, I> for BufferMut<T>
where
    I: TrustedLen<Item = T>,
{
    #[track_caller]
    default fn spec_extend(&mut self, iterator: I) {
        self.extend_trusted(iterator)
    }
}

impl<'a, T: 'a> SpecExtend<&'a T, slice::Iter<'a, T>> for BufferMut<T>
where
    T: Copy,
{
    #[track_caller]
    fn spec_extend(&mut self, iterator: slice::Iter<'a, T>) {
        let slice = iterator.as_slice();
        self.extend_from_slice(slice);
    }
}
