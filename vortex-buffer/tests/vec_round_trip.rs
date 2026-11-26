// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zero-copy conversions between buffers and `Vec<T>`, as asked for in
//! <https://github.com/vortex-data/vortex/issues/4719>.

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;

    #[test]
    fn vec_round_trips_through_buffer_without_copying() -> VortexResult<()> {
        let values: Vec<i32> = (0..1024).collect();
        let (ptr, capacity) = (values.as_ptr(), values.capacity());

        let buffer = Buffer::from(values);
        assert_eq!(buffer.as_ptr(), ptr);

        let values = buffer
            .try_into_vec()
            .map_err(|_| vortex_err!("the sole handle to a Vec's memory must hand it back"))?;
        assert_eq!(values.as_ptr(), ptr);
        assert_eq!(values.capacity(), capacity);
        assert_eq!(values, (0..1024).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn vec_round_trips_through_buffer_mut_with_its_spare_capacity() -> VortexResult<()> {
        let mut values = Vec::with_capacity(16);
        values.extend([1u64, 2, 3]);
        let ptr = values.as_ptr();

        let mut buffer = BufferMut::from(values);
        assert_eq!(buffer.capacity(), 16);
        buffer.extend_from_slice(&[4, 5]);
        assert_eq!(
            buffer.as_ptr(),
            ptr,
            "filling spare capacity must not move the data"
        );

        let values = buffer
            .try_into_vec()
            .map_err(|_| vortex_err!("a buffer adopted from a Vec must hand it back"))?;
        assert_eq!(values.as_ptr(), ptr);
        assert_eq!(values.capacity(), 16);
        assert_eq!(values, [1, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn growth_keeps_a_buffer_adopted_from_a_vec_returnable() -> VortexResult<()> {
        let mut buffer = BufferMut::from(vec![7u32; 3]);
        buffer.extend(std::iter::repeat_n(9u32, 1000));

        let values = buffer
            .try_into_vec()
            .map_err(|_| vortex_err!("growth must keep the memory freeable as a Vec"))?;
        assert_eq!(values[..3], [7; 3]);
        assert_eq!(values[3..], [9; 1000]);
        Ok(())
    }

    #[test]
    fn shared_and_sliced_buffers_are_not_handed_over() -> VortexResult<()> {
        let buffer = Buffer::from(vec![1u16, 2, 3, 4]);
        let clone = buffer.clone();
        let Err(buffer) = buffer.try_into_vec() else {
            vortex_bail!("a clone still reads the memory, so it cannot be handed over");
        };
        assert_eq!(buffer, clone);
        drop(clone);

        let tail = buffer.slice(1..);
        drop(buffer);
        let Err(tail) = tail.try_into_vec() else {
            vortex_bail!("a Vec has to start at the front of its memory");
        };
        assert_eq!(tail.as_slice(), &[2, 3, 4]);
        assert_eq!(tail.into_vec(), [2, 3, 4]);
        Ok(())
    }

    #[test]
    fn buffers_vortex_allocated_are_copied_out() {
        // Vortex lays out its own allocations for over-alignment, which a `Vec<u32>` cannot free.
        let buffer = Buffer::<u32>::copy_from([1, 2, 3]);
        let ptr = buffer.as_ptr();
        let values = buffer.into_vec();
        assert_ne!(values.as_ptr(), ptr);
        assert_eq!(values, [1, 2, 3]);
    }

    #[test]
    fn empty_buffers_become_empty_vecs() -> VortexResult<()> {
        let values = Buffer::<u32>::empty()
            .try_into_vec()
            .map_err(|_| vortex_err!("an empty buffer owns nothing to copy"))?;
        assert!(values.is_empty());
        Ok(())
    }
}
