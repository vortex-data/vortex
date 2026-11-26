// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Buffers over memory adopted from a foreign owner, read-only and writable.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::OnceLock;

    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;

    /// An owner that records the values it holds at the moment it is dropped.
    struct Recorder {
        values: Vec<u32>,
        dropped_with: Arc<OnceLock<Vec<u32>>>,
    }

    impl Recorder {
        fn new(values: Vec<u32>) -> (Self, Arc<OnceLock<Vec<u32>>>) {
            let dropped_with = Arc::new(OnceLock::new());
            let owner = Self {
                values,
                dropped_with: Arc::clone(&dropped_with),
            };
            (owner, dropped_with)
        }
    }

    impl AsRef<[u32]> for Recorder {
        fn as_ref(&self) -> &[u32] {
            &self.values
        }
    }

    impl AsMut<[u32]> for Recorder {
        fn as_mut(&mut self) -> &mut [u32] {
            &mut self.values
        }
    }

    impl Drop for Recorder {
        fn drop(&mut self) {
            assert!(
                self.dropped_with.set(self.values.clone()).is_ok(),
                "an owner is released exactly once"
            );
        }
    }

    #[test]
    fn read_only_owners_live_as_long_as_the_last_buffer() -> VortexResult<()> {
        let (owner, record) = Recorder::new(vec![1, 2, 3, 4]);
        let ptr = owner.values.as_ptr();

        let buffer = Buffer::from_owner(owner);
        assert_eq!(buffer.as_ptr(), ptr, "adopting an owner must not copy");
        let tail = buffer.slice(2..);
        drop(buffer);
        assert_eq!(record.get().cloned(), None);

        // Only reads were promised, so the memory is never handed out for writing.
        let Err(tail) = tail.try_into_mut() else {
            vortex_bail!("a read-only owner must not become writable");
        };
        assert_eq!(tail.as_slice(), &[3, 4]);
        drop(tail);
        assert_eq!(record.get().cloned(), Some(vec![1, 2, 3, 4]));
        Ok(())
    }

    #[test]
    fn writable_owners_are_written_in_place() -> VortexResult<()> {
        let (owner, record) = Recorder::new(vec![1, 2, 3]);
        let ptr = owner.values.as_ptr();

        let mut buffer = BufferMut::from_owner(owner);
        assert_eq!(buffer.as_ptr(), ptr);
        assert_eq!(buffer.capacity(), 3);
        buffer[0] = 10;

        // Frozen and thawed again, it is still the owner's memory.
        let mut buffer = buffer
            .freeze()
            .try_into_mut()
            .map_err(|_| vortex_err!("the sole handle to a writable owner can write again"))?;
        assert_eq!(buffer.as_ptr(), ptr);
        buffer[1] = 20;

        drop(buffer);
        assert_eq!(record.get().cloned(), Some(vec![10, 20, 3]));
        Ok(())
    }

    #[test]
    fn growing_past_an_owner_moves_out_and_releases_it() -> VortexResult<()> {
        let (owner, record) = Recorder::new(vec![1, 2, 3]);
        let mut buffer = BufferMut::from_owner(owner);

        buffer.push(4);
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(
            record.get().cloned(),
            Some(vec![1, 2, 3]),
            "the owner is released as soon as the data moves out"
        );

        buffer.shrink_to_fit();
        assert_eq!(buffer.into_vec(), [1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn adopted_memory_is_never_handed_out_as_a_vec() -> VortexResult<()> {
        let (owner, _released) = Recorder::new(vec![1, 2, 3]);
        let ptr = owner.values.as_ptr();
        let Err(buffer) = BufferMut::from_owner(owner).try_into_vec() else {
            vortex_bail!("only the owner may free its memory");
        };
        assert_eq!(buffer.as_ptr(), ptr);
        assert_eq!(buffer.into_vec(), [1, 2, 3]);
        Ok(())
    }

    /// An owner whose values live inside it rather than behind a pointer of its own.
    struct Inline([u32; 4]);

    impl AsMut<[u32]> for Inline {
        fn as_mut(&mut self) -> &mut [u32] {
            &mut self.0
        }
    }

    #[test]
    fn owners_holding_their_values_inline_stay_put() -> VortexResult<()> {
        // The buffer points into the owner itself, so the owner must never move once adopted.
        // Miri checks that nothing invalidates the pointer either.
        let mut buffer = BufferMut::from_owner(Inline([1, 2, 3, 4]));
        buffer[3] = 40;
        let frozen = buffer.freeze();
        let clone = frozen.clone();
        assert_eq!(clone.as_slice(), &[1, 2, 3, 40]);
        drop(clone);

        let mut buffer = frozen
            .try_into_mut()
            .map_err(|_| vortex_err!("the sole handle can write again"))?;
        buffer[0] = 10;
        assert_eq!(buffer.as_slice(), &[10, 2, 3, 40]);
        Ok(())
    }

    #[test]
    fn shrinking_leaves_an_owner_alone() -> VortexResult<()> {
        let (owner, record) = Recorder::new(vec![1, 2, 3, 4]);
        let mut buffer = BufferMut::from_owner(owner);
        buffer.truncate(1);
        buffer.shrink_to_fit();
        assert_eq!(buffer.as_slice(), &[1]);
        drop(buffer);
        assert_eq!(record.get().cloned(), Some(vec![1, 2, 3, 4]));
        Ok(())
    }
}
