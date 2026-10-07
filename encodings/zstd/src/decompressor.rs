// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A zstd decompression context reused across decodes on the same thread.

use std::cell::RefCell;

use vortex_error::VortexResult;
use zstd::bulk::Decompressor;

thread_local! {
    static DECOMPRESSOR: RefCell<Option<Decompressor<'static>>> = const { RefCell::new(None) };
}

/// Runs `f` with a decompressor that has no dictionary.
///
/// Creating a zstd decompression context allocates and initializes its state, which dominates
/// decoding many small arrays. zstd starts every frame afresh on the same context, so one context
/// per thread serves every decode. A context that saw an error is dropped rather than reused, and
/// a nested call on the same thread gets a context of its own.
pub(crate) fn with_decompressor<R>(
    f: impl FnOnce(&mut Decompressor<'static>) -> VortexResult<R>,
) -> VortexResult<R> {
    DECOMPRESSOR.with(|cell| match cell.try_borrow_mut() {
        Ok(mut slot) => {
            let decompressor = match slot.as_mut() {
                Some(decompressor) => decompressor,
                None => slot.insert(Decompressor::new()?),
            };
            let result = f(decompressor);
            if result.is_err() {
                *slot = None;
            }
            result
        }
        Err(_) => f(&mut Decompressor::new()?),
    })
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::with_decompressor;

    #[test]
    fn decodes_after_a_corrupt_frame() -> VortexResult<()> {
        let value = b"hello hello hello hello";
        let frame = zstd::bulk::compress(value, 3)?;
        let mut out = vec![0u8; 64];
        let decode = |input: &[u8], out: &mut [u8]| {
            with_decompressor(|d| Ok(d.decompress_to_buffer(input, out)?))
        };

        let n = decode(&frame, &mut out)?;
        assert_eq!(&out[..n], value);
        assert!(decode(b"not a zstd frame", &mut out).is_err());
        let n = decode(&frame, &mut out)?;
        assert_eq!(&out[..n], value);
        Ok(())
    }
}
