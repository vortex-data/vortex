// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST View encoding, an analog to raw FSST encoding.

mod array;
mod canonical;
mod compute;
mod ops;
mod serde;
mod validity;
mod visitor;

use std::fmt;
use std::fmt::Formatter;

pub use array::*;

const MAX_INLINE_STR: usize = 12;

#[macro_export]
macro_rules! view_outlined {
    ($len:expr, $index:expr, $prefix:expr) => {
        #[allow(clippy::cast_possible_truncation)]
        {
            let mut outline = $crate::fsst_view::OutlinedStr {
                index: $index as u32,
                len: $len as u32,
                prefix: [0u8; 8],
            };
            outline.prefix.copy_from_slice(&$prefix[0..8]);

            $crate::fsst_view::View { outline }
        }
    };
}

/// "View" structure that serves inlined strings, or points to compressed copies of longer strings.
#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub union View {
    inline: InlinedStr,
    outline: OutlinedStr,
    // little-endian bytes representation
    le_bytes: [u8; 16],
}

impl View {
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_inlined(data: &[u8]) -> Self {
        assert!(data.len() <= MAX_INLINE_STR);

        // Safe to truncate cast, always small enough.
        let len = data.len() as u32;
        let mut inlined_str = InlinedStr {
            len,
            bytes: [0; MAX_INLINE_STR],
        };

        inlined_str.bytes[..data.len()].copy_from_slice(data);

        Self {
            inline: inlined_str,
        }
    }

    pub fn len(&self) -> u32 {
        // SAFETY: len is always first 4 bytes for either variant
        unsafe { self.inline }.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_u128(self) -> u128 {
        u128::from_le_bytes(unsafe { self.le_bytes })
    }
}

impl fmt::Debug for View {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut w = f.debug_struct("View");
        if self.is_inlined() {
            let inlined = unsafe { self.inline };
            w.field("inlined", &inlined);
        } else {
            let outlined = unsafe { self.outline };
            w.field("outlined", &outlined);
        }
        w.finish()
    }
}

impl View {
    /// Returns true if the view is an inlined view
    #[inline(always)]
    pub fn is_inlined(&self) -> bool {
        let inner = unsafe { self.inline };
        inner.len as usize <= MAX_INLINE_STR
    }
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct InlinedStr {
    /// Uncompressed string length
    len: u32,
    /// Raw string bytes
    bytes: [u8; 12],
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct OutlinedStr {
    /// Uncompressed string length
    len: u32,
    /// 8 bytes of prefix, more than StringView!
    prefix: [u8; 8],
    /// Index into the buffer
    index: u32,
}
