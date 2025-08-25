// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST View encoding, an analog to raw FSST encoding.

mod array;
mod serde;
mod ops;
mod validity;
mod visitor;
mod compute;

use std::fmt;
use std::fmt::Formatter;

pub use array::*;

const MAX_INLINE_STR: usize = 12;

#[repr(C, align(8))]
#[derive(Copy, Clone)]
union View {
    inline: InlinedStr,
    outline: OutlinedStr,
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
    fn is_inlined(&self) -> bool {
        let inner = unsafe { self.inline };
        inner.len as usize <= MAX_INLINE_STR
    }
}

#[repr(C, align(8))]
#[derive(Debug, Copy, Clone)]
struct InlinedStr {
    /// Uncompressed string length
    len: u32,
    /// Raw string bytes
    bytes: [u8; 12],
}

#[repr(C, align(8))]
#[derive(Debug, Copy, Clone)]
struct OutlinedStr {
    /// Uncompressed string length
    len: u32,
    /// 8 bytes of prefix, more than StringView!
    prefix: [u8; 8],
    /// Index into the buffer
    index: u32,
}
