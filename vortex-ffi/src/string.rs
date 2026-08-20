// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::ptr;
use std::slice;

use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

/// A non owning view over a byte range.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_view {
    /// NULL "ptr" requires len == 0
    pub ptr: *const c_char,
    /// Length in bytes.
    pub len: usize,
}

impl vx_view {
    /// {NULL, 0} for absent values
    pub(crate) const fn null() -> vx_view {
        vx_view {
            ptr: ptr::null(),
            len: 0,
        }
    }

    /// Borrow a Rust string
    pub(crate) fn from_str(value: &str) -> vx_view {
        vx_view {
            ptr: value.as_ptr().cast(),
            len: value.len(),
        }
    }

    /// Borrow a Rust byte slice
    pub(crate) fn from_bytes(value: &[u8]) -> vx_view {
        vx_view {
            ptr: value.as_ptr().cast(),
            len: value.len(),
        }
    }

    /// View vx_view as bytes
    ///
    /// # Safety
    ///
    /// "ptr" must be valid for "len" reads or NULL with "len == 0".
    pub(crate) unsafe fn as_bytes<'a>(&self) -> VortexResult<&'a [u8]> {
        if self.ptr.is_null() {
            vortex_ensure!(self.len == 0, "null vx_view pointer with non-zero length");
            return Ok(&[]);
        }
        Ok(unsafe { slice::from_raw_parts(self.ptr.cast(), self.len) })
    }

    /// View vx_view as UTF-8.
    ///
    /// # Safety
    ///
    /// `self.ptr` must be valid for `self.len` reads, or null when `self.len` is zero.
    pub unsafe fn as_str<'a>(&self) -> VortexResult<&'a str> {
        str::from_utf8(unsafe { self.as_bytes() }?).map_err(|e| vortex_err!("invalid utf-8: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vx_view() -> VortexResult<()> {
        let source = "Hello 世界 🌍";
        let view = vx_view::from_str(source);
        assert_eq!(unsafe { view.as_str() }?, source);
        assert_eq!(unsafe { view.as_bytes() }?, source.as_bytes());

        assert_eq!(unsafe { vx_view::null().as_str() }?, "");
        let bad = vx_view {
            ptr: ptr::null(),
            len: 3,
        };
        assert!(unsafe { bad.as_str() }.is_err());

        assert!(unsafe { vx_view::from_bytes(&[0xFFu8, 0xFE]).as_str() }.is_err());
        Ok(())
    }
}
