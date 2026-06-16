// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex UTF-8 strings.
//!
//! In the hand-written C ABI this was an `arc_dyn_wrapper!(str, vx_string)` plus the free
//! functions `vx_string_new`, `vx_string_new_from_cstr`, `vx_string_len`, `vx_string_ptr`, and
//! the macro-generated `vx_string_clone` / `vx_string_free`.
//!
//! Under Diplomat we keep an opaque [`VxString`] so that owned UTF-8 strings can be returned
//! across the boundary (for example from `VxArray::get_utf8`). Diplomat auto-generates the
//! destructor, so there is no manual `_free`. Reading the contents back out is done by writing
//! into a `DiplomatWrite` rather than exposing a raw `(ptr, len)` pair, which lets each target
//! language materialize a native string.

pub use ffi::VxString;

#[diplomat::bridge]
pub mod ffi {
    use std::fmt::Write;
    use std::sync::Arc;

    use diplomat_runtime::{DiplomatStr, DiplomatWrite};

    /// An owned, reference-counted UTF-8 string for use across the Vortex FFI boundary.
    ///
    /// Replaces the C `vx_string` opaque type. Internally an `Arc<str>`, so cloning is cheap.
    #[diplomat::opaque]
    pub struct VxString(pub(crate) Arc<str>);

    impl VxString {
        /// Create a new Vortex string by copying from a UTF-8 byte buffer.
        ///
        /// Replaces `vx_string_new`. The bytes must be valid UTF-8; on invalid UTF-8 the lossy
        /// replacement is used (the C ABI panicked, which Diplomat avoids).
        #[diplomat::attr(auto, constructor)]
        pub fn new(value: &DiplomatStr) -> Box<VxString> {
            let s = String::from_utf8_lossy(value);
            Box::new(VxString(Arc::from(s.as_ref())))
        }

        /// Create a new Vortex string from a borrowed `&str`.
        ///
        /// The Diplomat analogue of `vx_string_new_from_cstr`: Diplomat handles the
        /// null-terminated/length-prefixed marshalling, so a plain `&str` suffices.
        #[diplomat::attr(auto, named_constructor = "from_str")]
        pub fn from_str(value: &str) -> Box<VxString> {
            Box::new(VxString(Arc::from(value)))
        }

        /// The length of the string in bytes.
        ///
        /// Replaces `vx_string_len`.
        #[diplomat::attr(auto, getter)]
        pub fn len(&self) -> usize {
            self.0.len()
        }

        /// Whether the string is empty.
        #[diplomat::attr(auto, getter)]
        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        /// Write the string contents into `out`.
        ///
        /// Replaces `vx_string_ptr`: rather than handing back a raw pointer into Vortex-owned
        /// memory, the bytes are written to a `DiplomatWrite` that each language materializes as a
        /// native string.
        #[diplomat::attr(auto, getter)]
        pub fn value(&self, out: &mut DiplomatWrite) {
            // Infallible: writing to a DiplomatWrite cannot fail.
            let _ = write!(out, "{}", self.0);
        }
    }
}
