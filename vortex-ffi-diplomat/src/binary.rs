// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex binary blobs.
//!
//! In the hand-written C ABI this was an `arc_dyn_wrapper!([u8], vx_binary)` plus the free
//! functions `vx_binary_new`, `vx_binary_len`, `vx_binary_ptr`, and the macro-generated
//! `vx_binary_clone` / `vx_binary_free`.
//!
//! Under Diplomat we keep an opaque [`VxBinary`] so that owned byte buffers can be returned
//! across the boundary (for example from `VxArray::get_binary`). Diplomat auto-generates the
//! destructor. The raw `(ptr, len)` accessors become a borrowing `as_bytes` method that returns
//! a `&[u8]` slice that each target language copies into a native byte container.

pub use ffi::VxBinary;

#[diplomat::bridge]
pub mod ffi {
    use std::sync::Arc;

    /// An owned, reference-counted binary blob for use across the Vortex FFI boundary.
    ///
    /// Replaces the C `vx_binary` opaque type. Internally an `Arc<[u8]>`, so cloning is cheap.
    #[diplomat::opaque]
    pub struct VxBinary(pub(crate) Arc<[u8]>);

    impl VxBinary {
        /// Create a new Vortex binary blob by copying from a byte buffer.
        ///
        /// Replaces `vx_binary_new`, which took a `(ptr, len)` pair. Diplomat marshals the slice
        /// directly.
        #[diplomat::attr(auto, constructor)]
        pub fn new(bytes: &[u8]) -> Box<VxBinary> {
            Box::new(VxBinary(Arc::from(bytes)))
        }

        /// The length of the blob in bytes.
        ///
        /// Replaces `vx_binary_len`.
        #[diplomat::attr(auto, getter)]
        pub fn len(&self) -> usize {
            self.0.len()
        }

        /// Whether the blob is empty.
        #[diplomat::attr(auto, getter)]
        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        /// Borrow the underlying bytes.
        ///
        /// Replaces `vx_binary_ptr`: instead of a raw pointer into Vortex-owned memory, Diplomat
        /// returns a borrowed slice tied to the lifetime of `self` that the target language copies
        /// out.
        pub fn as_bytes<'a>(&'a self) -> &'a [u8] {
            &self.0
        }
    }
}
