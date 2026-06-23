// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Vortex encoding that mirrors Arrow's [8-bit Boolean canonical extension type][spec].
//!
//! Each element is stored as a single byte. The zero byte represents `false` and any
//! non-zero byte represents `true`, matching the truthy semantics of the Arrow spec. This
//! trades 8x the storage of the bit-packed `Bool` layout for cheaper per-byte access —
//! useful when data arrives from a C ABI or other source that already emits byte-wide
//! booleans. On execution the array materializes into the standard bit-packed
//! [`BoolArray`][vortex_array::arrays::BoolArray].
//!
//! # Examples
//!
//! Any non-zero byte in the backing buffer is treated as `true` when the array executes
//! to a canonical [`BoolArray`][vortex_array::arrays::BoolArray]:
//!
//! ```
//! # use vortex_array::{IntoArray, VortexSessionExecute, array_session};
//! # use vortex_array::arrays::BoolArray;
//! # use vortex_array::arrays::bool::BoolArrayExt;
//! # use vortex_array::buffer::BufferHandle;
//! # use vortex_array::validity::Validity;
//! # use vortex_buffer::ByteBuffer;
//! # use vortex_bytebool::ByteBool;
//! # use vortex_error::VortexResult;
//! # fn main() -> VortexResult<()> {
//! # let mut ctx = array_session().create_execution_ctx();
//! let handle = BufferHandle::new_host(ByteBuffer::from(vec![0u8, 1, 42, 0]));
//! let array = ByteBool::new(handle, Validity::NonNullable);
//!
//! let bits = array.into_array().execute::<BoolArray>(&mut ctx)?.to_bit_buffer();
//! assert!(!bits.value(0));
//! assert!(bits.value(1));
//! assert!(bits.value(2)); // byte 42 is truthy
//! assert!(!bits.value(3));
//! # Ok(())
//! # }
//! ```
//!
//! [spec]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#bit-boolean

pub use array::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod array;
mod compute;
mod kernel;
mod rules;
mod slice;

/// Initialize bytebool encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(ByteBool);
    kernel::initialize(session);
}
