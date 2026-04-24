// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ZigZag encoding for signed integer arrays.
//!
//! ZigZag is a lossless transformation that maps signed integers to unsigned
//! integers of the same bit width, with small-magnitude values (positive or
//! negative) mapped to small unsigned values. For 8-bit signed integers the
//! mapping is:
//!
//! | signed i8 | bits (i8)  | bits (u8)  | zigzag u8 |
//! |----------:|:----------:|:----------:|----------:|
//! |         0 | `00000000` | `00000000` |         0 |
//! |        -1 | `11111111` | `00000001` |         1 |
//! |         1 | `00000001` | `00000010` |         2 |
//! |        -2 | `11111110` | `00000011` |         3 |
//! |         2 | `00000010` | `00000100` |         4 |
//! |        -3 | `11111101` | `00000101` |         5 |
//! |         3 | `00000011` | `00000110` |         6 |
//!
//! This is not a compression step on its own: the output has the same bit width as
//! the input. Its value is as a *preprocessor*. After ZigZag, small-magnitude values
//! have short bit representations, which makes downstream encodings such as
//! bit-packing, delta, or variable-length integers far more effective than when
//! applied directly to signed data.
//!
//! Supported input types are the signed primitive integers `I8`, `I16`, `I32`, and
//! `I64`. Other input types cause [`zigzag_encode`] to return an error.
//!
//! # Example
//!
//! ```
//! # use vortex_array::IntoArray;
//! # use vortex_array::LEGACY_SESSION;
//! # use vortex_array::VortexSessionExecute;
//! # use vortex_array::arrays::PrimitiveArray;
//! # use vortex_error::VortexResult;
//! # use vortex_zigzag::{ZigZag, zigzag_encode};
//! # fn main() -> VortexResult<()> {
//! let original = PrimitiveArray::from_iter(-10_i32..10);
//! let encoded = zigzag_encode(original.as_view())?.into_array();
//! assert!(encoded.is::<ZigZag>());
//!
//! // Executing the encoded array decodes back to the original signed values.
//! let mut ctx = LEGACY_SESSION.create_execution_ctx();
//! let decoded = encoded.execute::<PrimitiveArray>(&mut ctx)?;
//! assert_eq!(decoded.len(), 20);
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! Protocol Buffers uses the same ZigZag transformation for its `sint32` and
//! `sint64` scalar types. See the [protobuf encoding guide][pb] for the
//! mathematical definition.
//!
//! [pb]: https://protobuf.dev/programming-guides/encoding/#signed-ints

pub use array::*;
pub use compress::*;

mod array;
mod compress;
mod compute;
mod kernel;
mod rules;
mod slice;
