// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::DeltaData;
pub use array::delta_compress::delta_compress;
// Exposed for benchmarks: decode entry points so a bench can A/B the fused fast path against the
// generic (pre-fusion) decode on the same array.
#[cfg(feature = "_test-harness")]
pub use array::delta_decompress::{delta_decompress, delta_decompress_generic};

mod compute;

mod vtable;
pub use vtable::Delta;
pub use vtable::DeltaArray;
