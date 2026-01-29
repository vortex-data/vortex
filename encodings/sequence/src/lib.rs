// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
mod array;
mod compress;
mod compute;
mod kernel;
mod rules;

#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitrarySequenceArray;
/// Represents the equation A\[i\] = a * i + b.
/// This can be used for compression, fast comparisons and also for row ids.
pub use array::SequenceArray;
pub use array::SequenceArrayParts;
/// Represents the equation A\[i\] = a * i + b.
/// This can be used for compression, fast comparisons and also for row ids.
pub use array::SequenceVTable;
pub use compress::sequence_encode;

// TODO(joe): hook up to the compressor
// TODO(joe): support comparisons with other operators
// TODO(joe): support list in expr pushdown
