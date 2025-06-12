mod array;
mod compute;
mod serde;

/// Represents the equation A\[i\] = a * i b.
/// This can be used for compression, fast comparisons and also for row ids.
pub use array::{SequenceArray, SequenceEncoding, SequenceVTable};

// TODO(joe): hook up to the compressor
// TODO(joe): support comparisons with other operators
// TODO(joe): support list in expr pushdown
