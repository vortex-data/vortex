pub use array::*;
pub use iter::trimmed_ends_iter;

mod array;
pub mod compress;
mod compute;
mod iter;
mod serde;

#[doc(hidden)]
pub mod _benchmarking {
    pub use compute::filter::filter_run_end;
    pub use compute::take::take_indices_unchecked;

    use super::*;
}
