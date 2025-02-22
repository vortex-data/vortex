pub use array::*;

mod array;
pub mod compress;
mod compute;
mod iter;
mod serde;
mod statistics;

#[doc(hidden)]
pub mod _benchmarking {
    pub use compute::filter::filter_run_end;
    pub use compute::take::take_indices_unchecked;

    use super::*;
}
