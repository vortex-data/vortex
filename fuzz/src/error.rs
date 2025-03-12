use std::fmt;
use std::fmt::{Debug, Display, Formatter};

use vortex_array::ArrayRef;
use vortex_array::compute::{SearchResult, SearchSortedSide};
use vortex_error::VortexError;
use vortex_scalar::Scalar;

fn tree_display(arr: &ArrayRef) -> impl Display {
    arr.tree_display()
}

#[derive(thiserror::Error)]
pub enum VortexFuzzError {
    #[error("Expected to find {0} at {1} in {tree} from {3} but instead found it at {4} in step {5}", tree = tree_display(.2))]
    SearchSortedError(
        Scalar,
        SearchResult,
        ArrayRef,
        SearchSortedSide,
        SearchResult,
        usize,
    ),

    #[error("{0} != {1} at index {2}, lhs is {lhs_tree} rhs is {rhs_tree} in step {5}",lhs_tree = tree_display(.3), rhs_tree = tree_display(.4))]
    ArrayNotEqual(Scalar, Scalar, usize, ArrayRef, ArrayRef, usize),

    #[error("LHS len {0} != RHS len {1}, lhs is {lhs_tree} rhs is {rhs_tree} in step {4}", lhs_tree = tree_display(.2), rhs_tree = tree_display(.3))]
    LengthMismatch(usize, usize, ArrayRef, ArrayRef, usize),

    #[error(transparent)]
    VortexError(
        #[from]
        #[backtrace]
        VortexError,
    ),
}

impl Debug for VortexFuzzError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

pub type VortexFuzzResult<T> = Result<T, VortexFuzzError>;
