// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::backtrace::Backtrace;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};

use vortex_array::ArrayRef;
use vortex_array::search_sorted::{SearchResult, SearchSortedSide};
use vortex_error::VortexError;
use vortex_scalar::Scalar;

fn tree_display(arr: &ArrayRef) -> impl Display {
    arr.display_tree()
}

#[non_exhaustive]
pub enum VortexFuzzError {
    SearchSortedError(
        Scalar,
        SearchResult,
        ArrayRef,
        SearchSortedSide,
        SearchResult,
        usize,
        Backtrace,
    ),

    ArrayNotEqual(Scalar, Scalar, usize, ArrayRef, ArrayRef, usize, Backtrace),

    LengthMismatch(usize, usize, ArrayRef, ArrayRef, usize, Backtrace),

    VortexError(VortexError, Backtrace),
}

impl Debug for VortexFuzzError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for VortexFuzzError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VortexFuzzError::SearchSortedError(
                a,
                expected,
                array,
                from,
                actual,
                step,
                backtrace,
            ) => {
                write!(
                    f,
                    "Expected to find {a} at {expected} in {} from {from} but instead found it at {actual} in step {step}\nBacktrace:\n{backtrace}",
                    tree_display(array),
                )
            }
            VortexFuzzError::ArrayNotEqual(expected, actual, idx, lhs, rhs, step, backtrace) => {
                write!(
                    f,
                    "{expected} != {actual} at index {idx}, lhs is {} rhs is {} in step {step}\nBacktrace:\n{backtrace}",
                    tree_display(lhs),
                    tree_display(rhs),
                )
            }
            VortexFuzzError::LengthMismatch(lhs_len, rhs_len, lhs, rhs, step, backtrace) => {
                write!(
                    f,
                    "LHS len {lhs_len} != RHS len {rhs_len}, lhs is {} rhs is {} in step {step}\nBacktrace:\n{backtrace}",
                    tree_display(lhs),
                    tree_display(rhs),
                )
            }
            VortexFuzzError::VortexError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
        }
    }
}

impl Error for VortexFuzzError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            VortexFuzzError::SearchSortedError(..) => None,
            VortexFuzzError::ArrayNotEqual(..) => None,
            VortexFuzzError::LengthMismatch(..) => None,
            VortexFuzzError::VortexError(err, ..) => Some(err),
        }
    }
}

pub type VortexFuzzResult<T> = Result<T, VortexFuzzError>;
