// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::backtrace::Backtrace;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};

use vortex_array::compute::MinMaxResult;
use vortex_array::search_sorted::{SearchResult, SearchSortedSide};
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexError;
use vortex_scalar::Scalar;

#[non_exhaustive]
pub enum VortexFuzzError {
    ScalarMismatch(Scalar, Scalar, usize, Backtrace),

    SearchSortedError(
        Scalar,
        SearchResult,
        ArrayRef,
        SearchSortedSide,
        SearchResult,
        usize,
        Backtrace,
    ),

    MinMaxMismatch(Option<MinMaxResult>, Option<MinMaxResult>, usize, Backtrace),

    ArrayNotEqual(Scalar, Scalar, usize, ArrayRef, ArrayRef, usize, Backtrace),

    DTypeMismatch(ArrayRef, ArrayRef, usize, Backtrace),

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
            VortexFuzzError::ScalarMismatch(lhs, rhs, step, backtrace) => {
                write!(
                    f,
                    "Scalar mismatch: expected {lhs}, got {rhs} in step {step}\nBacktrace:\n{backtrace}"
                )
            }
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
                    array.display_tree(),
                )
            }
            VortexFuzzError::MinMaxMismatch(lhs, rhs, step, backtrace) => {
                write!(
                    f,
                    "MinMax mismatch: expected {lhs:?} got {rhs:?} in step {step}\nBacktrace:\n{backtrace}"
                )
            }
            VortexFuzzError::ArrayNotEqual(expected, actual, idx, lhs, rhs, step, backtrace) => {
                write!(
                    f,
                    "{expected} != {actual} at index {idx}, lhs is {} rhs is {} in step {step}\nBacktrace:\n{backtrace}",
                    lhs.display_tree(),
                    rhs.display_tree(),
                )
            }
            VortexFuzzError::DTypeMismatch(lhs, rhs, step, backtrace) => {
                write!(
                    f,
                    "DType mismatch: expected {}, got {} in step {step}\nBacktrace:\n{backtrace}",
                    lhs.dtype(),
                    rhs.dtype()
                )
            }
            VortexFuzzError::LengthMismatch(lhs_len, rhs_len, lhs, rhs, step, backtrace) => {
                write!(
                    f,
                    "LHS len {lhs_len} != RHS len {rhs_len}, lhs is {} rhs is {} in step {step}\nBacktrace:\n{backtrace}",
                    lhs.display_tree(),
                    rhs.display_tree(),
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
            VortexFuzzError::VortexError(err, ..) => Some(err),
            VortexFuzzError::SearchSortedError(..)
            | VortexFuzzError::ArrayNotEqual(..)
            | VortexFuzzError::LengthMismatch(..)
            | VortexFuzzError::ScalarMismatch(..)
            | VortexFuzzError::MinMaxMismatch(..)
            | VortexFuzzError::DTypeMismatch(..) => None,
        }
    }
}

pub type VortexFuzzResult<T> = Result<T, VortexFuzzError>;
