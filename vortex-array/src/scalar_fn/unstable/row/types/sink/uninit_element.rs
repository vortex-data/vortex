// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Uninitialized single-element output for row kernels.
//!
//! [`UninitElementSink`] exposes one uninitialized slot per output row. [`InitializedElement`]
//! proves that a successful callback wrote its slot before the sink publishes the output.

use std::mem::MaybeUninit;

use vortex_error::VortexResult;

use super::OutputSink;
use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::OutputElement;

/// Proof that one uninitialized element row was initialized.
///
/// The private field prevents safe construction without calling [`write`](Self::write):
///
/// ```compile_fail,E0423
/// use vortex_array::scalar_fn::unstable::row::InitializedElement;
///
/// let _evidence = InitializedElement(());
/// ```
#[must_use = "return this token from the row closure to prove that it initialized the output"]
pub struct InitializedElement(
    /// Private so constructing initialization evidence requires an unsafe operation.
    (),
);

impl InitializedElement {
    /// Write `value` into an uninitialized row and return its proof token.
    ///
    /// # Safety
    ///
    /// `row` must be the [`UninitElementSink`] row supplied to the current callback. The caller
    /// must return the token from that callback. Using another row or returning the token from
    /// another callback can cause undefined behavior.
    #[inline]
    pub unsafe fn write<T>(row: &mut MaybeUninit<T>, value: T) -> Self {
        row.write(value);

        Self(())
    }
}

/// An element sink that leaves dense output uninitialized before the row loop.
///
/// The row closure must return the [`InitializedElement`] from [`InitializedElement::write`] on
/// success. The token is zero-sized, so the proof adds no runtime row state.
///
/// When execution omits invalid rows, it initializes placeholders first. Errors and unwinds are
/// safe because `values` keeps length zero until `finish`. The `T: Copy` bound means that
/// initialized spare-capacity elements require no destruction.
pub struct UninitElementSink<T> {
    /// Spare storage written in increasing row order.
    values: Vec<T>,

    /// The number of slots exposed to the row loop and initialized before finishing.
    row_count: usize,
}

// SAFETY: the row slice covers exactly the reserved spare-capacity range, so each accepted index
// names one distinct slot. Safe code cannot construct `InitializedElement`. Its unsafe constructor
// writes the supplied slot and requires the caller to return that exact evidence. The
// skipped-row initializer writes `T::default()` into every slot before masked traversal.
unsafe impl<T: OutputElement + Copy + Default> OutputSink for UninitElementSink<T> {
    type Params = ();
    type Rows<'a> = &'a mut [MaybeUninit<T>];
    type Row<'a> = &'a mut MaybeUninit<T>;
    type WriteToken = InitializedElement;

    fn skipped_rows_initializer() -> Option<for<'a> fn(&mut Self::Rows<'a>)> {
        Some(|rows| {
            for row in rows.iter_mut() {
                row.write(T::default());
            }
        })
    }

    fn storage_dtype(_params: &Self::Params) -> DType {
        T::element_dtype()
    }

    fn with_capacity(rows: usize, _params: &Self::Params) -> VortexResult<Self> {
        Ok(Self {
            values: Vec::with_capacity(rows),
            row_count: rows,
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        &mut self.values.spare_capacity_mut()[..self.row_count]
    }

    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        // SAFETY: required by this method's contract.
        unsafe { rows.get_unchecked_mut(index) }
    }

    unsafe fn finish(mut self) -> VortexResult<ArrayRef> {
        // SAFETY: the caller guarantees every slot in `0..row_count` was initialized, and
        // `with_capacity` reserved every slot in that range.
        unsafe { self.values.set_len(self.row_count) };

        Ok(T::build(self.values))
    }
}
