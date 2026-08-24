// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Runtime-width fixed-size-list output for row kernels.
//!
//! [`FixedSizeListSink`] stores each row in a contiguous slice of one flat element allocation.
//! [`InitializedRow`] proves that a row callback filled its entire slice before finishing.

use std::mem::MaybeUninit;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::OutputSink;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FixedSizeListArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::ViewLen;
use crate::validity::Validity;

/// Proof that every element in one uninitialized fixed-size row was initialized.
///
/// The private field prevents construction without calling [`fill`](Self::fill), which writes the
/// complete row before returning.
#[must_use = "return this token from the row closure to prove that it initialized the output"]
pub struct InitializedRow(());

impl InitializedRow {
    /// Fill every element in `row` and return its proof token.
    #[inline]
    pub fn fill<T>(
        row: &mut [MaybeUninit<T>],
        mut value_for_index: impl FnMut(usize) -> T,
    ) -> Self {
        for (index, element) in row.iter_mut().enumerate() {
            element.write(value_for_index(index));
        }

        Self(())
    }
}

/// A loop-local view of the flat storage and runtime shape of a [`FixedSizeListSink`].
pub struct FixedSizeRows<'a, T> {
    /// The flat elements for all output rows.
    elements: &'a mut [MaybeUninit<T>],

    /// The number of elements in each output row.
    width: usize,

    /// The number of output rows, stored separately so zero-width rows remain addressable.
    row_count: usize,
}

impl<T> ViewLen for FixedSizeRows<'_, T> {
    fn len(&self) -> usize {
        self.row_count
    }
}

/// A fixed-size-list sink whose row width is supplied at dispatch time.
///
/// The row closure must return the [`InitializedRow`] from [`InitializedRow::fill`] on success.
/// The width must fit in the `u32` list size stored by [`FixedSizeListArray`]. A dispatch derives
/// and validates that physical parameter before calling [`RowVisitor::visit_into`].
///
/// [`RowVisitor::visit_into`]: crate::scalar_fn::unstable::row::RowVisitor::visit_into
pub struct FixedSizeListSink<T> {
    /// Spare flat storage written one fixed-size row at a time.
    values: Vec<T>,

    /// The number of elements in each output row.
    width: usize,

    /// The number of output rows.
    row_count: usize,
}

// SAFETY: `with_capacity` reserves `row_count * width` elements, and `FixedSizeRows` retains that
// shape for its lifetime. Each row is one disjoint `width`-element slice. `InitializedRow::fill`
// writes every element before returning its private token, and the skipped-row initializer writes
// every flat element before masked traversal. `values` retains length zero until every row is safe
// to publish in `finish`.
unsafe impl<T: OutputElement + Copy + Default> OutputSink for FixedSizeListSink<T> {
    type Params = usize;
    type Rows<'a> = FixedSizeRows<'a, T>;
    type Row<'a> = &'a mut [MaybeUninit<T>];
    type WriteToken = InitializedRow;

    fn skipped_rows_initializer() -> Option<for<'a> fn(&mut Self::Rows<'a>)> {
        Some(|rows| {
            for element in rows.elements.iter_mut() {
                element.write(T::default());
            }
        })
    }

    fn storage_dtype(params: &Self::Params) -> DType {
        DType::FixedSizeList(
            Arc::new(T::element_dtype()),
            fixed_size_list_size(*params),
            Nullability::NonNullable,
        )
    }

    fn with_capacity(rows: usize, params: &Self::Params) -> VortexResult<Self> {
        let width = *params;
        let element_capacity = rows.checked_mul(width).ok_or_else(|| {
            vortex_err!(
                InvalidArgument:
                "fixed-size-list sink capacity must fit in usize, got {rows} rows with width {width}"
            )
        })?;

        Ok(Self {
            values: Vec::with_capacity(element_capacity),
            width,
            row_count: rows,
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        FixedSizeRows {
            elements: &mut self.values.spare_capacity_mut()[..self.row_count * self.width],
            width: self.width,
            row_count: self.row_count,
        }
    }

    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        let start = index * rows.width;
        let end = start + rows.width;

        // SAFETY: the caller guarantees `index < row_count`, and construction guarantees the
        // element slice has length `row_count * width`.
        unsafe { rows.elements.get_unchecked_mut(start..end) }
    }

    unsafe fn finish(mut self) -> VortexResult<ArrayRef> {
        let element_count = self.row_count * self.width;

        // SAFETY: the caller guarantees every row was initialized, and `with_capacity` reserved
        // `row_count * width` elements.
        unsafe { self.values.set_len(element_count) };

        let elements = T::build(self.values);
        let lists = FixedSizeListArray::new(
            elements,
            fixed_size_list_size(self.width),
            Validity::NonNullable,
            self.row_count,
        );

        Ok(lists.into_array())
    }
}

fn fixed_size_list_size(width: usize) -> u32 {
    u32::try_from(width)
        .vortex_expect("fixed-size-list sink width must fit in u32; dispatch validated it")
}
