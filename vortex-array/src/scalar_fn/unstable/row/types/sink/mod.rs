// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Output builders for row kernels that cannot return independent owned values.
//!
//! [`OutputSink`] owns the shared lifecycle and safety contract. [`UninitElementSink`] provides
//! uninitialized scalar storage, while [`FixedSizeListSink`] provides runtime-width row storage.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::ViewLen;

mod fixed_size_list;
pub use fixed_size_list::FixedSizeListSink;
pub use fixed_size_list::InitializedRow;

mod uninit_element;
pub use uninit_element::InitializedElement;
pub use uninit_element::UninitElementSink;

/// A column allocated once per batch that a row closure writes into, one row at a time.
///
/// A sink owns batch-wide state that an independent owned value cannot express, such as
/// uninitialized storage or a row handle covering more than one element. The executor passes each
/// row slot into an [`Fn`] closure.
///
/// A sink describes only how rows are physically written. An output dtype derived from the
/// function options or argument dtypes is declared by [`RowVisitor::with_output_dtype`], which
/// labels the column this sink builds.
///
/// Rows arrive in increasing index order. Ordinary execution visits `0..row_count` exactly once.
/// Execution can omit invalid rows when [`skipped_rows_initializer`] returns an initializer.
///
/// # Errors
///
/// Lifecycle methods report only incidental failures such as allocation. A semantic error that
/// depends on input values **must** come from the row callback through a fallible [`SinkResult`],
/// or [`RowFn::INFALLIBLE`] cannot protect optimizations such as dictionary push-down.
///
/// # Safety
///
/// An implementation must uphold all of these requirements:
///
/// - Every index below [`ViewLen::len`] for [`Rows`] **must** identify one distinct row owned by
///   this sink.
/// - A borrowed [`Rows`] view **must** retain its length and index-to-row mapping until it is
///   dropped. Calls to [`row_unchecked`](Self::row_unchecked) and safe uses of a returned
///   [`Row`](Self::Row) **must** preserve both properties.
/// - [`skipped_rows_initializer`] is the only exception to this stability requirement. The executor
///   checks the length again after the initializer. The initializer **must** initialize every row.
/// - A row must either be initialized before the callback or require a
///   [`WriteToken`] that safe code cannot produce without initializing that exact row. Evidence for
///   an uninitialized row **must not** be safely forgeable, reusable, or substitutable.
/// - `Self` and every borrowed [`Rows`] view **must** remain safe to drop if decoding,
///   preparation, skipped-row initialization, or a row callback returns an error or unwinds. The
///   executor can abandon a sink after any prefix of rows.
/// - [`finish`] **must** be sound once every visited callback returned its required token and the
///   skipped-row initializer, when present, ran successfully.
/// - Violating these requirements can cause undefined behavior.
///
/// [`Rows`]: Self::Rows
/// [`WriteToken`]: Self::WriteToken
/// [`finish`]: Self::finish
/// [`RowFn::INFALLIBLE`]: crate::scalar_fn::unstable::row::RowFn::INFALLIBLE
/// [`RowVisitor::with_output_dtype`]: crate::scalar_fn::unstable::row::RowVisitor::with_output_dtype
/// [`SinkResult`]: crate::scalar_fn::unstable::row::SinkResult
/// [`skipped_rows_initializer`]: Self::skipped_rows_initializer
pub unsafe trait OutputSink: 'static + Sized {
    /// Physical parameters required to construct this sink before the row loop.
    ///
    /// This type describes only physical storage. A logical output dtype belongs on
    /// [`RowVisitor::with_output_dtype`](crate::scalar_fn::unstable::row::RowVisitor::with_output_dtype).
    type Params: 'static;

    /// A loop-local view of all output rows.
    ///
    /// Borrowed once before execution so the sink's buffer descriptor and shape become loop
    /// invariants rather than being re-read through `&mut Self` for every row.
    type Rows<'a>: ViewLen
    where
        Self: 'a;

    /// The place a row closure writes one row through, borrowed from the sink.
    type Row<'a>
    where
        Self: 'a;

    /// Proof that a successful row closure left its row handle initialized.
    ///
    /// Use `()` for initialized row handles. A sink exposing uninitialized storage uses a token
    /// returned after initialization. If a sink uses the token to justify unsafe code, safe code
    /// **must not** be able to construct one without establishing the invariant.
    type WriteToken: 'static;

    /// The operation that initializes every output position before
    /// [skip-invalid execution](crate::scalar_fn::unstable::row).
    ///
    /// `Some` enables this strategy. The initializer **must** make every row safe to finish.
    /// Callbacks overwrite valid rows, and batch execution masks skipped rows.
    ///
    /// `None` makes skip-invalid execution unavailable for this sink.
    fn skipped_rows_initializer() -> Option<for<'a> fn(&mut Self::Rows<'a>)> {
        None
    }

    /// The dtype of the column this sink builds.
    ///
    /// **Must** be non-nullable: batch execution derives nullability from the inputs, widens the
    /// result, and masks the null rows.
    fn storage_dtype(params: &Self::Params) -> DType;

    /// Allocate a sink for `rows` rows.
    fn with_capacity(rows: usize, params: &Self::Params) -> VortexResult<Self>;

    /// Borrow all output rows for the hot loop.
    fn rows(&mut self) -> Self::Rows<'_>;

    /// Hand out the place to write row `index`. Must be `O(1)`: it is called in the row loop.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`ViewLen::len`] for `rows`.
    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a>;

    /// Finish into the built column, whose dtype **must** be this sink's
    /// [`storage_dtype`](Self::storage_dtype) for the parameters passed to
    /// [`with_capacity`](Self::with_capacity). Called once per batch.
    ///
    /// # Safety
    ///
    /// The executor must have completed every row callback successfully, and each callback must
    /// have returned this sink's [`WriteToken`](Self::WriteToken). When skipped rows are allowed,
    /// the initializer returned by
    /// [`skipped_rows_initializer`](Self::skipped_rows_initializer) must have run before traversal.
    unsafe fn finish(self) -> VortexResult<ArrayRef>;
}
