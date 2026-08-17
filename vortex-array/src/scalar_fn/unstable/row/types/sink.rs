// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Output builders for row kernels that cannot return independent owned values.
//!
//! [`OutputSink`] allocates batch-wide state and lends one row handle to each callback.
//! [`UninitElementSink`] is the fixed-width implementation used when avoiding output
//! initialization matters.

use std::mem::MaybeUninit;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::OutputElement;

/// A column allocated once per batch that a row closure writes into, one row at a time.
///
/// A sink can use function options and input dtypes to build a runtime-shaped output or own shared
/// batch state. The executor passes each row slot into an [`Fn`] closure.
///
/// Rows arrive in increasing index order. Ordinary execution visits `0..row_count` exactly once.
/// Execution can omit invalid rows when [`skipped_rows_initializer`] returns an initializer.
///
/// # Errors
///
/// Lifecycle methods report only incidental failures such as allocation. A semantic error that
/// depends on input values **must** come from the row callback through a fallible [`SinkResult`],
/// or [`RowFn::FALLIBLE`] cannot protect optimizations such as dictionary push-down.
///
/// # Safety
///
/// An implementation must uphold all of these requirements:
///
/// - Every index in `0..row_count(rows)` **must** identify one distinct row owned by this sink.
/// - A row must either be initialized before the callback or require a
///   [`WriteToken`] that safe code cannot produce without initializing that exact row. Evidence for
///   an uninitialized row **must not** be safely forgeable, reusable, or substitutable.
/// - An initializer returned by [`skipped_rows_initializer`] **must** initialize every row.
/// - `Self` and every borrowed [`Rows`] view **must** remain safe to drop if decoding,
///   preparation, skipped-row initialization, or a row callback returns an error or unwinds. The
///   executor can abandon a sink after any prefix of rows.
/// - [`finish`] **must** be sound once every visited callback returned its required token and the
///   skipped-row initializer, when present, ran successfully.
///
/// [`Rows`]: Self::Rows
/// [`WriteToken`]: Self::WriteToken
/// [`finish`]: Self::finish
/// [`row_count`]: Self::row_count
/// [`RowFn::FALLIBLE`]: crate::scalar_fn::unstable::row::RowFn::FALLIBLE
/// [`SinkResult`]: crate::scalar_fn::unstable::row::SinkResult
/// [`skipped_rows_initializer`]: Self::skipped_rows_initializer
pub unsafe trait OutputSink<Options>: 'static + Sized {
    /// A loop-local view of all output rows.
    ///
    /// Borrowed once before execution so the sink's buffer descriptor and shape become loop
    /// invariants rather than being re-read through `&mut Self` for every row.
    type Rows<'a>
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
    /// `None` makes the executor fall back to filtering the inputs.
    fn skipped_rows_initializer() -> Option<for<'a> fn(&mut Self::Rows<'a>)> {
        None
    }

    /// The dtype of the column this sink builds, given the function options and input dtypes.
    ///
    /// **Must** be non-nullable: batch execution derives nullability from the inputs, widens the
    /// result, and masks the null rows.
    fn output_dtype(options: &Options, args: &[DType]) -> VortexResult<DType>;

    /// Allocate a sink for `rows` rows.
    fn with_capacity(rows: usize) -> VortexResult<Self>;

    /// Borrow all output rows for the hot loop.
    fn rows(&mut self) -> Self::Rows<'_>;

    /// The number of rows addressable through [`row_unchecked`](Self::row_unchecked).
    fn row_count(rows: &Self::Rows<'_>) -> usize;

    /// Hand out the place to write row `index`. Must be `O(1)`: it is called in the row loop.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`row_count`](Self::row_count) for `rows`.
    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a>;

    /// Finish into the built column, whose dtype **must** be this sink's
    /// [`output_dtype`](Self::output_dtype). Called once per batch.
    ///
    /// # Safety
    ///
    /// The executor must have completed every row callback successfully, and each callback must
    /// have returned this sink's [`WriteToken`](Self::WriteToken). When skipped rows are allowed,
    /// the initializer returned by
    /// [`skipped_rows_initializer`](Self::skipped_rows_initializer) must have run before traversal.
    unsafe fn finish(self) -> VortexResult<ArrayRef>;
}

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
unsafe impl<T: OutputElement + Copy + Default, Options> OutputSink<Options>
    for UninitElementSink<T>
{
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

    fn output_dtype(_options: &Options, _args: &[DType]) -> VortexResult<DType> {
        Ok(T::element_dtype())
    }

    fn with_capacity(rows: usize) -> VortexResult<Self> {
        Ok(Self {
            values: Vec::with_capacity(rows),
            row_count: rows,
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        &mut self.values.spare_capacity_mut()[..self.row_count]
    }

    fn row_count(rows: &Self::Rows<'_>) -> usize {
        rows.len()
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
