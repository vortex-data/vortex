// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validated indexed access to tuples of decoded input columns.
//!
//! [`IndexedElementTuple`] adapts row arguments to the lane-kernel interface after batch execution
//! proves that every input covers the requested row range.

use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::LaneZip;

use super::ElementTuple;
use super::element_tuple::ArgColumn;
use super::element_tuple::ArgColumnKind;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::ViewLen;

/// An argument tuple that supports a validated dense indexed traversal.
///
/// Every [`ElementTuple`] implements this trait. Its source delegates each lane read to the tuple's
/// unchecked view access after batch execution validates every decoded column length once.
pub trait IndexedElementTuple: ElementTuple + private::DecodedSource {
    /// The source used when no input is batch-constant.
    ///
    /// Its length must be the common view length. For every valid index it must preserve row order,
    /// return the same value as [`ElementTuple::get_from_views`], and uphold the unchecked read
    /// contract of [`IndexedSource`].
    type Source<'a>: IndexedSource<Item = Self::Elems<'a>>;

    /// Build a source from views already validated to cover the complete batch.
    ///
    /// # Safety
    ///
    /// Every view in `views` **must** address exactly `row_count` rows. Violating this requirement
    /// can make a safe lane kernel read outside a column's allocation.
    unsafe fn indexed_source<'a>(views: Self::Views<'a>, row_count: usize) -> Self::Source<'a>;
}

pub(in crate::scalar_fn::unstable::row) fn decoded_source<'a, Args: IndexedElementTuple>(
    columns: &'a Args::Columns,
    row_count: usize,
) -> Option<impl IndexedSource<Item = Args::Elems<'a>>> {
    <Args as private::DecodedSource>::decoded_source(columns, row_count)
}

/// Indexed access to one decoded [`ArgColumn`].
///
/// [`ArgColumn`] already records whether an input is row-wise or batch-constant. Keeping that choice
/// in each argument source lets LLVM unswitch it before vectorizing Boolean collection into packed
/// words. Routing every input through [`ElementTuple::get`] obscures the independent choices inside
/// the row loop.
enum ArgColumnSource<'a, T: InputElement> {
    Rows(T::View<'a>),

    /// A validated one-row view that logically addresses `row_count` rows.
    Constant {
        view: T::View<'a>,
        row_count: usize,
    },
}

impl<'a, T: InputElement> ArgColumnSource<'a, T> {
    fn try_new(column: &'a ArgColumn<T>, row_count: usize) -> Option<Self> {
        match &column.0 {
            ArgColumnKind::Column(column) => {
                let view = T::view(column);
                (view.len() == row_count).then_some(Self::Rows(view))
            }
            ArgColumnKind::Const(column) => {
                let view = T::view(column);
                (view.len() == 1).then_some(Self::Constant { view, row_count })
            }
        }
    }
}

impl<'a, T: InputElement> IndexedSource for ArgColumnSource<'a, T> {
    type Item = T::Elem<'a>;

    fn len(&self) -> usize {
        match self {
            Self::Rows(view) => view.len(),
            Self::Constant { row_count, .. } => *row_count,
        }
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        match self {
            Self::Rows(view) => {
                // SAFETY: `try_new` checked that this retained view has `row_count` rows, and the
                // caller guarantees that `index` is below the source length.
                unsafe { T::get_from_view_unchecked(view, index) }
            }
            Self::Constant { view, .. } => {
                // SAFETY: `try_new` checked that this exact retained view contains row zero.
                unsafe { T::get_from_view_unchecked(view, 0) }
            }
        }
    }
}

/// Indexed access to a tuple of decoded argument sources.
struct ArgTupleSource<Sources> {
    sources: Sources,
    row_count: usize,
}

impl IndexedSource for ArgTupleSource<()> {
    type Item = ();

    fn len(&self) -> usize {
        self.row_count
    }

    unsafe fn get_unchecked(&self, _index: usize) -> Self::Item {}
}

/// Indexed access to one element view.
pub struct ElementSource<'a, T: InputElement> {
    view: T::View<'a>,
}

impl<'a, T: InputElement> ElementSource<'a, T> {
    fn new(view: T::View<'a>) -> Self {
        Self { view }
    }
}

impl<'a, T: InputElement> IndexedSource for ElementSource<'a, T> {
    type Item = T::Elem<'a>;

    fn len(&self) -> usize {
        self.view.len()
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        // SAFETY: the source length is the number of rows addressable by `view`, and the caller
        // guarantees that `index` is below that length.
        unsafe { T::get_from_view_unchecked(&self.view, index) }
    }
}

/// An indexed element source yielding the one-tuples expected by a unary row closure.
pub struct UnaryTupleSource<Source>(Source);

impl<Source: IndexedSource> IndexedSource for UnaryTupleSource<Source> {
    type Item = (Source::Item,);

    fn len(&self) -> usize {
        self.0.len()
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        // SAFETY: forwarded from this method's contract.
        (unsafe { self.0.get_unchecked(index) },)
    }
}

/// Indexed access to the views of an element tuple.
pub struct ElementTupleSource<'a, Args: ElementTuple> {
    views: Args::Views<'a>,
    row_count: usize,
}

impl<'a, Args: ElementTuple> IndexedSource for ElementTupleSource<'a, Args> {
    type Item = Args::Elems<'a>;

    fn len(&self) -> usize {
        self.row_count
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        // SAFETY: the caller guarantees that `index` is below `row_count`. Batch execution checks
        // that every view addresses exactly `row_count` rows before constructing this source.
        unsafe { Args::get_from_views_unchecked(&self.views, index) }
    }
}

impl IndexedElementTuple for () {
    type Source<'a> = ElementTupleSource<'a, ()>;

    unsafe fn indexed_source<'a>(views: Self::Views<'a>, row_count: usize) -> Self::Source<'a> {
        ElementTupleSource { views, row_count }
    }
}

impl private::DecodedSource for () {
    fn decoded_source<'a>(
        _columns: &'a Self::Columns,
        row_count: usize,
    ) -> Option<impl IndexedSource<Item = Self::Elems<'a>>> {
        Some(ArgTupleSource {
            sources: (),
            row_count,
        })
    }
}

impl<A: InputElement> IndexedElementTuple for (A,) {
    type Source<'a> = UnaryTupleSource<ElementSource<'a, A>>;

    unsafe fn indexed_source<'a>(views: Self::Views<'a>, _row_count: usize) -> Self::Source<'a> {
        UnaryTupleSource(ElementSource::new(views.0))
    }
}

impl<A: InputElement> private::DecodedSource for (A,) {
    fn decoded_source<'a>(
        columns: &'a Self::Columns,
        row_count: usize,
    ) -> Option<impl IndexedSource<Item = Self::Elems<'a>>> {
        Some(ArgTupleSource {
            sources: (ArgColumnSource::try_new(&columns.0, row_count)?,),
            row_count,
        })
    }
}

impl<A: InputElement, B: InputElement> IndexedElementTuple for (A, B) {
    type Source<'a> = LaneZip<ElementSource<'a, A>, ElementSource<'a, B>>;

    unsafe fn indexed_source<'a>(views: Self::Views<'a>, _row_count: usize) -> Self::Source<'a> {
        LaneZip::new(ElementSource::new(views.0), ElementSource::new(views.1))
    }
}

impl<A: InputElement, B: InputElement> private::DecodedSource for (A, B) {
    fn decoded_source<'a>(
        columns: &'a Self::Columns,
        row_count: usize,
    ) -> Option<impl IndexedSource<Item = Self::Elems<'a>>> {
        Some(ArgTupleSource {
            sources: (
                ArgColumnSource::try_new(&columns.0, row_count)?,
                ArgColumnSource::try_new(&columns.1, row_count)?,
            ),
            row_count,
        })
    }
}

macro_rules! arg_tuple_source {
    ($($source:ident: $idx:tt),+) => {
        impl<$($source: IndexedSource),+> IndexedSource
            for ArgTupleSource<($($source,)+)>
        {
            type Item = ($($source::Item,)+);

            fn len(&self) -> usize {
                self.row_count
            }

            unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
                // SAFETY: forwarded from this method's contract. Every source has `row_count`
                // rows by construction.
                ($(unsafe { self.sources.$idx.get_unchecked(index) },)+)
            }
        }
    };
}

arg_tuple_source!(A: 0);
arg_tuple_source!(A: 0, B: 1);
arg_tuple_source!(A: 0, B: 1, C: 2);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3, E: 4);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10);
arg_tuple_source!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11);

macro_rules! indexed_element_tuple {
    ($($t:ident: $idx:tt),+) => {
        impl<$($t: InputElement),+> IndexedElementTuple for ($($t,)+) {
            type Source<'a> = ElementTupleSource<'a, ($($t,)+)>;

            unsafe fn indexed_source<'a>(
                views: Self::Views<'a>,
                row_count: usize,
            ) -> Self::Source<'a> {
                ElementTupleSource { views, row_count }
            }
        }

        impl<$($t: InputElement),+> private::DecodedSource for ($($t,)+) {
            fn decoded_source<'a>(
                columns: &'a Self::Columns,
                row_count: usize,
            ) -> Option<impl IndexedSource<Item = Self::Elems<'a>>> {
                Some(ArgTupleSource {
                    sources: ($(ArgColumnSource::try_new(
                        &columns.$idx,
                        row_count,
                    )?,)+),
                    row_count,
                })
            }
        }
    };
}

indexed_element_tuple!(A: 0, B: 1, C: 2);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10);
indexed_element_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11);

mod private {
    use vortex_compute::lane_kernels::IndexedSource;

    use super::ElementTuple;

    /// The crate-private half of [`super::IndexedElementTuple`].
    ///
    /// Rust trait methods have the visibility of their trait. This companion keeps source
    /// construction from decoded columns out of the public unstable API.
    pub trait DecodedSource: ElementTuple {
        fn decoded_source<'a>(
            columns: &'a Self::Columns,
            row_count: usize,
        ) -> Option<impl IndexedSource<Item = Self::Elems<'a>>>;
    }
}
