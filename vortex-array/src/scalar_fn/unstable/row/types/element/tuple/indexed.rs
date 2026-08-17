// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validated indexed access to tuples of decoded input columns.
//!
//! [`IndexedElementTuple`] adapts row arguments to the lane-kernel interface after batch execution
//! proves that every input covers the requested row range.

use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::LaneZip;

use super::ElementTuple;
use crate::scalar_fn::unstable::row::InputElement;

/// An argument tuple that supports a validated dense indexed traversal.
///
/// Every [`ElementTuple`] implements this trait. Its source delegates each lane read to the tuple's
/// unchecked view access after batch execution validates every decoded column length once.
pub trait IndexedElementTuple: ElementTuple {
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
        T::view_len(&self.view)
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

impl<A: InputElement> IndexedElementTuple for (A,) {
    type Source<'a> = UnaryTupleSource<ElementSource<'a, A>>;

    unsafe fn indexed_source<'a>(views: Self::Views<'a>, _row_count: usize) -> Self::Source<'a> {
        UnaryTupleSource(ElementSource::new(views.0))
    }
}

impl<A: InputElement, B: InputElement> IndexedElementTuple for (A, B) {
    type Source<'a> = LaneZip<ElementSource<'a, A>, ElementSource<'a, B>>;

    unsafe fn indexed_source<'a>(views: Self::Views<'a>, _row_count: usize) -> Self::Source<'a> {
        LaneZip::new(ElementSource::new(views.0), ElementSource::new(views.1))
    }
}

macro_rules! indexed_element_tuple {
    ($($t:ident),+) => {
        impl<$($t: InputElement),+> IndexedElementTuple for ($($t,)+) {
            type Source<'a> = ElementTupleSource<'a, ($($t,)+)>;

            unsafe fn indexed_source<'a>(
                views: Self::Views<'a>,
                row_count: usize,
            ) -> Self::Source<'a> {
                ElementTupleSource { views, row_count }
            }
        }
    };
}

indexed_element_tuple!(A, B, C);
indexed_element_tuple!(A, B, C, D);
indexed_element_tuple!(A, B, C, D, E);
indexed_element_tuple!(A, B, C, D, E, F);
indexed_element_tuple!(A, B, C, D, E, F, G);
indexed_element_tuple!(A, B, C, D, E, F, G, H);
indexed_element_tuple!(A, B, C, D, E, F, G, H, I);
indexed_element_tuple!(A, B, C, D, E, F, G, H, I, J);
indexed_element_tuple!(A, B, C, D, E, F, G, H, I, J, K);
indexed_element_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
