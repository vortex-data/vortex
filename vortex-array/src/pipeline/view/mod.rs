// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod flatten;

use crate::ArrayRef;
use crate::pipeline::N;
use crate::pipeline::selection::Selection;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

use crate::pipeline::bits::BitVector;
use crate::pipeline::bits::BitViewMut;
use crate::pipeline::types::{Element, VType};

pub struct View<'a> {
    _phantom: std::marker::PhantomData<&'a ()>,
}

pub struct TypedView<'a, E> {
    view: View<'a>,
    _phantom: std::marker::PhantomData<E>,
}

impl<E> AsRef<[E; N]> for TypedView<'_, E>
where
    E: Element,
{
    fn as_ref(&self) -> &[E; N] {
        todo!()
    }
}

pub struct TypedViewMut<'a, E> {
    view: ViewMut<'a>,
    _phantom: std::marker::PhantomData<E>,
}

impl<E> AsRef<[E; N]> for TypedViewMut<'_, E>
where
    E: Element,
{
    fn as_ref(&self) -> &[E; N] {
        todo!()
    }
}

impl<E> AsMut<[E; N]> for TypedViewMut<'_, E>
where
    E: Element,
{
    fn as_mut(&mut self) -> &mut [E; N] {
        todo!()
    }
}

/// A vector is the atomic unit of canonical data in Vortex.
///
/// Like with our canonical arrays, it is physically typed, not logically typed. So each DType
/// has a well-defined physical representation in vector form.
///
/// I'm not quite sure on sizing yet. We could follow DuckDB and make vectors 2k elements. We
/// could follow FastLanes and make them 1k elements. Or we could do something interesting where
/// we pick the largest power of two that lets us fit ~3-4 vectors into the L1 cache. For example,
/// strings may be 1k elements, but u8 integers may be 8k elements. Many compute functions operate
/// over two inputs of the same type, so this could be interesting.
///
/// Interestingly, Vectors don't own their own data. In fact, I'm considering calling them 'views'.
/// This also solves our problem of zero-copy export by allowing us to pass down an output buffer
/// from an external source. This tends to work well since these external sources typically agree
/// with us on the physical representation of data, e.g. DuckDB, Arrow, Numpy, etc.
///
/// ## Why is this a single type-erased struct?
///
/// If we used generics at this level, we would taint all functions that use this type with a
/// generic type. To remove the generic, we'd need to introduce a trait, at which point we're
/// forced into both dynamic dispatch, and boxed return types. We also cannot down-cast a dynamic
/// trait that holds borrowed data since `Any` requires a static lifetime.
///
/// ## How do we handle custom encodings, e.g. FSST, RoaringBitMap, ZStd, etc.?
///
/// I could imagine a VarBinView vector (i.e. it has 16-byte views in its elements buffer), but
/// is able to delay decompression of the data buffers. These could be stored as child arrays and
/// decompressed on-demand, since this is now an opaque operation (and then call export on the
/// child data arrays using a slices mask? We'd be masking binary data... that sounds slow).
/// In conclusion... I'm not really sure yet.
///
/// What about dictionary arrays? Are they even important at this level?
/// I have done a "medium amount" of thinking on this, and I think we can actually get away with
/// not supporting dictionary encoding at the vector level. Vortex arrays actually define the
/// encoding tree, and therefore can decide how to implement a compute function. So where it's
/// possible to return a dictionary encoded thing, e.g. to DuckDB, we would have some sort of
/// Vortex Array -> DuckDB function that would check for top-level dictionaries and handle the
/// conversion at that layer.

/// ## Can we re-use parts of the pipeline to avoid common-subexpression elimination?
///
/// This gets tricky... Suppose we start with a Vortex expression. We can then pass that naively
/// through pipeline construction. This now represents a physical execution plan. At this point,
/// we could in theory optimize the pipeline by removing common sub-expressions, such as
/// canonicalizing the same field multiple times to pass into two comparison operators.
///
/// We'd then need some way to buffer the intermediate results as both expressions are driven.
/// Maybe this works? Not sure yet.
pub struct ViewMut<'a> {
    /// The physical type of the vector, which defines how the elements are stored.
    pub(super) vtype: VType,
    /// A pointer to the allocated elements buffer.
    /// Alignment is at least the size of the element type.
    /// The capacity of the elements buffer is N * size_of::<T>() where T is the element type.
    // TODO(ngates): it would be nice to guarantee _wider_ alignment, ideally 128 bytes, so that
    //  we can use aligned load/store instructions for wide SIMD lanes.
    pub(super) elements: *mut u8,
    /// The validity mask for the vector, indicating which elements in the buffer are valid.
    /// This value can be `None` if the expected DType is `NonNullable`.
    pub(super) validity: Option<BitViewMut<'a>>,
    // A selection mask over the elements and validity of the vector.
    pub(super) selection: Selection,

    /// Additional buffers of data used by the vector, such as string data.
    // TODO(ngates): ideally these buffers are compressed somehow? E.g. using FSST?
    #[allow(dead_code)]
    pub(super) data: Vec<ByteBuffer>,
    // Additional arrays used by the vector, such as...?
    #[allow(dead_code)]
    pub(super) children: Vec<ArrayRef>,

    /// Marker defining the lifetime of the contents of the vector.
    pub(super) _marker: std::marker::PhantomData<&'a mut ()>,
}

impl<'a> ViewMut<'a> {
    pub fn new<E: Element>(elements: &'a mut [E], validity: Option<BitViewMut<'a>>) -> Self {
        assert_eq!(elements.len(), N);
        Self {
            vtype: E::vtype(),
            elements: elements.as_mut_ptr().cast(),
            validity,
            selection: Selection::default(),
            data: vec![],
            children: vec![],
            _marker: Default::default(),
        }
    }

    pub fn as_view(&self) -> View<'a> {
        View {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Re-interpret cast the vector into a new type where the element has the same width.
    #[inline(always)]
    pub fn reinterpret_as<E: Element>(&mut self) {
        assert_eq!(
            self.vtype.byte_width(),
            size_of::<E>(),
            "Cannot reinterpret {} as {}",
            self.vtype,
            E::vtype()
        );
        self.vtype = E::vtype();
    }

    /// Return the logical length of the vector, which is the number of selected elements.
    pub fn len(&self) -> usize {
        match self.selection {
            Selection::Prefix { len } => len,
            Selection::Constant { len, .. } => len,
            Selection::Mask(ref mask) => mask.true_count(),
        }
    }

    /// Returns a mutable handle to the elements array.
    #[inline(always)]
    pub fn elements<E: Element>(&mut self) -> &'a mut [E; N] {
        assert_eq!(self.vtype, E::vtype(), "Invalid type for canonical view");
        // SAFETY: We assume that the elements are of type E and that the view is valid.
        unsafe { &mut *(self.elements.cast::<[E; N]>()) }
    }

    /// Returns an immutable slice of the elements in the vector.
    #[inline(always)]
    pub fn as_ref<E: Element>(&self) -> &'a [E] {
        assert_eq!(self.vtype, E::vtype(), "Invalid type for canonical view");
        unsafe { std::slice::from_raw_parts(self.elements.cast::<E>(), N) }
    }

    /// Returns a mutable slice of the elements in the vector, allowing for modification.
    /// FIXME(ngates): test the performance if we return &mut [E; N] instead of &[E].
    #[inline(always)]
    pub fn as_mut<E: Element>(&mut self) -> &'a mut [E] {
        assert_eq!(self.vtype, E::vtype(), "Invalid type for canonical view");
        unsafe { std::slice::from_raw_parts_mut(self.elements.cast::<E>(), N) }
    }

    /// Access the validity mask of the vector.
    ///
    /// ## Panics
    ///
    /// Panics if the vector does not support validity, i.e. if the DType was non-nullable when
    /// it was created.
    pub fn validity(&mut self) -> &mut BitViewMut<'a> {
        self.validity
            .as_mut()
            .vortex_expect("Vector does not support validity")
    }

    pub fn add_buffer(&mut self, buffer: ByteBuffer) {
        self.data.push(buffer);
    }

    #[inline(always)]
    pub fn set_selection(&mut self, selection: Selection) {
        #[cfg(debug_assertions)]
        {
            match &selection {
                Selection::Prefix { len } => {
                    assert!(
                        *len <= N,
                        "Selection prefix length must be less than or equal to N"
                    );
                }
                Selection::Constant { len, element } => {
                    assert!(
                        *len <= N,
                        "Selection constant length must be less than or equal to N"
                    );
                    assert!(
                        *element < N,
                        "Selection constant element must be less than N"
                    );
                }
                Selection::Mask(_mask) => {}
            }
        }
        self.selection = selection;
    }

    pub fn set_selection_len(&mut self, len: usize) {
        assert!(len <= N);
        self.selection = Selection::Prefix { len };
    }

    pub fn set_selection_mask(&mut self, mask: BitVector) {
        match mask.true_count() {
            0 => self.selection = Selection::Prefix { len: 0 },
            N => self.selection = Selection::Prefix { len: N },
            _ => {
                self.selection = Selection::Mask(mask);
            }
        }
    }

    /// Whether the vector is in a flat representation, meaning it has no selection reordering.
    pub fn is_flat(&self) -> bool {
        match self.selection {
            Selection::Prefix { .. } => true,
            Selection::Constant { .. } => true,
            Selection::Mask(_) => false,
        }
    }
}
