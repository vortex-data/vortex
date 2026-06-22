// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stack-allocatable parent representation used by the `reduce_parent` dispatch chain.
//!
//! [`ParentRef`] either borrows an existing heap-allocated [`ArrayRef`], or borrows
//! stack-allocated construction state. The construction-side optimizer can borrow
//! `ArrayParts` before materializing an `ArrayInner`, so matchers and parent-reduce
//! rules can attempt reduction without first allocating an `Arc<ArrayInner<_>>`.
//!
//! Stack-backed parents lazily materialize an `ArrayRef` into an internal [`OnceLock`]
//! only when a downstream consumer explicitly asks a [`ParentView`] to materialize.

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::OnceLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::array::ArrayData;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArraySlots;
use crate::array::ParentView;
use crate::array::VTable;
use crate::dtype::DType;
use crate::matcher::AsParent;
use crate::matcher::Matcher;
use crate::optimizer::optimize_owned;

/// A parent array, possibly stack-allocated, used by the `reduce_parent` dispatch chain.
///
/// Carries the metadata needed to dispatch parent-reduce rules (encoding id, dtype,
/// length, encoding-specific data, slots) regardless of whether the parent is backed
/// by an existing [`ArrayRef`] or by borrowed [`ArrayParts`]. Stack-backed parents
/// materialize an [`ArrayRef`] into an internal cache on first explicit materialization.
pub struct ParentRef<'a> {
    encoding_id: ArrayId,
    dtype: &'a DType,
    len: usize,
    slots: &'a [Option<ArrayRef>],
    data: ParentData<'a>,
    /// Lazily-populated materialization slot used by stack-backed parents.
    /// Heap-backed parents return their borrowed [`ArrayRef`] directly and never
    /// touch this cache.
    cache: OnceLock<ArrayRef>,
}

/// Type-erased payload for [`ParentRef`].
///
/// Carries `&dyn Any` rather than `&V`/`&V::TypedArrayData` so [`ParentRef`] is not
/// itself generic over `V`. The `+ Send + Sync` bound mirrors the bounds on
/// [`VTable`] and `V::TypedArrayData`, keeping [`ParentRef`]
/// and the [`ParentView`] built on top of it `Send + Sync`.
type AnyRef<'a> = &'a (dyn Any + Send + Sync);

enum ParentData<'a> {
    Heap {
        array: &'a ArrayRef,
        data: AnyRef<'a>,
    },
    Parts {
        vtable: AnyRef<'a>,
        data: AnyRef<'a>,
        materialize: MaterializeFn,
        reduce: ReduceFn,
    },
}

/// Function pointer that materializes stack-borrowed parts into an owned [`ArrayRef`].
///
/// The `vtable` and `data` arguments are the borrowed `&V` and `&V::TypedArrayData`
/// previously stashed as `&dyn Any` in [`ParentData::Parts`]. The implementation
/// downcasts them, clones into owned values, and produces an `ArrayRef`.
type MaterializeFn = fn(
    vtable: &(dyn Any + Send + Sync),
    data: &(dyn Any + Send + Sync),
    dtype: &DType,
    len: usize,
    slots: &[Option<ArrayRef>],
) -> ArrayRef;

/// Function pointer that runs encoding `V`'s self-reduce rules against a (possibly
/// stack-borrowed) parent.
///
/// Stored alongside [`MaterializeFn`] in [`ParentData::Parts`] so [`ArrayParts::optimize`](crate::array::ArrayParts::optimize)
/// can dispatch `V::reduce` without being generic over `V`. The implementation builds a
/// [`ParentView`] over the borrowed parts, so a rule that only inspects metadata never
/// forces a materialization.
type ReduceFn = fn(parent: &ParentRef<'_>) -> VortexResult<Option<ArrayRef>>;

impl<'a> ParentRef<'a> {
    /// Build a [`ParentRef`] borrowing a heap-allocated [`ArrayRef`].
    #[inline]
    pub fn from_array_ref(array: &'a ArrayRef) -> Self {
        let inner = array.inner();
        Self {
            encoding_id: inner.encoding_id,
            dtype: &inner.dtype,
            len: inner.len,
            slots: &inner.slots,
            data: ParentData::Heap {
                array,
                data: inner.data.as_any(),
            },
            cache: OnceLock::new(),
        }
    }

    /// Build a [`ParentRef`] borrowing construction parts before materialization.
    ///
    /// The returned [`ParentRef`] owns the cache slot for the lazily materialized
    /// [`ArrayRef`], so callers don't need to thread an external scratch through.
    #[inline]
    pub fn from_parts<V: VTable>(parts: &'a ArrayParts<V>) -> Self {
        Self {
            encoding_id: parts.vtable.id(),
            dtype: &parts.dtype,
            len: parts.len,
            slots: &parts.slots,
            data: ParentData::Parts {
                vtable: &parts.vtable,
                data: &parts.data,
                materialize: materialize_parts::<V>,
                reduce: reduce_parts::<V>,
            },
            cache: OnceLock::new(),
        }
    }

    /// Run the parent encoding's self-reduce rules against the parent.
    ///
    /// Mirrors [`ArrayRef::reduce`](crate::ArrayRef::reduce) for the `ParentRef` dispatch
    /// chain. Heap-backed parents delegate to the existing array; stack-backed parents
    /// dispatch through the stored [`ReduceFn`] so the borrowed parts only materialize if a
    /// rule reaches for an [`ArrayRef`]. The reduced array is validated to preserve the
    /// parent's len and dtype, matching the heap path.
    fn reduce(&self) -> VortexResult<Option<ArrayRef>> {
        let reduced = match self.data {
            ParentData::Heap { array, .. } => return array.reduce(),
            ParentData::Parts { reduce, .. } => reduce(self)?,
        };
        let Some(reduced) = reduced else {
            return Ok(None);
        };
        vortex_ensure!(
            reduced.len() == self.len,
            "Reduced array length mismatch from {} to {}",
            self.encoding_id,
            reduced.encoding_id()
        );
        vortex_ensure!(
            reduced.dtype() == self.dtype,
            "Reduced array dtype mismatch from {} to {}",
            self.encoding_id,
            reduced.encoding_id()
        );
        Ok(Some(reduced))
    }

    /// Returns the encoding id of the parent.
    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn encoding_id(&self) -> ArrayId {
        self.encoding_id
    }

    /// Returns the dtype of the parent.
    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn dtype(&self) -> &DType {
        self.dtype
    }

    /// Returns the length of the parent.
    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the parent is empty.
    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the slots of the parent.
    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn slots(&self) -> &[Option<ArrayRef>] {
        self.slots
    }

    /// Consume this `ParentRef` and return the cached materialization, if one exists.
    ///
    /// This is used by owned [`ArrayParts::optimize`] to avoid materializing twice when
    /// a stack-backed parent was forced into an [`ArrayRef`] by a rule that did not fire.
    fn into_cached_array_ref(self) -> Option<ArrayRef> {
        self.cache.into_inner()
    }

    /// Returns `true` if this parent's encoding matches `V`.
    ///
    /// Cheap encoding-id check that works for both heap- and stack-backed parents
    /// without forcing materialization.
    #[inline]
    #[allow(clippy::same_name_method)]
    pub(crate) fn is_encoding<V: VTable>(&self) -> bool {
        match self.data {
            ParentData::Heap { data, .. } => data.is::<ArrayData<V>>(),
            ParentData::Parts { vtable, .. } => vtable.is::<V>(),
        }
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub(crate) fn typed_data<V: VTable>(&self) -> Option<&V::TypedArrayData> {
        match self.data {
            ParentData::Heap { data, .. } => data
                .downcast_ref::<ArrayData<V>>()
                .map(|array_data| &array_data.data),
            ParentData::Parts { data, .. } => data.downcast_ref::<V::TypedArrayData>(),
        }
    }

    /// Try to extract a [`ParentView`] for the parent's encoding `V`.
    ///
    /// Returns `None` if the parent's encoding is not `V`. No materialization happens
    /// up front. Materialization is only available through
    /// [`ParentView::materialize_array_ref`].
    ///
    /// This is the low-level entry point used by the blanket `VTable` matcher
    /// implementation. Prefer [`AsParent::as_opt`] for matcher-based downcasts.
    #[allow(clippy::same_name_method)]
    pub fn as_parent_view<V: VTable>(&self) -> Option<ParentView<'_, V>> {
        let data = self.typed_data::<V>()?;
        // SAFETY: `typed_data::<V>()` returned Some, so the parent's encoding is
        // `V` and `data` is the `V::TypedArrayData` reachable through `self`.
        Some(unsafe { ParentView::new_unchecked(self, data) })
    }

    /// Does the parent match the given matcher.
    ///
    /// Mirrors [`ArrayRef::is`](ArrayRef::is) for the parent-side dispatch
    /// chain. Routes through [`Matcher::matches`] so matchers that can answer with
    /// a cheap encoding-id check don't force a downcast.
    #[allow(clippy::same_name_method)]
    pub fn is<M: Matcher>(&self) -> bool {
        M::matches(self)
    }

    /// Returns the parent downcast by the given matcher, or `None` if it doesn't match.
    ///
    /// Mirrors [`ArrayRef::as_opt`](ArrayRef::as_opt) for the parent-side
    /// dispatch chain. The returned match borrows from `self`, so stack-backed
    /// parents stay on the stack until a consumer explicitly materializes a
    /// [`ParentView`].
    #[allow(clippy::same_name_method)]
    pub fn as_opt<M: Matcher>(&self) -> Option<M::Match<'_>> {
        M::try_match(self)
    }

    /// Returns the parent downcast by the given matcher, panicking if it doesn't match.
    ///
    /// Mirrors [`ArrayRef::as_`](ArrayRef::as_).
    #[allow(clippy::same_name_method)]
    pub fn as_<M: Matcher>(&self) -> M::Match<'_> {
        self.as_opt::<M>().vortex_expect("Failed to downcast")
    }
}

#[allow(clippy::same_name_method)]
impl AsParent for ParentRef<'_> {
    #[inline]
    fn encoding_id(&self) -> ArrayId {
        ParentRef::encoding_id(self)
    }

    #[inline]
    fn dtype(&self) -> &DType {
        ParentRef::dtype(self)
    }

    #[inline]
    fn len(&self) -> usize {
        ParentRef::len(self)
    }

    #[inline]
    fn slots(&self) -> &[Option<ArrayRef>] {
        ParentRef::slots(self)
    }

    #[inline]
    fn is_encoding<V: VTable>(&self) -> bool {
        ParentRef::is_encoding::<V>(self)
    }

    #[inline]
    fn typed_data<V: VTable>(&self) -> Option<&V::TypedArrayData> {
        ParentRef::typed_data::<V>(self)
    }

    #[inline]
    fn as_parent_view<V: VTable>(&self) -> Option<ParentView<'_, V>> {
        ParentRef::as_parent_view::<V>(self)
    }
}

impl<V: VTable> ArrayParts<V> {
    /// Optimize already-valid construction parts, consuming the original parts on a miss.
    ///
    /// This mirrors one iteration of [`ArrayRef::optimize`](crate::optimizer::ArrayOptimizer):
    /// the parent's own `reduce` rules are tried first, then `reduce_parent` on each child
    /// slot. Both run against the stack-borrowed parent, so a reduction that only inspects
    /// metadata never allocates an `Arc<ArrayInner<_>>`. If no rule applies and the
    /// stack-backed parent was not materialized by a rule, the result is built with
    /// [`ArrayParts::into_array`] directly without cloning the parts.
    pub fn optimize(self) -> VortexResult<ArrayRef> {
        let parent = ParentRef::from_parts(&self);
        if let Some(reduced) = parent.reduce()? {
            return Ok(optimize_owned(reduced, None)?.0);
        }

        for (slot_idx, slot) in parent.slots.iter().enumerate() {
            let Some(child) = slot else { continue };

            if let Some(reduced) = child.reduce_parent(&parent, slot_idx)? {
                return Ok(optimize_owned(reduced, None)?.0);
            }
        }

        if let Some(cached) = parent.into_cached_array_ref() {
            return Ok(cached);
        }

        Ok(self.into_array())
    }
}

impl Debug for ParentRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let heap_backed = matches!(self.data, ParentData::Heap { .. });
        f.debug_struct("ParentRef")
            .field("encoding", &self.encoding_id())
            .field("dtype", self.dtype())
            .field("len", &self.len())
            .field("heap_backed", &heap_backed)
            .finish()
    }
}

impl<'a> From<&'a ArrayRef> for ParentRef<'a> {
    fn from(array: &'a ArrayRef) -> Self {
        Self::from_array_ref(array)
    }
}

/// Explicit materialization hook used by [`ParentView`].
pub(crate) trait ParentMaterializer: Send + Sync {
    /// Returns a materialized [`ArrayRef`].
    ///
    /// For heap-backed parents this is a cheap reference return. For stack-backed parents this
    /// triggers materialization on first call and caches the result.
    fn materialize_array_ref(&self) -> &ArrayRef;
}

impl ParentMaterializer for ParentRef<'_> {
    #[inline]
    fn materialize_array_ref(&self) -> &ArrayRef {
        match self.data {
            ParentData::Heap { array, .. } => array,
            ParentData::Parts {
                vtable,
                data,
                materialize,
                ..
            } => self
                .cache
                .get_or_init(|| materialize(vtable, data, self.dtype, self.len, self.slots)),
        }
    }
}

/// Materializes stack-borrowed parts of encoding `V` into an owned [`ArrayRef`].
///
/// Used as the function pointer stored inside [`ParentData::Parts`]. The
/// `vtable`/`data` arguments are `&V` and `&V::TypedArrayData` erased to `&dyn Any`;
/// they are downcast and cloned into a fresh `ArrayParts<V>` which is then turned
/// into an `ArrayRef`. Validation is skipped: stack-borrowed parts were validated
/// when the originating `ArrayParts<V>` was constructed.
fn materialize_parts<V: VTable>(
    vtable: &(dyn Any + Send + Sync),
    data: &(dyn Any + Send + Sync),
    dtype: &DType,
    len: usize,
    slots: &[Option<ArrayRef>],
) -> ArrayRef {
    let vtable = vtable
        .downcast_ref::<V>()
        .vortex_expect("ParentRef materialize: vtable type mismatch");
    let data = data
        .downcast_ref::<V::TypedArrayData>()
        .vortex_expect("ParentRef materialize: data type mismatch");
    let slots: ArraySlots = slots.iter().cloned().collect();
    ArrayParts::new(vtable.clone(), dtype.clone(), len, data.clone())
        .with_slots(slots)
        .into_array()
}

/// Runs encoding `V`'s self-reduce rules against a (possibly stack-borrowed) parent.
///
/// Used as the [`ReduceFn`] stored inside [`ParentData::Parts`]. Builds a
/// [`ParentView`] over the borrowed parts and dispatches to [`VTable::reduce`]; the view
/// only materializes if a rule explicitly asks for an [`ArrayRef`].
fn reduce_parts<V: VTable>(parent: &ParentRef<'_>) -> VortexResult<Option<ArrayRef>> {
    let view = parent
        .as_parent_view::<V>()
        .vortex_expect("ParentRef reduce: encoding mismatch");
    V::reduce(view)
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::ParentRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFnArray;
    use crate::arrays::Slice;
    use crate::arrays::SliceArray;
    use crate::arrays::Struct;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::pack::Pack;
    use crate::scalar_fn::fns::pack::PackOptions;

    #[test]
    fn parts_parent_ref_exposes_array_view() -> VortexResult<()> {
        let child = BoolArray::from_iter([true, false, true]).into_array();
        let parts = SliceArray::try_new_parts(child, 1..3)?;
        let parent = ParentRef::from_parts(&parts);

        let view = parent
            .as_opt::<Slice>()
            .expect("Slice parts should match a Slice array view");

        assert_eq!(view.slice_range(), &(1..3));
        assert_eq!(view.len(), 2);

        Ok(())
    }

    #[test]
    fn parts_parent_ref_explicit_materialize() -> VortexResult<()> {
        let child = BoolArray::from_iter([true, false, true]).into_array();
        let parts = SliceArray::try_new_parts(child, 1..3)?;
        let parent = ParentRef::from_parts(&parts);

        let view = parent
            .as_opt::<Slice>()
            .expect("Slice parts should match a Slice array view");

        // Reading metadata through the view does NOT force materialization.
        assert_eq!(view.slice_range(), &(1..3));
        assert_eq!(view.len(), 2);

        // Explicit materialization produces an ArrayRef.
        let array_ref = view.materialize_array_ref();
        assert_eq!(array_ref.len(), 2);

        Ok(())
    }

    /// Optimizing borrowed parts must produce the same array as materializing them and
    /// calling [`ArrayRef::optimize`](crate::optimizer::ArrayOptimizer) — the two paths
    /// differ only in whether the wrapper is heap-allocated.
    ///
    /// Regression test for [`ArrayParts::optimize`] skipping the parent's own `reduce`
    /// rules. A `Pack` scalar function collapses to a `StructArray` via the `ScalarFn`
    /// encoding's self-`reduce`. No `reduce_parent` rule mirrors this, so the reduction is
    /// only reachable through self-`reduce`: before `optimize` ran `reduce` first the stack
    /// path returned the `ScalarFn` wrapper while materialize-then-optimize returned the
    /// struct.
    #[test]
    fn optimize_matches_heap_path() -> VortexResult<()> {
        let a = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let b = PrimitiveArray::from_iter([4i32, 5, 6]).into_array();
        let len = a.len();
        let pack = Pack.bind(PackOptions {
            names: ["a", "b"].into(),
            nullability: Nullability::NonNullable,
        });

        let heap = ScalarFnArray::try_new_with_len(pack.clone(), vec![a.clone(), b.clone()], len)?
            .into_array()
            .optimize()?;
        let parts = ScalarFnArray::try_new_parts(pack, vec![a, b], len)?;
        let stack = parts.optimize()?;

        assert!(
            heap.is::<Struct>(),
            "heap path should collapse Pack to a struct"
        );
        assert!(
            stack.is::<Struct>(),
            "stack path should collapse Pack to a struct"
        );
        assert_arrays_eq!(stack, heap);

        Ok(())
    }
}
