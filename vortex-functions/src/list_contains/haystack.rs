// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lookups of needles among the elements of an array.

use std::hash::BuildHasher;

use num_traits::ToPrimitive;
use num_traits::WrappingSub;
use rustc_hash::FxBuildHasher;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::slice::SliceArraySlotsExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_integer_ptype;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar_fn::builtin::compare::build_row_comparator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::collect_bits;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashTable;
use vortex_utils::aliases::hash_map::HashTableEntry;
/// A set whose span of values needs at most this many bits per element is looked up through a
/// bitmap over the span, bounding the bitmap to a few words per element.
const BITMAP_BITS_PER_ELEMENT: u128 = 64;
/// A span this narrow fits in L1 and is looked up through a bitmap regardless of set size.
const L1_BITMAP_MIN_BITS: u128 = 1 << 12;
/// A span this narrow is looked up through a bitmap once the set is large enough that binary
/// search loses to L2-resident bitmap probes.
///
/// A lookup in a bitmap of up to 128 KiB stays in the L2 cache and does not branch, where a
/// binary search through a sparse set mispredicts about half its comparisons.
const BITMAP_MIN_BITS: u128 = 1 << 20;
/// Minimum set size for the wide L2 bitmap; smaller sets stay in L1 via sorted binary search.
const L2_BITMAP_MIN_ELEMENTS: usize = 16;
/// Needles beyond this many are decoded and looked up a batch of this many at a time.
const NEEDLE_BATCH_LEN: usize = 8192;

/// Looks up needles among the elements of an array, for many batches of needles.
///
/// The haystack counterpart of [`RepeatedArrayProbe`](vortex_array::RepeatedArrayProbe): where a probe
/// reads the rows of an array by index, a haystack answers whether each row of a batch of
/// needles is an element of the array. A haystack pays off only over repeated lookups, so it
/// has no one-off form. It is built eagerly. A holder that must build it lazily, or share it
/// between threads, keeps it in a shared cell.
///
/// Integers, and floats by their bit patterns, are looked up in an [`IntegerSet`] of their own
/// type, and decimals in an [`IntegerSet`] of their unscaled values. UTF-8 and binary values are
/// found through a hash table, and nested values through sorted row indices with the same
/// comparator as equality. No haystack constructs per-element expressions or materializes
/// scalars in its loop.
pub(crate) struct ArrayHaystack {
    state: Box<dyn HaystackState>,
}

impl ArrayHaystack {
    /// Builds the haystack of the elements of `array`.
    ///
    /// A null element never equals a needle, so the haystack leaves null elements out.
    pub(crate) fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let valid = array.validity()?.execute_mask(array.len(), ctx)?;
        let members = if valid.all_true() {
            array.clone()
        } else {
            array.filter(valid)?
        };

        Ok(Self {
            state: new_state(members, ctx)?,
        })
    }

    /// One membership bit per needle, and the validity of the needles, which have the dtype of
    /// the elements.
    pub(crate) fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        self.state.contains(needles, ctx)
    }
}

/// The state that an [`ArrayHaystack`] keeps for the canonical encoding of its elements.
trait HaystackState: Send + Sync {
    /// One membership bit per needle, and the validity of the needles.
    fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)>;
}

/// Builds the state of the non-null `elements`.
fn new_state(elements: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Box<dyn HaystackState>> {
    Ok(match elements.dtype() {
        DType::Primitive(ptype, _) => {
            let ptype = bit_pattern_ptype(*ptype);
            let elements = elements
                .execute::<PrimitiveArray>(ctx)?
                .reinterpret_cast(ptype);
            match_each_integer_ptype!(ptype, |T| {
                let set = IntegerSet::<T>::new(elements.into_buffer(), ctx.allocator());
                Box::new(PrimitiveSet(set)) as Box<dyn HaystackState>
            })
        }
        DType::Decimal(..) => {
            let elements = elements.execute::<DecimalArray>(ctx)?;
            match_each_decimal_value_type!(elements.values_type(), |T| {
                let set = IntegerSet::new(elements.buffer::<T>(), ctx.allocator());
                Box::new(DecimalSet(set)) as Box<dyn HaystackState>
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            Box::new(BytesSet::new(elements.execute::<VarBinViewArray>(ctx)?))
        }
        _ => Box::new(RowSet::try_new(elements, ctx)?),
    })
}

/// The distinct values of a set of integers of one type.
enum IntegerSet<T> {
    /// Values spanning a dense range: one bit per value of the span above `min`.
    Bitmap { min: T, bitmap: BitBuffer },
    /// Values sorted without duplicates.
    Sorted(Buffer<T>),
}

impl<T: SetInteger> IntegerSet<T> {
    /// Direct equality for up to four elements, a bitmap over the values' span when the span is
    /// dense, and a sorted slice with binary search otherwise.
    fn new(values: Buffer<T>, allocator: &BufferAllocatorRef) -> Self {
        if values.len() <= 4 {
            return Self::Sorted(sorted_values(values));
        }
        let (Some(&min), Some(&max)) = (values.iter().min(), values.iter().max()) else {
            return Self::Sorted(values);
        };
        let Some(span) = max.offset_from(min) else {
            return Self::Sorted(sorted_values(values));
        };
        if span == usize::MAX {
            return Self::Sorted(sorted_values(values));
        }
        let span_u128 = span as u128;
        if span_u128 < L1_BITMAP_MIN_BITS
            || (span_u128 < BITMAP_MIN_BITS
                && has_at_least_distinct(&values, L2_BITMAP_MIN_ELEMENTS))
        {
            return Self::Bitmap {
                min,
                bitmap: build_integer_bitmap(min, span, &values, allocator),
            };
        }
        if span_u128 >= values.len() as u128 * BITMAP_BITS_PER_ELEMENT {
            return Self::Sorted(sorted_values(values));
        }
        let distinct = sorted_values(values);
        if span_u128 >= distinct.len() as u128 * BITMAP_BITS_PER_ELEMENT {
            return Self::Sorted(distinct);
        }
        Self::Bitmap {
            min,
            bitmap: build_integer_bitmap(min, span, &distinct, allocator),
        }
    }

    /// One bit per needle, set when the needle is an element.
    fn contains(&self, needles: &[T], allocator: &BufferAllocatorRef) -> BitBuffer {
        match self {
            Self::Bitmap { min, bitmap } => {
                let bytes = bitmap.inner().as_slice();
                let (bit_offset, len) = (bitmap.offset(), bitmap.len());
                collect_bits(
                    needles,
                    // A needle below the smallest element wraps past the bitmap, so one
                    // comparison checks both bounds. A needle outside the bitmap reads its
                    // first bit and is masked off, so the lookup does not branch.
                    |needle| {
                        let offset = needle.offset_from(*min).unwrap_or(usize::MAX);
                        let in_span = offset < len;
                        let bit = if in_span { offset } else { 0 } + bit_offset;
                        // SAFETY: `bit` is below the bitmap's offset plus its length.
                        let byte = unsafe { *bytes.get_unchecked(bit / 8) };
                        in_span & ((byte >> (bit % 8)) & 1 != 0)
                    },
                    allocator,
                )
            }
            Self::Sorted(sorted) => match sorted.as_slice() {
                [] => BitBuffer::full_in(false, needles.len(), allocator.clone()),
                &[m0] => collect_bits(needles, |needle| needle == m0, allocator),
                &[m0, m1] => {
                    collect_bits(needles, |needle| (needle == m0) | (needle == m1), allocator)
                }
                &[m0, m1, m2] => collect_bits(
                    needles,
                    |needle| (needle == m0) | (needle == m1) | (needle == m2),
                    allocator,
                ),
                &[m0, m1, m2, m3] => collect_bits(
                    needles,
                    |needle| (needle == m0) | (needle == m1) | (needle == m2) | (needle == m3),
                    allocator,
                ),
                sorted => collect_bits(
                    needles,
                    |needle| sorted.binary_search(&needle).is_ok(),
                    allocator,
                ),
            },
        }
    }
}

/// Primitive integers, or floats by their bit patterns.
///
/// A float is a member exactly when the compare kernel would call it equal to an element, which
/// is when their bit patterns match — distinguishing `-0.0` from `0.0` and one NaN payload from
/// another — so floats are looked up by their bits, as integers.
struct PrimitiveSet<T>(IntegerSet<T>);

impl<T: NativePType + SetInteger> HaystackState for PrimitiveSet<T> {
    fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        if needles.len() > NEEDLE_BATCH_LEN && !needles.is::<Primitive>() {
            let first = needles.slice(0..NEEDLE_BATCH_LEN)?.optimize_ctx(ctx)?;
            if slices_without_full_decode(&first, ctx)? {
                // Decoding a long compressed array in batches keeps each batch of needles in the
                // cache between decoding and looking it up, instead of writing out the whole array
                // first.
                let len = needles.len();
                let mut bits = BitBufferMut::with_capacity_in(len, ctx.allocator().clone());
                let batch = first
                    .execute::<PrimitiveArray>(ctx)?
                    .reinterpret_cast(T::PTYPE);
                bits.append_buffer(&self.0.contains(batch.as_slice::<T>(), ctx.allocator()));
                for start in (NEEDLE_BATCH_LEN..len).step_by(NEEDLE_BATCH_LEN) {
                    let end = (start + NEEDLE_BATCH_LEN).min(len);
                    let batch = needles
                        .slice(start..end)?
                        .execute::<PrimitiveArray>(ctx)?
                        .reinterpret_cast(T::PTYPE);
                    bits.append_buffer(&self.0.contains(batch.as_slice::<T>(), ctx.allocator()));
                }
                return Ok((bits.freeze(), needles.validity()?));
            }
        }

        let needles = needles
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .reinterpret_cast(T::PTYPE);
        let bits = self.0.contains(needles.as_slice::<T>(), ctx.allocator());
        Ok((bits, needles.validity()?))
    }
}

/// Whether every remaining `Slice` node in `array` is handled by a slice kernel rather than
/// falling back to `Slice::execute`, which materializes the full child column.
fn slices_without_full_decode(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    if let Some(slice) = array.as_opt::<Slice>() {
        let child = slice.child();

        // The chunked slice kernel slices each chunk that a batch touches, and the batches
        // together touch every chunk, so each chunk must also slice without a full decode.
        if let Some(chunked) = child.as_opt::<Chunked>() {
            for chunk in chunked.iter_chunks() {
                // A full-range slice returns the chunk itself, so drop the last row to get a
                // `Slice` node that shows how the chunk handles a slice.
                let sliced = chunk
                    .slice(0..chunk.len().saturating_sub(1))?
                    .optimize_ctx(ctx)?;
                if !slices_without_full_decode(&sliced, ctx)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }

        let Some(rewritten) = ctx.try_execute_parent(array)? else {
            return Ok(false);
        };
        let rewritten = rewritten.optimize_ctx(ctx)?;
        return slices_without_full_decode(&rewritten, ctx);
    }

    for child in array.children().iter() {
        if !slices_without_full_decode(child, ctx)? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Unscaled decimal values at the canonical width shared by the elements and needles.
struct DecimalSet<T>(IntegerSet<T>);

impl<T: NativeDecimalType + SetInteger> HaystackState for DecimalSet<T> {
    fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        let needles = needles.clone().execute::<DecimalArray>(ctx)?;
        let bits = self.0.contains(&needles.buffer::<T>(), ctx.allocator());
        Ok((bits, needles.validity()?))
    }
}

/// UTF-8 or binary elements, found by their views where the view decides.
///
/// A value of at most 12 bytes is inlined whole in its view, zero-padded as the compare kernel
/// requires, so a short needle is an element exactly when its whole view is the view of a short
/// element: no data buffer is read and no bytes are hashed. A long needle is found through a
/// table of the long elements, hashed by their bytes. The first 8 bytes of its view, the length
/// and a 4-byte prefix, are compared before the bytes after the prefix.
///
/// A one-bit filter over the heads of the elements was measured in front of these lookups and
/// removed: it helped short needles by about a tenth and cost long needles that share a prefix,
/// whose heads collide, up to a quarter.
struct BytesSet {
    hasher: FxBuildHasher,
    /// The distinct whole views of the elements of at most 12 bytes.
    short: HashTable<u128>,
    /// The elements, for the bytes of the long ones.
    elements: VarBinViewArray,
    /// The indices of the distinct long elements, hashed by their bytes.
    long: HashTable<u32>,
}

impl BytesSet {
    fn new(elements: VarBinViewArray) -> Self {
        let hasher = FxBuildHasher;
        let views = elements.views();
        let buffers = data_buffers(&elements);

        // Sized up front, so that no insert grows a table and hashes every element again.
        let short_len = views.iter().filter(|view| view.is_inlined()).count();
        let mut short = HashTable::with_capacity(short_len);
        let mut long = HashTable::with_capacity(views.len() - short_len);

        for (idx, view) in views.iter().enumerate() {
            if view.is_inlined() {
                let whole = view.as_u128();
                if let HashTableEntry::Vacant(vacant) = short.entry(
                    hasher.hash_one(whole),
                    |&other| other == whole,
                    |&other| hasher.hash_one(other),
                ) {
                    vacant.insert(whole);
                }
                continue;
            }

            let value = view.bytes(&buffers);
            let bytes = |other: u32| views[other as usize].bytes(&buffers);
            if let HashTableEntry::Vacant(vacant) = long.entry(
                hasher.hash_one(value),
                |&other| bytes(other) == value,
                |&other| hasher.hash_one(bytes(other)),
            ) {
                vacant.insert(
                    u32::try_from(idx).vortex_expect("a list holds fewer than 2^32 elements"),
                );
            }
        }

        Self {
            hasher,
            short,
            elements,
            long,
        }
    }

    /// Whether `view`, which points into `buffers`, is the view of an element.
    #[inline]
    fn contains_view(
        &self,
        view: &BinaryView,
        buffers: &[&[u8]],
        element_views: &[BinaryView],
        element_buffers: &[&[u8]],
    ) -> bool {
        if view.is_inlined() {
            let whole = view.as_u128();
            return self
                .short
                .find(self.hasher.hash_one(whole), |&other| other == whole)
                .is_some();
        }

        // Once the heads are equal, the bytes can differ only after the 4-byte prefix.
        let head = view_head(view);
        let value = view.bytes(buffers);
        self.long
            .find(self.hasher.hash_one(value), |&idx| {
                let element = &element_views[idx as usize];
                view_head(element) == head && element.bytes(element_buffers)[4..] == value[4..]
            })
            .is_some()
    }
}

impl HaystackState for BytesSet {
    fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        let element_views = self.elements.views();
        let element_buffers = data_buffers(&self.elements);
        let array = needles.clone().execute::<VarBinViewArray>(ctx)?;
        let buffers = data_buffers(&array);
        let bits = collect_bits(
            array.views(),
            |view: BinaryView| self.contains_view(&view, &buffers, element_views, &element_buffers),
            ctx.allocator(),
        );
        Ok((bits, array.validity()?))
    }
}

/// The head of a view: the `u32` length of its value and the first 4 bytes of the value,
/// zero-padded for a value shorter than 4 bytes.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the head is the low 8 bytes"
)]
fn view_head(view: &BinaryView) -> u64 {
    view.as_u128() as u64
}

/// Recursively canonical elements, indexed in sorted order with duplicates removed.
struct RowSet {
    elements: ArrayRef,
    indices: Vec<usize>,
}

impl RowSet {
    fn try_new(elements: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let elements = elements.execute::<RecursiveCanonical>(ctx)?.0.into_array();
        let mut indices: Vec<usize> = (0..elements.len()).collect();
        if !indices.is_empty() {
            let compare = build_row_comparator(&elements, &elements, ctx)?;
            if !indices.is_sorted_by(|&lhs, &rhs| compare(lhs, rhs).is_le()) {
                indices.sort_unstable_by(|&lhs, &rhs| compare(lhs, rhs));
            }
            indices.dedup_by(|lhs, rhs| compare(*lhs, *rhs).is_eq());
        }
        Ok(Self { elements, indices })
    }
}

impl HaystackState for RowSet {
    fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        if self.indices.is_empty() {
            return Ok((
                BitBuffer::full_in(false, needles.len(), ctx.allocator().clone()),
                needles.validity()?,
            ));
        }
        let needles = needles
            .clone()
            .execute::<RecursiveCanonical>(ctx)?
            .0
            .into_array();
        // The comparator materializes child buffers, validity, and nested offsets once. Each
        // search then reads those buffers directly.
        let compare = build_row_comparator(&self.elements, &needles, ctx)?;
        let bits = BitBuffer::collect_bool_in(
            needles.len(),
            |row| {
                self.indices
                    .binary_search_by(|&element| compare(element, row))
                    .is_ok()
            },
            ctx.allocator().clone(),
        );
        Ok((bits, needles.validity()?))
    }
}

/// The host slices of an array's data buffers, indexed by a view's buffer index.
fn data_buffers(array: &VarBinViewArray) -> Vec<&[u8]> {
    (0..array.data_buffers().len())
        .map(|idx| array.buffer(idx).as_slice())
        .collect()
}

/// The integer type with a float's bit pattern, or the type itself for an integer.
fn bit_pattern_ptype(ptype: PType) -> PType {
    match ptype {
        PType::F16 => PType::U16,
        PType::F32 => PType::U32,
        PType::F64 => PType::U64,
        _ => ptype,
    }
}

/// An unsigned modular distance, rejecting offsets too wide to address a bitmap.
/// Keeping the subtraction at the physical width also handles signed ranges spanning zero.
trait SetInteger: Copy + Ord + Send + Sync + 'static {
    fn offset_from(self, min: Self) -> Option<usize>;
}

macro_rules! impl_set_integer {
    ($($signed:ty => $unsigned:ty),* $(,)?) => {
        $(impl SetInteger for $signed {
            fn offset_from(self, min: Self) -> Option<usize> {
                usize::try_from((self as $unsigned).wrapping_sub(min as $unsigned)).ok()
            }
        })*
    };
}

impl_set_integer!(
    u8 => u8, u16 => u16, u32 => u32, u64 => u64,
    i8 => u8, i16 => u16, i32 => u32, i64 => u64, i128 => u128,
);

impl SetInteger for i256 {
    fn offset_from(self, min: Self) -> Option<usize> {
        self.wrapping_sub(&min).to_usize()
    }
}

/// Whether `values` contains at least `min_distinct` distinct values, without allocating or sorting.
fn has_at_least_distinct<T: Copy + PartialEq>(values: &[T], min_distinct: usize) -> bool {
    debug_assert!(min_distinct <= L2_BITMAP_MIN_ELEMENTS);
    if values.len() < min_distinct {
        return false;
    }
    let Some(&first) = values.first() else {
        return min_distinct == 0;
    };
    let mut seen = [first; L2_BITMAP_MIN_ELEMENTS];
    let mut count = 1;
    for &value in &values[1..] {
        if !seen[..count].contains(&value) {
            if count + 1 >= min_distinct {
                return true;
            }
            seen[count] = value;
            count += 1;
        }
    }
    count >= min_distinct
}

fn build_integer_bitmap<T: SetInteger>(
    min: T,
    span: usize,
    values: &[T],
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let mut bitmap = BitBufferMut::from_buffer(
        BufferMut::zeroed_in((span + 1).div_ceil(8), allocator.clone()),
        0,
        span + 1,
    );
    for &value in values {
        bitmap.set(
            value
                .offset_from(min)
                .vortex_expect("value within bitmap span"),
        );
    }
    bitmap.freeze()
}

fn sorted_values<T: Copy + Ord + Send + Sync + 'static>(values: Buffer<T>) -> Buffer<T> {
    if values.is_sorted_by(|a, b| a < b) {
        return values;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    Buffer::from(sorted)
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::hash::Hash;
    use std::hash::Hasher;
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_array::Array;
    use vortex_array::ArrayEq;
    use vortex_array::ArrayHash;
    use vortex_array::ArrayId;
    use vortex_array::ArrayParts;
    use vortex_array::ArraySlots;
    use vortex_array::ArrayView;
    use vortex_array::EqMode;
    use vortex_array::ExecutionResult;
    use vortex_array::OperationsVTable;
    use vortex_array::VTable;
    use vortex_array::ValidityVTable;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::WidenArray;
    use vortex_array::arrays::slice::SliceExecuteAdaptor;
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::Nullability;
    use vortex_array::optimizer::kernels::ArrayKernelsExt;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySession;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use super::*;

    #[derive(Clone, Debug)]
    struct CountingEncoding;

    #[derive(Clone, Debug)]
    struct CountingData {
        values: PrimitiveArray,
        decodes: Arc<AtomicUsize>,
    }

    impl fmt::Display for CountingData {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("CountingData")
        }
    }

    impl ArrayHash for CountingData {
        fn array_hash<H: Hasher>(&self, state: &mut H, _eq_mode: EqMode) {
            self.values.len().hash(state);
        }
    }

    impl ArrayEq for CountingData {
        fn array_eq(&self, other: &Self, _eq_mode: EqMode) -> bool {
            Arc::ptr_eq(&self.decodes, &other.decodes)
        }
    }

    impl VTable for CountingEncoding {
        type TypedArrayData = CountingData;
        type OperationsVTable = Self;
        type ValidityVTable = Self;

        fn id(&self) -> ArrayId {
            static ID: CachedId = CachedId::new("vortex.test.counting_haystack_needles");
            *ID
        }

        fn validate(
            &self,
            _data: &Self::TypedArrayData,
            _dtype: &DType,
            _len: usize,
            _slots: &[Option<ArrayRef>],
        ) -> VortexResult<()> {
            Ok(())
        }

        fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
            0
        }

        fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
            vortex_panic!("CountingEncoding has no buffers")
        }

        fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
            None
        }

        fn with_buffers(
            &self,
            _array: ArrayView<'_, Self>,
            _buffers: &[BufferHandle],
        ) -> VortexResult<ArrayParts<Self>> {
            vortex_panic!("CountingEncoding has no buffers")
        }

        fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
            vortex_panic!("CountingEncoding has no slots")
        }

        fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
            array.decodes.fetch_add(1, Ordering::Relaxed);
            Ok(ExecutionResult::done(array.values.clone().into_array()))
        }
    }

    impl OperationsVTable<CountingEncoding> for CountingEncoding {
        type ProbeState = ();

        fn scalar_at(
            array: ArrayView<'_, CountingEncoding>,
            index: usize,
            ctx: &mut ExecutionCtx,
        ) -> VortexResult<Scalar> {
            array.values.execute_scalar(index, ctx)
        }
    }

    impl ValidityVTable<CountingEncoding> for CountingEncoding {
        fn validity(array: ArrayView<'_, CountingEncoding>) -> VortexResult<Validity> {
            array.values.validity()
        }
    }

    impl SliceKernel for CountingEncoding {
        fn slice(
            _array: ArrayView<'_, Self>,
            _range: Range<usize>,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>> {
            Ok(None)
        }
    }

    /// An array without a slice kernel that counts how many times it is decoded.
    fn counting_array(
        values: PrimitiveArray,
        decodes: &Arc<AtomicUsize>,
    ) -> VortexResult<ArrayRef> {
        Ok(Array::try_from_parts(ArrayParts::new(
            CountingEncoding,
            values.dtype().clone(),
            values.len(),
            CountingData {
                values,
                decodes: Arc::clone(decodes),
            },
            ArraySlots::new(),
        ))?
        .into_array())
    }

    #[test]
    fn small_wide_span_uses_sorted_large_uses_bitmap() {
        let session = VortexSession::empty().with::<ArraySession>();
        let ctx = session.create_execution_ctx();

        let small_wide = IntegerSet::new(Buffer::from(vec![0i64, 1, 2, 1 << 19]), ctx.allocator());
        assert!(matches!(small_wide, IntegerSet::Sorted(_)));

        let small_narrow =
            IntegerSet::new(Buffer::from(vec![0i64, 1, 2, 3, 1000]), ctx.allocator());
        assert!(matches!(small_narrow, IntegerSet::Bitmap { .. }));

        let tiny_narrow = IntegerSet::new(Buffer::from(vec![0i64, 1, 2, 1000]), ctx.allocator());
        assert!(matches!(tiny_narrow, IntegerSet::Sorted(_)));

        let large_wide: Vec<i64> = (0..256).map(|i| i * 4000).collect();
        let large_wide = IntegerSet::new(Buffer::from(large_wide), ctx.allocator());
        assert!(matches!(large_wide, IntegerSet::Bitmap { .. }));

        let mut dup_wide = vec![0i64; L2_BITMAP_MIN_ELEMENTS];
        dup_wide.push(1 << 19);
        let dup_wide = IntegerSet::new(Buffer::from(dup_wide), ctx.allocator());
        assert!(matches!(dup_wide, IntegerSet::Sorted(_)));
    }

    #[test]
    fn non_sliceable_needles_decode_once_instead_of_per_batch() -> VortexResult<()> {
        let len = NEEDLE_BATCH_LEN * 3;
        let narrow = PrimitiveArray::from_iter(0..i32::try_from(len).vortex_expect("fits in i32"));
        let widened = WidenArray::try_new(
            narrow.clone().into_array(),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .into_array();
        let chunked = ChunkedArray::try_new(
            vec![narrow.clone().into_array(), narrow.clone().into_array()],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();

        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let widened_slice = widened.slice(0..NEEDLE_BATCH_LEN)?.optimize_ctx(&ctx)?;
        assert!(slices_without_full_decode(&widened_slice, &mut ctx)?);

        let chunked_slice = chunked.slice(0..NEEDLE_BATCH_LEN)?.optimize_ctx(&ctx)?;
        assert!(slices_without_full_decode(&chunked_slice, &mut ctx)?);

        let decodes = Arc::new(AtomicUsize::new(0));
        let unsliced_needles = counting_array(narrow, &decodes)?;

        let elements = PrimitiveArray::from_iter([5i32, 9_000, 20_000]).into_array();
        let haystack = ArrayHaystack::try_new(&elements, &mut ctx)?;
        let (bits, _) = haystack.contains(&unsliced_needles, &mut ctx)?;

        assert_eq!(decodes.load(Ordering::Relaxed), 1);
        assert!(bits.value(5));
        assert!(bits.value(9_000));
        assert!(bits.value(20_000));
        assert!(!bits.value(0));
        assert_eq!(bits.true_count(), 3);

        Ok(())
    }

    #[test]
    fn declining_slice_kernel_decodes_once_instead_of_per_batch() -> VortexResult<()> {
        let len = NEEDLE_BATCH_LEN * 3;
        let values = PrimitiveArray::from_iter(0..i32::try_from(len).vortex_expect("fits in i32"));
        let decodes = Arc::new(AtomicUsize::new(0));
        let needles = counting_array(values, &decodes)?;

        let session = VortexSession::empty().with::<ArraySession>();
        session.kernels().register_execute_parent_kernel(
            Slice.id(),
            CountingEncoding,
            SliceExecuteAdaptor(CountingEncoding),
        );
        let mut ctx = session.create_execution_ctx();

        let first = needles.slice(0..NEEDLE_BATCH_LEN)?.optimize_ctx(&ctx)?;
        assert!(!slices_without_full_decode(&first, &mut ctx)?);

        let elements = PrimitiveArray::from_iter([5i32, 9_000, 20_000]).into_array();
        let haystack = ArrayHaystack::try_new(&elements, &mut ctx)?;
        let (bits, _) = haystack.contains(&needles, &mut ctx)?;

        assert_eq!(decodes.load(Ordering::Relaxed), 1);
        assert_eq!(bits.true_count(), 3);

        Ok(())
    }

    #[test]
    fn chunks_without_slice_kernel_decode_once_instead_of_per_batch() -> VortexResult<()> {
        let len = NEEDLE_BATCH_LEN * 3;
        let values = PrimitiveArray::from_iter(0..i32::try_from(len).vortex_expect("fits in i32"));
        let decodes = Arc::new(AtomicUsize::new(0));

        // The first chunk is a lazy slice, so slicing it again keeps the counting array below.
        let sliced_chunk = counting_array(values.clone(), &decodes)?.slice(1..len)?;
        let chunk = counting_array(values, &decodes)?;
        let needles = ChunkedArray::try_new(
            vec![sliced_chunk, chunk],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();

        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let first = needles.slice(0..NEEDLE_BATCH_LEN)?.optimize_ctx(&ctx)?;
        assert!(!slices_without_full_decode(&first, &mut ctx)?);

        let elements = PrimitiveArray::from_iter([5i32, 9_000, 20_000]).into_array();
        let haystack = ArrayHaystack::try_new(&elements, &mut ctx)?;
        let (bits, _) = haystack.contains(&needles, &mut ctx)?;

        assert_eq!(decodes.load(Ordering::Relaxed), 2);
        assert_eq!(bits.true_count(), 6);

        Ok(())
    }
}
