// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lookups of needles among the elements of an array.

use std::hash::BuildHasher;

use num_traits::ToPrimitive;
use num_traits::WrappingSub;
use rustc_hash::FxBuildHasher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::collect_bits;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashTable;
use vortex_utils::aliases::hash_map::HashTableEntry;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::decimal::converted_buffer;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::scalar_fn::fns::binary::build_row_comparator;
use crate::validity::Validity;

/// A set whose span of values needs at most this many bits per element is looked up through a
/// bitmap over the span, bounding the bitmap to a few words per element.
const BITMAP_BITS_PER_ELEMENT: u128 = 64;
/// A span this narrow is looked up through a bitmap whatever the size of the set.
///
/// A lookup in a bitmap of up to 128 KiB stays in the L2 cache and does not branch, where a
/// binary search through a sparse set mispredicts about half its comparisons.
const BITMAP_MIN_BITS: u128 = 1 << 20;
/// Needles beyond this many are decoded and looked up a batch of this many at a time.
const NEEDLE_BATCH_LEN: usize = 8192;

/// Looks up needles among the elements of an array, for many batches of needles.
///
/// The haystack counterpart of [`RepeatedArrayProbe`](crate::RepeatedArrayProbe): where a probe
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
        DType::Decimal(decimal, _) => {
            let values_type = decimal.canonical_values_type();
            let elements = elements.execute::<DecimalArray>(ctx)?;
            let all_valid = Mask::new_true(elements.len());
            match_each_decimal_value_type!(values_type, |T| {
                let values = converted_buffer::<T>(&elements, &all_valid)?;
                Box::new(DecimalSet(IntegerSet::new(values, ctx.allocator())))
                    as Box<dyn HaystackState>
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
    /// A bitmap over the values' span when the span is dense, and a sorted slice otherwise.
    ///
    /// A hash set and, for a handful of elements, a linear scan both lost to the binary search at
    /// every set size measured by the `list_contains_set` benchmark, up to 16 384 elements.
    fn new(values: Buffer<T>, allocator: &BufferAllocatorRef) -> Self {
        match integer_bitmap(&values, allocator) {
            Some((min, bitmap)) => Self::Bitmap { min, bitmap },
            None => Self::Sorted(sorted_values(values)),
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
            Self::Sorted(sorted) => collect_bits(
                needles,
                |needle| sorted.binary_search(&needle).is_ok(),
                allocator,
            ),
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
        if needles.len() <= NEEDLE_BATCH_LEN || needles.is::<Primitive>() {
            let needles = needles
                .clone()
                .execute::<PrimitiveArray>(ctx)?
                .reinterpret_cast(T::PTYPE);
            let bits = self.0.contains(needles.as_slice::<T>(), ctx.allocator());
            return Ok((bits, needles.validity()?));
        }

        // Decoding a long compressed array in batches keeps each batch of needles in the cache
        // between decoding and looking it up, instead of writing out the whole array first.
        let len = needles.len();
        let mut bits = BitBufferMut::with_capacity_in(len, ctx.allocator().clone());
        for start in (0..len).step_by(NEEDLE_BATCH_LEN) {
            let end = (start + NEEDLE_BATCH_LEN).min(len);
            let batch = needles
                .slice(start..end)?
                .execute::<PrimitiveArray>(ctx)?
                .reinterpret_cast(T::PTYPE);
            bits.append_buffer(&self.0.contains(batch.as_slice::<T>(), ctx.allocator()));
        }
        Ok((bits.freeze(), needles.validity()?))
    }
}

/// The unscaled values of decimals, as the narrowest type that holds every value of their
/// precision.
///
/// A needle has the precision of the elements, so a valid needle converts to that type whatever
/// its own storage width, and a needle stored at that type converts without a copy.
struct DecimalSet<T>(IntegerSet<T>);

impl<T: NativeDecimalType + SetInteger> HaystackState for DecimalSet<T> {
    fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        let needles = needles.clone().execute::<DecimalArray>(ctx)?;
        let validity = needles.validity()?;
        let valid = validity.execute_mask(needles.len(), ctx)?;
        let values = converted_buffer::<T>(&needles, &valid)?;
        Ok((self.0.contains(&values, ctx.allocator()), validity))
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
        // The comparator materializes child buffers and validity once, including decimal
        // widening and nested offsets. Each search then reads those buffers directly.
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

fn integer_bitmap<T: SetInteger>(
    values: &[T],
    allocator: &BufferAllocatorRef,
) -> Option<(T, BitBuffer)> {
    let min = *values.iter().min()?;
    let max = *values.iter().max()?;
    let span = max.offset_from(min)?;
    if span == usize::MAX
        || span as u128 >= (values.len() as u128 * BITMAP_BITS_PER_ELEMENT).max(BITMAP_MIN_BITS)
    {
        return None;
    }
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
    Some((min, bitmap.freeze()))
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
