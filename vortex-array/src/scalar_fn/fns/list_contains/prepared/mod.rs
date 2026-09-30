// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Probing a constant set. [`PreparedSetArray`] holds a constant list together with the probe
//! built from its elements, so that the kernels of a needle encoding can probe their own values.

mod array;
mod literal;

use std::hash::BuildHasher;

pub use array::PreparedSet;
pub use array::PreparedSetArray;
pub use array::PreparedSetData;
pub use literal::PreparedSetLiteral;
use num_traits::ToPrimitive;
use num_traits::WrappingSub;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashTable;
use vortex_utils::aliases::hash_map::HashTableEntry;
use vortex_utils::aliases::hash_map::RandomState;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::decimal::converted_buffer;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::scalar_fn::fns::binary::build_row_comparator;
use crate::scalar_fn::fns::binary::collect_bits;
use crate::validity::Validity;

/// A set whose span of values needs at most this many bits per element is probed through a bitmap
/// over the span, bounding the bitmap to a few words per element.
const BITMAP_BITS_PER_ELEMENT: u128 = 64;
/// A span this narrow is probed through a bitmap whatever the size of the set.
const BITMAP_MIN_BITS: u128 = 1 << 12;
/// The bits per element of the filter of view heads. With one bit per head, about one in eight
/// needles that are not elements passes the filter of a large set.
const HEAD_FILTER_BITS_PER_ELEMENT: usize = 8;

/// Probes the non-null elements of a set for membership.
///
/// Integers, and floats by their bit patterns, are probed as an [`IntegerSet`] of their own type,
/// and decimals as an [`IntegerSet`] of their unscaled values. UTF-8 and binary values are found
/// through a hash table, and nested values through sorted row indices with the same comparator as
/// equality. No probe constructs per-element expressions or materializes scalars in its loop.
trait Probe: Send + Sync {
    /// One membership bit per needle, and the validity of the needles, which have the dtype of
    /// the elements.
    fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)>;

    /// Whether the probe holds its values in a bitmap.
    #[cfg(test)]
    fn is_bitmap(&self) -> bool {
        false
    }
}

/// Builds the probe of the non-null `elements`.
fn new_probe(elements: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Box<dyn Probe>> {
    Ok(match elements.dtype() {
        DType::Primitive(ptype, _) => {
            let ptype = bit_pattern_ptype(*ptype);
            let elements = elements
                .execute::<PrimitiveArray>(ctx)?
                .reinterpret_cast(ptype);
            match_each_integer_ptype!(ptype, |T| {
                let set = IntegerSet::<T>::new(elements.into_buffer(), ctx.allocator());
                Box::new(PrimitiveSet(set)) as Box<dyn Probe>
            })
        }
        DType::Decimal(decimal, _) => {
            let values_type = DecimalType::smallest_decimal_value_type(decimal);
            let elements = elements.execute::<DecimalArray>(ctx)?;
            let all_valid = Mask::new_true(elements.len());
            match_each_decimal_value_type!(values_type, |T| {
                let values = converted_buffer::<T>(&elements, &all_valid)?;
                Box::new(DecimalSet(IntegerSet::new(values, ctx.allocator()))) as Box<dyn Probe>
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
            Self::Bitmap { min, bitmap } => collect_bits(
                needles,
                // A needle below the smallest element wraps past the bitmap, so one comparison
                // checks both bounds.
                |needle| {
                    needle
                        .offset_from(*min)
                        .is_some_and(|offset| offset < bitmap.len() && bitmap.value(offset))
                },
                allocator,
            ),
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
/// another — so floats are probed by their bits, as integers.
struct PrimitiveSet<T>(IntegerSet<T>);

impl<T: NativePType + SetInteger> Probe for PrimitiveSet<T> {
    fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        let needles = needles
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .reinterpret_cast(T::PTYPE);
        let bits = self.0.contains(needles.as_slice::<T>(), ctx.allocator());
        Ok((bits, needles.validity()?))
    }

    #[cfg(test)]
    fn is_bitmap(&self) -> bool {
        matches!(self.0, IntegerSet::Bitmap { .. })
    }
}

/// The unscaled values of decimals, as the narrowest type that holds every value of their
/// precision.
///
/// A needle has the precision of the elements, so a valid needle converts to that type whatever
/// its own storage width, and a needle stored at that type converts without a copy.
struct DecimalSet<T>(IntegerSet<T>);

impl<T: NativeDecimalType + SetInteger> Probe for DecimalSet<T> {
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

    #[cfg(test)]
    fn is_bitmap(&self) -> bool {
        matches!(self.0, IntegerSet::Bitmap { .. })
    }
}

/// UTF-8 or binary elements, found by their views, so that most needles are decided without a
/// read of a data buffer or a hash of their bytes.
///
/// The first 8 bytes of a view, its head, hold the length of the value and its first 4 bytes,
/// zero-padded, whether the value is inlined or not. A value of at most 12 bytes is inlined whole,
/// zero-padded as the compare kernel requires. Thus a needle is probed in three steps, the
/// cheapest first:
///
/// 1. The filter of the elements' heads rejects most non-members with one multiply and one bit.
/// 2. A short needle is an element exactly when its whole view is the view of a short element.
/// 3. A long needle is found through a table of the long elements, hashed by their bytes. Its
///    head is compared before the bytes after the prefix.
struct BytesSet {
    heads: HeadFilter,
    hasher: RandomState,
    /// The distinct whole views of the elements of at most 12 bytes.
    short: HashTable<u128>,
    /// The elements, for the bytes of the long ones.
    elements: VarBinViewArray,
    /// The indices of the distinct long elements, hashed by their bytes.
    long: HashTable<u32>,
}

impl BytesSet {
    fn new(elements: VarBinViewArray) -> Self {
        let hasher = RandomState::default();
        let views = elements.views();
        let buffers = data_buffers(&elements);

        let mut heads = HeadFilter::with_capacity(views.len());
        // Sized up front, so that no insert grows a table and hashes every element again.
        let short_len = views.iter().filter(|view| view.is_inlined()).count();
        let mut short = HashTable::with_capacity(short_len);
        let mut long = HashTable::with_capacity(views.len() - short_len);

        for (idx, view) in views.iter().enumerate() {
            heads.insert(view_head(view));

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
            heads,
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
        let head = view_head(view);
        if !self.heads.may_contain(head) {
            return false;
        }

        if view.is_inlined() {
            let whole = view.as_u128();
            return self
                .short
                .find(self.hasher.hash_one(whole), |&other| other == whole)
                .is_some();
        }

        // The heads are equal, so the bytes can differ only after the 4-byte prefix.
        let value = view.bytes(buffers);
        self.long
            .find(self.hasher.hash_one(value), |&idx| {
                let element = &element_views[idx as usize];
                view_head(element) == head && element.bytes(element_buffers)[4..] == value[4..]
            })
            .is_some()
    }
}

impl Probe for BytesSet {
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

/// A filter of view heads with no false negatives: an inserted head always tests as present.
///
/// It holds about [`HEAD_FILTER_BITS_PER_ELEMENT`] bits per element, and tests one bit, chosen by
/// a multiplicative hash of the head.
struct HeadFilter {
    words: Box<[u64]>,
    /// The shift that takes the top bits of the hash as the index of a bit.
    shift: u32,
}

impl HeadFilter {
    fn with_capacity(elements: usize) -> Self {
        let bits = (elements * HEAD_FILTER_BITS_PER_ELEMENT)
            .next_power_of_two()
            .max(64);
        Self {
            words: vec![0; bits / 64].into_boxed_slice(),
            shift: u64::BITS - bits.trailing_zeros(),
        }
    }

    #[inline]
    fn bit(&self, head: u64) -> usize {
        // Fibonacci hashing: the top bits of the product depend on every bit of the head.
        let hash = head.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        usize::try_from(hash >> self.shift).vortex_expect("a bit index fits a usize")
    }

    fn insert(&mut self, head: u64) {
        let bit = self.bit(head);
        self.words[bit / 64] |= 1 << (bit % 64);
    }

    #[inline]
    fn may_contain(&self, head: u64) -> bool {
        let bit = self.bit(head);
        self.words[bit / 64] & (1 << (bit % 64)) != 0
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

impl Probe for RowSet {
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

#[cfg(test)]
mod tests;
