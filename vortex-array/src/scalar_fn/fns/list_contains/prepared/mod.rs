// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Probing a constant set: the probe structure is built from a [`ListContainsSet`] for canonical
//! evaluation. Chunked expressions dispatch per chunk so each encoding can specialize membership.

use std::hash::BuildHasher;

use num_traits::ToPrimitive;
use num_traits::WrappingSub;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
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
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::widened_buffer;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::scalar::DecimalValue;
use crate::scalar_fn::fns::binary::build_row_comparator;
use crate::scalar_fn::fns::binary::collect_bits;
use crate::scalar_fn::fns::list_contains::ListContainsSet;
use crate::validity::Validity;

/// A set whose span of values needs at most this many bits per element is probed through a bitmap
/// over the span, bounding the bitmap to a few words per element.
const BITMAP_BITS_PER_ELEMENT: u128 = 64;
/// A span this narrow is probed through a bitmap whatever the size of the set.
const BITMAP_MIN_BITS: u128 = 1 << 12;

/// A [`ListContainsSet`] with the structure that probes it, prepared once and then run over any
/// number of needle arrays of the set's dtype.
///
/// Every needle is probed against a bitmap, a sorted set or a hash table. Decimals use their
/// unscaled integers with the same bitmap and sorted-value strategy as primitives. Nested
/// values use sorted row indices with the same comparator as equality. No probe constructs
/// per-element expressions or materializes scalars in its loop.
pub struct PreparedSet {
    set: ListContainsSet,
    probe: Probe,
}

enum Probe {
    /// Integers, or floats by their bit patterns, spanning a dense range: one bit per value of the
    /// span above `min_offset`, the smallest element as a `usize`.
    Bitmap {
        min_offset: usize,
        span: usize,
        bitmap: BitBuffer,
    },
    /// Integers, or floats by their bit patterns, sorted without duplicates.
    Sorted(PrimitiveArray),
    /// Dense unscaled decimal values, including 128- and 256-bit storage.
    DecimalBitmap {
        min: DecimalValue,
        bitmap: BitBuffer,
    },
    /// Sorted, distinct unscaled decimal values.
    DecimalSorted(DecimalArray),
    /// UTF-8 or binary elements, found through a table of their indices hashed by their bytes, so
    /// that no element is copied.
    Bytes {
        elements: VarBinViewArray,
        hasher: RandomState,
        table: HashTable<u32>,
    },
    /// Recursively canonical elements, indexed in sorted order with duplicates removed.
    Rows {
        elements: ArrayRef,
        indices: Vec<usize>,
    },
}

impl ListContainsSet {
    /// Builds the structure that probes this set.
    pub fn prepare(self, ctx: &mut ExecutionCtx) -> VortexResult<PreparedSet> {
        let probe = match self.elements().dtype() {
            DType::Primitive(ptype, _) => {
                let ptype = bit_pattern_ptype(*ptype);
                let elements = self
                    .elements()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?
                    .reinterpret_cast(ptype);
                match_each_integer_ptype!(ptype, |T| {
                    integer_probe::<T>(elements, ctx.allocator())
                })
            }
            DType::Decimal(..) => {
                let elements = self.elements().clone().execute::<DecimalArray>(ctx)?;
                match_each_decimal_value_type!(elements.values_type(), |T| {
                    let values = elements.buffer::<T>();
                    if let Some((min, bitmap)) = integer_bitmap(&values, ctx.allocator()) {
                        Probe::DecimalBitmap {
                            min: min.into(),
                            bitmap,
                        }
                    } else {
                        Probe::DecimalSorted(DecimalArray::new(
                            sorted_values(values),
                            elements.decimal_dtype(),
                            Validity::NonNullable,
                        ))
                    }
                })
            }
            DType::Utf8(_) | DType::Binary(_) => {
                bytes_probe(self.elements().clone().execute::<VarBinViewArray>(ctx)?)
            }
            _ => {
                let elements = self
                    .elements()
                    .clone()
                    .execute::<RecursiveCanonical>(ctx)?
                    .0
                    .into_array();
                let mut indices: Vec<usize> = (0..elements.len()).collect();
                if !indices.is_empty() {
                    let compare = build_row_comparator(&elements, &elements, ctx)?;
                    if !indices.is_sorted_by(|&lhs, &rhs| compare(lhs, rhs).is_le()) {
                        indices.sort_unstable_by(|&lhs, &rhs| compare(lhs, rhs));
                    }
                    indices.dedup_by(|lhs, rhs| compare(*lhs, *rhs).is_eq());
                }
                Probe::Rows { elements, indices }
            }
        };
        Ok(PreparedSet { set: self, probe })
    }
}

impl PreparedSet {
    /// The dtype of every result, as the expression declares it.
    pub fn dtype(&self) -> DType {
        DType::Bool(self.set.nullability)
    }

    /// Whether each of `needles`, which have the set's dtype, is a member of the set.
    ///
    /// The result has the nullability the expression declares, whatever the needles' encoding.
    pub fn contains(&self, needles: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        match &self.probe {
            Probe::Bitmap { .. } | Probe::Sorted(_) => self.contains_primitive(needles, ctx),
            Probe::DecimalBitmap { .. } | Probe::DecimalSorted(_) => {
                self.contains_decimal(needles, ctx)
            }
            Probe::Bytes {
                elements,
                hasher,
                table,
            } => self.contains_bytes(elements, hasher, table, needles, ctx),
            Probe::Rows { elements, indices } => {
                self.contains_rows(elements, indices, needles, ctx)
            }
        }
    }

    /// A float is a member exactly when the compare kernel would call it equal to an element, which
    /// is when their bit patterns match — distinguishing `-0.0` from `0.0` and one NaN payload from
    /// another — so floats are probed by their bits, as integers.
    fn contains_primitive(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = needles.clone().execute::<PrimitiveArray>(ctx)?;
        let ptype = bit_pattern_ptype(primitive.ptype());
        let values = primitive.reinterpret_cast(ptype);
        let bits = match_each_integer_ptype!(ptype, |T| {
            self.probe
                .integer_bits(values.as_slice::<T>(), ctx.allocator())
        });
        self.set.finish(bits, primitive.validity()?)
    }

    fn contains_decimal(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let needles = needles.clone().execute::<DecimalArray>(ctx)?;
        // Logical precision and scale agree, but physical widths can differ. Widen once to
        // the common width so out-of-range needles cannot truncate into false matches.
        let bits = match &self.probe {
            Probe::DecimalBitmap { min, bitmap } => {
                let common = min.decimal_type().max(needles.values_type());
                match_each_decimal_value_type!(common, |T| {
                    let min = min.cast::<T>().vortex_expect("lossless decimal widening");
                    let values = widened_buffer::<T>(&needles);
                    collect_bits(
                        &values,
                        |value| {
                            value
                                .offset_from(min)
                                .is_some_and(|offset| offset < bitmap.len() && bitmap.value(offset))
                        },
                        ctx.allocator(),
                    )
                })
            }
            Probe::DecimalSorted(sorted) => {
                let common = sorted.values_type().max(needles.values_type());
                match_each_decimal_value_type!(common, |T| {
                    let sorted = widened_buffer::<T>(sorted);
                    let values = widened_buffer::<T>(&needles);
                    sorted_bits(&sorted, &values, ctx.allocator())
                })
            }
            _ => unreachable!("decimal needles meet a decimal probe"),
        };
        self.set.finish(bits, needles.validity()?)
    }

    fn contains_bytes(
        &self,
        elements: &VarBinViewArray,
        hasher: &RandomState,
        table: &HashTable<u32>,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let element_views = elements.views();
        let element_buffers = data_buffers(elements);
        let array = needles.clone().execute::<VarBinViewArray>(ctx)?;
        let buffers = data_buffers(&array);
        let bits = collect_bits(
            array.views(),
            |view: BinaryView| {
                let value = view_bytes(&view, &buffers);
                table
                    .find(hasher.hash_one(value), |&idx| {
                        view_bytes(&element_views[idx as usize], &element_buffers) == value
                    })
                    .is_some()
            },
            ctx.allocator(),
        );
        self.set.finish(bits, array.validity()?)
    }

    fn contains_rows(
        &self,
        elements: &ArrayRef,
        indices: &[usize],
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if indices.is_empty() {
            return self.set.finish(
                BitBuffer::full_in(false, needles.len(), ctx.allocator().clone()),
                needles.validity()?,
            );
        }
        let needles = needles
            .clone()
            .execute::<RecursiveCanonical>(ctx)?
            .0
            .into_array();
        // The comparator materializes child buffers and validity once, including decimal
        // widening and nested offsets. Each search then reads those buffers directly.
        let compare = build_row_comparator(elements, &needles, ctx)?;
        let bits = BitBuffer::collect_bool_in(
            needles.len(),
            |row| {
                indices
                    .binary_search_by(|&element| compare(element, row))
                    .is_ok()
            },
            ctx.allocator().clone(),
        );
        self.set.finish(bits, needles.validity()?)
    }
}

impl Probe {
    /// One bit per needle, set when the needle is an element.
    fn integer_bits<T: IntegerPType>(
        &self,
        needles: &[T],
        allocator: &BufferAllocatorRef,
    ) -> BitBuffer {
        match self {
            Self::Bitmap {
                min_offset,
                span,
                bitmap,
            } => collect_bits(
                needles,
                // A needle below the smallest element wraps past `span`, so one comparison checks
                // both bounds.
                |needle| {
                    let offset = needle.as_().wrapping_sub(*min_offset);
                    offset <= *span && bitmap.value(offset)
                },
                allocator,
            ),
            Self::Sorted(sorted) => {
                let sorted = sorted.as_slice::<T>();
                sorted_bits(sorted, needles, allocator)
            }
            Self::DecimalBitmap { .. }
            | Self::DecimalSorted(_)
            | Self::Bytes { .. }
            | Self::Rows { .. } => {
                unreachable!("integer needles meet an integer probe")
            }
        }
    }
}

/// A table of the elements' indices, hashed by their bytes, holding each distinct value once.
fn bytes_probe(elements: VarBinViewArray) -> Probe {
    let hasher = RandomState::default();
    let mut table = HashTable::with_capacity(elements.len());
    {
        let views = elements.views();
        let buffers = data_buffers(&elements);
        let bytes = |idx: u32| view_bytes(&views[idx as usize], &buffers);
        for (idx, view) in views.iter().enumerate() {
            let value = view_bytes(view, &buffers);
            if let HashTableEntry::Vacant(vacant) = table.entry(
                hasher.hash_one(value),
                |&other| bytes(other) == value,
                |&other| hasher.hash_one(bytes(other)),
            ) {
                vacant.insert(
                    u32::try_from(idx).vortex_expect("a list holds fewer than 2^32 elements"),
                );
            }
        }
    }
    Probe::Bytes {
        elements,
        hasher,
        table,
    }
}

/// The host slices of an array's data buffers, indexed by a view's buffer index.
fn data_buffers(array: &VarBinViewArray) -> Vec<&[u8]> {
    (0..array.data_buffers().len())
        .map(|idx| array.buffer(idx).as_slice())
        .collect()
}

/// The bytes a view points at, inlined in the view itself or out of line in one of `buffers`.
fn view_bytes<'a>(view: &'a BinaryView, buffers: &[&'a [u8]]) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let reference = view.as_view();
        &buffers[reference.buffer_index as usize][reference.as_range()]
    }
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

/// A bitmap over the elements' span when the span is dense, and a sorted slice otherwise.
///
/// A hash set and, for a handful of elements, a linear scan both lost to the binary search at every
/// set size measured by the `list_contains_set` benchmark, up to 16 384 elements.
fn integer_probe<T: IntegerPType + SetInteger>(
    elements: PrimitiveArray,
    allocator: &BufferAllocatorRef,
) -> Probe {
    // The primitive hot loop computes offsets in usize, which must hold every value of T.
    if size_of::<T>() <= size_of::<usize>()
        && let Some((min, bitmap)) = integer_bitmap(elements.as_slice::<T>(), allocator)
    {
        return Probe::Bitmap {
            min_offset: min.as_(),
            span: bitmap.len() - 1,
            bitmap,
        };
    }
    Probe::Sorted(PrimitiveArray::new(
        sorted_values(elements.into_buffer::<T>()),
        Validity::NonNullable,
    ))
}

/// An unsigned modular distance, rejecting offsets too wide to address a bitmap.
/// Keeping the subtraction at the physical width also handles signed ranges spanning zero.
trait SetInteger: Copy + Ord {
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

fn sorted_bits<T: Copy + Ord>(
    sorted: &[T],
    needles: &[T],
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    collect_bits(
        needles,
        |needle| sorted.binary_search(&needle).is_ok(),
        allocator,
    )
}

#[cfg(test)]
mod tests;
