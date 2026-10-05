// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::mem;
use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use fastlanes::FoR;
use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::builders::UninitRange;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::FoRArray;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;
use crate::unpack_iter::for_each_packed_chunk;

pub fn decompress(array: &FoRArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    match array.constant_reference() {
        Some(reference) => decompress_one_ref(array, &reference, ctx),
        None => decompress_many_refs(array, ctx),
    }
}

/// Decompress an array whose chunks all share `reference`.
fn decompress_one_ref(
    array: &FoRArray,
    reference: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    // Try to do fused unpack.
    if let Some(bp) = array.encoded().as_opt::<BitPacked>() {
        return fused_decompress(array, bp, ctx);
    }

    add_reference(array, reference, ctx)
}

/// Unpack a BitPacked child and add the constant reference in one pass.
pub(crate) fn fused_decompress(
    for_: &FoRArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(for_.ptype(), |T| {
        fused_decompress_typed::<T>(for_, bp, ctx)
    })
}

fn fused_decompress_typed<
    T: PhysicalPType<Physical: FoR + BitPacking> + AsPrimitive<T::Physical> + WrappingAdd,
>(
    for_: &FoRArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let ref_ = for_
        .constant_reference()
        .ok_or_else(|| vortex_err!("fused FoR decompression requires a constant reference"))?
        .as_primitive()
        .as_::<T>()
        .vortex_expect("cannot be null");

    fused_unpack(for_, bp, |_| ref_, ctx)
}

/// Decode `encoded`, then add `reference` to every value.
fn add_reference(
    array: &FoRArray,
    reference: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(array.ptype(), |T| {
        add_reference_typed::<T>(array, reference, ctx)
    })
}

fn add_reference_typed<T: NativePType + WrappingAdd + PrimInt>(
    array: &FoRArray,
    reference: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let encoded = array.encoded().as_::<Primitive>().into_owned();
    let min = reference
        .as_primitive()
        .typed_value::<T>()
        .vortex_expect("reference must be non-null");
    if min == T::zero() {
        return Ok(encoded);
    }
    let validity = encoded.validity()?;
    Ok(PrimitiveArray::new(
        decompress_primitive(encoded.try_into_buffer_mut::<T>(), min, ctx),
        validity,
    ))
}

/// Adds `min` to every value. A uniquely owned buffer is mapped in place, a shared one is mapped
/// into a new allocation.
fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: Result<BufferMut<T>, Buffer<T>>,
    min: T,
    ctx: &mut ExecutionCtx,
) -> Buffer<T> {
    let add = |v: T| v.wrapping_add(&min);

    match values {
        Ok(mut values) => {
            values.as_mut_slice().map_into_in_place(add);
            values.freeze()
        }
        Err(values) => {
            let len = values.len();
            let mut decoded = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
            values
                .as_slice()
                .map_into(&mut decoded.spare_capacity_mut()[..len], add);

            // SAFETY: `map_into` writes every lane of the `len` items.
            unsafe { decoded.set_len(len) };

            decoded.freeze()
        }
    }
}

/// Decompress an array whose chunks have different references.
fn decompress_many_refs(array: &FoRArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    // Try to do fused unpack. BitPacked chunks line up with FoR chunks when the offsets match.
    if let Some(bp) = array.encoded().as_opt::<BitPacked>()
        && bp.offset() == array.offset()
    {
        return fused_decompress_many_refs(array, bp, ctx);
    }

    add_references(array, ctx)
}

/// Decode `encoded`, then add each chunk's reference in place.
fn add_references(array: &FoRArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(array.ptype(), |T| { add_references_typed::<T>(array, ctx) })
}

fn add_references_typed<T: NativePType + WrappingAdd + PrimInt>(
    array: &FoRArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let encoded = array.encoded().as_::<Primitive>().into_owned();
    if encoded.is_empty() {
        return Ok(encoded);
    }
    let validity = encoded.validity()?;
    let references = array.references().as_::<Primitive>().into_owned();
    let references = references.as_slice::<T>();

    // The first chunk may be partial when the array was sliced.
    let first_len = (FL_CHUNK_SIZE - usize::from(array.offset())).min(array.len());
    let values = match encoded.into_buffer::<T>().try_into_mut() {
        // Try to add references in place if we hold only strong reference.
        Ok(mut values) => {
            add_references_in_place(&mut values, first_len, references);
            values
        }
        // Otherwise add the references and encoded values in a new buffer.
        Err(encoded) => {
            add_references_copied(&encoded, first_len, references, ctx.allocator().clone())
        }
    };
    Ok(PrimitiveArray::new(values.freeze(), validity))
}

/// Add each chunk's reference to `values`, whose first chunk holds `first_len` values.
fn add_references_in_place<T: NativePType + WrappingAdd>(
    values: &mut [T],
    first_len: usize,
    references: &[T],
) {
    for (chunk, &reference) in chunks_mut(values, first_len).zip_eq(references) {
        for value in chunk {
            *value = value.wrapping_add(&reference);
        }
    }
}

/// Copy `encoded`, whose first chunk holds `first_len` values, into a new buffer, adding each
/// chunk's reference on the way.
fn add_references_copied<T: NativePType + WrappingAdd>(
    encoded: &[T],
    first_len: usize,
    references: &[T],
    allocator: BufferAllocatorRef,
) -> BufferMut<T> {
    let len = encoded.len();
    let mut values = BufferMut::<T>::with_capacity_in(len, allocator);
    let (first, rest) = encoded.split_at(first_len);
    let inputs = iter::once(first).chain(rest.chunks(FL_CHUNK_SIZE));
    let outputs = chunks_mut(&mut values.spare_capacity_mut()[..len], first_len);
    for ((output, input), &reference) in outputs.zip(inputs).zip_eq(references) {
        for (output, value) in output.iter_mut().zip(input) {
            output.write(value.wrapping_add(&reference));
        }
    }
    // SAFETY: the loop above initialized every value.
    unsafe { values.set_len(len) };
    values
}

/// Split `values` into FoR chunks, the first of which holds `first_len` values.
fn chunks_mut<V>(values: &mut [V], first_len: usize) -> impl Iterator<Item = &mut [V]> {
    let (first, rest) = values.split_at_mut(first_len);
    iter::once(first).chain(rest.chunks_mut(FL_CHUNK_SIZE))
}

/// Unpack each BitPacked chunk and add its chunk's reference in one pass.
fn fused_decompress_many_refs(
    for_: &FoRArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(for_.ptype(), |T| {
        fused_decompress_many_refs_typed::<T>(for_, bp, ctx)
    })
}

fn fused_decompress_many_refs_typed<
    T: PhysicalPType<Physical: FoR + BitPacking> + AsPrimitive<T::Physical> + WrappingAdd,
>(
    for_: &FoRArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let references = for_.references().as_::<Primitive>().into_owned();
    let references = references.as_slice::<T>();
    fused_unpack(for_, bp, |chunk| references[chunk], ctx)
}

/// Unpack each BitPacked chunk and add its reference in one pass.
///
/// `chunk_reference` maps the index of a chunk, counted from the first chunk of `bp`, to its
/// reference.
fn fused_unpack<
    T: PhysicalPType<Physical: FoR + BitPacking> + AsPrimitive<T::Physical> + WrappingAdd,
>(
    for_: &FoRArray,
    bp: ArrayView<'_, BitPacked>,
    chunk_reference: impl Fn(usize) -> T,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let len = bp.len();
    let mut builder =
        PrimitiveBuilder::<T>::with_capacity_in(for_.dtype().nullability(), len, ctx.allocator());
    let mut uninit_range = builder.uninit_range(len);
    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(&bp.validity()?.execute_mask(len, ctx)?);
    }

    // SAFETY: `unpack_chunks` initializes every value in this range.
    let output = unsafe { uninit_range.slice_uninit_mut(0, len) };
    unpack_chunks(bp, &chunk_reference, output)?;

    if let Some(patches) = bp.patches() {
        let offset = usize::from(bp.offset());
        apply_patches(&mut uninit_range, &patches, offset, &chunk_reference, ctx)?;
    }

    // SAFETY: We have set a correct validity mask via `append_mask` with `len` values and
    // initialized every value in `unpack_chunks`.
    unsafe {
        uninit_range.finish();
    }

    Ok(builder.finish_into_primitive())
}

/// Unpack each chunk of `bp` into `output` and add the chunk's reference.
///
/// Full chunks unpack straight into `output`. A partial first or last chunk unpacks into a scratch
/// chunk, and only its values in `output` are copied over.
fn unpack_chunks<
    T: PhysicalPType<Physical: FoR + BitPacking> + AsPrimitive<T::Physical> + WrappingAdd,
>(
    bp: ArrayView<'_, BitPacked>,
    chunk_reference: impl Fn(usize) -> T,
    output: &mut [MaybeUninit<T>],
) -> VortexResult<()> {
    let offset = usize::from(bp.offset());
    let bit_width = bp.bit_width() as usize;
    // SAFETY: `T::Physical` is `T` with the same size and alignment, and the unpack is the same
    // wrapping addition in two's complement whichever signedness `T` has.
    let output =
        unsafe { mem::transmute::<&mut [MaybeUninit<T>], &mut [MaybeUninit<T::Physical>]>(output) };
    let mut scratch = [const { MaybeUninit::<T::Physical>::uninit() }; FL_CHUNK_SIZE];
    for_each_packed_chunk::<T, _>(
        bp.packed_slice::<T::Physical>(),
        bit_width,
        offset,
        bp.len(),
        |packed, range| {
            let reference = chunk_reference(range.start / FL_CHUNK_SIZE).as_();
            // `range` counts from the start of the first chunk, and the output starts at `offset`.
            let skip = offset.saturating_sub(range.start);
            let dst = &mut output[range.start + skip - offset..range.end - offset];
            if dst.len() == FL_CHUNK_SIZE {
                // SAFETY: `packed` holds one chunk at `bit_width` and `dst` has room for a chunk.
                unsafe { unfor_pack_into(bit_width, packed, reference, dst) };
            } else {
                // SAFETY: as above, with `scratch` as the destination.
                unsafe { unfor_pack_into(bit_width, packed, reference, &mut scratch) };
                dst.copy_from_slice(&scratch[skip..range.len()]);
            }
        },
    )
}

/// Unpack one chunk into `dst` and add `reference` to every value.
///
/// # Safety
///
/// `packed` must hold one chunk at `bit_width`, and `dst` must have room for a full chunk.
#[inline]
unsafe fn unfor_pack_into<T: FoR>(
    bit_width: usize,
    packed: &[T],
    reference: T,
    dst: &mut [MaybeUninit<T>],
) {
    // SAFETY: the caller guarantees the sizes, and the unpack initializes every value of `dst`.
    unsafe {
        T::unchecked_unfor_pack(
            bit_width,
            packed,
            reference,
            mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(dst),
        );
    }
}

/// Write each patch value plus the reference of the chunk it falls in.
///
/// `offset` is the position of the first value within the first chunk, as in `fused_unpack`.
fn apply_patches<T: NativePType + WrappingAdd>(
    uninit_range: &mut UninitRange<T>,
    patches: &Patches,
    offset: usize,
    chunk_reference: impl Fn(usize) -> T,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    assert_eq!(patches.array_len(), uninit_range.len());

    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;

    assert!(values.all_valid(ctx)?, "Patch values must be all valid");

    let values = values.as_slice::<T>();
    match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
        for (&index, &value) in indices.as_slice::<P>().iter().zip_eq(values) {
            let index = <P as AsPrimitive<usize>>::as_(index) - patches.offset();
            let reference = chunk_reference((offset + index) / FL_CHUNK_SIZE);
            uninit_range.set_value(index, value.wrapping_add(&reference));
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::assert_arrays_eq;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedData;
    use crate::FoR;
    use crate::FoRData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_decompress() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a range offset by a million.
        let array = PrimitiveArray::from_iter((0u32..100_000).step_by(1024).map(|v| v + 1_000_000));
        let compressed = FoRData::encode(array.clone(), &mut ctx).unwrap();
        assert_arrays_eq!(compressed, array, &mut ctx);
    }

    #[test]
    fn test_decompress_fused() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a range offset by a million.
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedData::encode(&array.into_array(), 3, &mut ctx).unwrap();
        let compressed = FoR::try_new(bp.into_array(), 10u32.into()).unwrap();
        assert_arrays_eq!(compressed, expect, &mut ctx);
    }

    #[test]
    fn test_decompress_fused_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a range offset by a million.
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedData::encode(&array.into_array(), 2, &mut ctx)?;
        let compressed = FoR::try_new(bp.clone().into_array(), 10u32.into())?;
        let decompressed = fused_decompress(&compressed, bp.as_view(), &mut ctx)?;
        assert_arrays_eq!(decompressed, expect, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_decompress_fused_signed_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let expect = PrimitiveArray::from_iter((0i64..1024).map(|x| x % 7 - 1_000_000));
        let array = PrimitiveArray::from_iter((0i64..1024).map(|x| x % 7));
        let bp = BitPackedData::encode(&array.into_array(), 2, &mut ctx)?;
        let compressed = FoR::try_new(bp.clone().into_array(), (-1_000_000i64).into())?;
        let decompressed = fused_decompress(&compressed, bp.as_view(), &mut ctx)?;
        assert_arrays_eq!(decompressed, expect, &mut ctx);
        Ok(())
    }
}
