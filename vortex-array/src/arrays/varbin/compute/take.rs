// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::varbin::VarBinArrayExt;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

/// The widened offset type used for a taken `VarBinArray`: offsets are widened to at least 32 bits
/// (to avoid overflow) while preserving signedness, so a signed result stays Arrow-compatible.
fn taken_offset_ptype(offsets_ptype: PType) -> PType {
    match offsets_ptype {
        PType::U8 | PType::U16 | PType::U32 => PType::U32,
        PType::U64 => PType::U64,
        PType::I8 | PType::I16 | PType::I32 => PType::I32,
        PType::I64 => PType::I64,
        _ => unreachable!("invalid PType for offsets"),
    }
}

impl TakeExecute for VarBin {
    fn take(
        array: ArrayView<'_, VarBin>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(joe): Be lazy with execute
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let data = array.bytes();
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let dtype = array
            .dtype()
            .clone()
            .union_nullability(indices.dtype().nullability());
        let array_validity = array
            .varbin_validity()
            .execute_mask(array.as_ref().len(), ctx)?;
        let indices_validity = indices
            .as_ref()
            .validity()?
            .execute_mask(indices.as_ref().len(), ctx)?;

        // Offsets and indices are non-negative; read them through their unsigned reinterpretations
        // so we only monomorphize over the 4 unsigned widths each (4x4 instead of 8x8). On take,
        // offsets get widened to either 32- or 64-bit (to avoid overflow); the built output offsets
        // are reinterpreted back to `out_offset_ptype` to preserve the result's offset signedness.
        let out_offset_ptype = taken_offset_ptype(offsets.ptype());
        let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
        let indices = indices.reinterpret_cast(indices.ptype().to_unsigned());

        let array = match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            match offsets.ptype() {
                PType::U8 => take::<I, u8, u32>(
                    dtype,
                    offsets.as_slice::<u8>(),
                    data.as_slice(),
                    indices.as_slice::<I>(),
                    array_validity,
                    indices_validity,
                    out_offset_ptype,
                ),
                PType::U16 => take::<I, u16, u32>(
                    dtype,
                    offsets.as_slice::<u16>(),
                    data.as_slice(),
                    indices.as_slice::<I>(),
                    array_validity,
                    indices_validity,
                    out_offset_ptype,
                ),
                PType::U32 => take::<I, u32, u32>(
                    dtype,
                    offsets.as_slice::<u32>(),
                    data.as_slice(),
                    indices.as_slice::<I>(),
                    array_validity,
                    indices_validity,
                    out_offset_ptype,
                ),
                PType::U64 => take::<I, u64, u64>(
                    dtype,
                    offsets.as_slice::<u64>(),
                    data.as_slice(),
                    indices.as_slice::<I>(),
                    array_validity,
                    indices_validity,
                    out_offset_ptype,
                ),
                _ => unreachable!("invalid PType for offsets"),
            }
        });

        Ok(Some(array?.into_array()))
    }
}

fn take<Index: IntegerPType, Offset: IntegerPType, NewOffset: IntegerPType>(
    dtype: DType,
    offsets: &[Offset],
    data: &[u8],
    indices: &[Index],
    validity_mask: Mask,
    indices_validity_mask: Mask,
    out_offset_ptype: PType,
) -> VortexResult<VarBinArray> {
    if !validity_mask.all_true() || !indices_validity_mask.all_true() {
        return Ok(take_nullable::<Index, Offset, NewOffset>(
            dtype,
            offsets,
            data,
            indices,
            validity_mask,
            indices_validity_mask,
            out_offset_ptype,
        ));
    }

    let mut new_offsets = BufferMut::<NewOffset>::with_capacity(indices.len() + 1);
    new_offsets.push(NewOffset::zero());
    let mut current_offset = NewOffset::zero();

    for &idx in indices {
        let idx = idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", idx));
        let start = offsets[idx];
        let stop = offsets[idx + 1];

        current_offset += NewOffset::from(stop - start).vortex_expect("offset type overflow");
        new_offsets.push(current_offset);
    }

    let mut new_data = ByteBufferMut::with_capacity(current_offset.as_());

    for idx in indices {
        let idx = idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", idx));
        let start = offsets[idx]
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        let stop = offsets[idx + 1]
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        new_data.extend_from_slice(&data[start..stop]);
    }

    let array_validity = Validity::from(dtype.nullability());

    // Built unsigned; reinterpret back to the signedness-preserving result offset type.
    let new_offsets = PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable)
        .reinterpret_cast(out_offset_ptype)
        .into_array();

    // Safety:
    // All variants of VarBinArray are satisfied here.
    unsafe {
        Ok(VarBinArray::new_unchecked(
            new_offsets,
            new_data.freeze(),
            dtype,
            array_validity,
        ))
    }
}

fn take_nullable<Index: IntegerPType, Offset: IntegerPType, NewOffset: IntegerPType>(
    dtype: DType,
    offsets: &[Offset],
    data: &[u8],
    indices: &[Index],
    data_validity: Mask,
    indices_validity: Mask,
    out_offset_ptype: PType,
) -> VarBinArray {
    let mut new_offsets = BufferMut::<NewOffset>::with_capacity(indices.len() + 1);
    new_offsets.push(NewOffset::zero());
    let mut current_offset = NewOffset::zero();

    let mut validity_buffer = BitBufferMut::with_capacity(indices.len());

    // Convert indices once and store valid ones with their positions
    let mut valid_indices = Vec::with_capacity(indices.len());

    // First pass: calculate offsets and validity
    for (idx, data_idx) in indices.iter().enumerate() {
        if !indices_validity.value(idx) {
            validity_buffer.append(false);
            new_offsets.push(current_offset);
            continue;
        }
        let data_idx_usize = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));
        if data_validity.value(data_idx_usize) {
            validity_buffer.append(true);
            let start = offsets[data_idx_usize];
            let stop = offsets[data_idx_usize + 1];
            current_offset += NewOffset::from(stop - start).vortex_expect("offset type overflow");
            new_offsets.push(current_offset);
            valid_indices.push(data_idx_usize);
        } else {
            validity_buffer.append(false);
            new_offsets.push(current_offset);
        }
    }

    let mut new_data = ByteBufferMut::with_capacity(current_offset.as_());

    // Second pass: copy data for valid indices only
    for data_idx in valid_indices {
        let start = offsets[data_idx]
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        let stop = offsets[data_idx + 1]
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        new_data.extend_from_slice(&data[start..stop]);
    }

    let array_validity = Validity::from(validity_buffer.freeze());

    // Built unsigned; reinterpret back to the signedness-preserving result offset type.
    let new_offsets = PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable)
        .reinterpret_cast(out_offset_ptype)
        .into_array();

    // Safety:
    // All variants of VarBinArray are satisfied here.
    unsafe { VarBinArray::new_unchecked(new_offsets, new_data.freeze(), dtype, array_validity) }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbin::compute::take::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    #[test]
    fn test_null_take() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            arr.take(idx1.into_array()).unwrap().dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            arr.take(idx2.into_array()).unwrap().dtype(),
            &DType::Utf8(Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(
        [Some("hello"), None, Some("test"), Some("data"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case(VarBinArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(["single"].map(Some), DType::Utf8(Nullability::NonNullable)))]
    fn test_take_varbin_conformance(#[case] array: VarBinArray) {
        test_take_conformance(&array.into_array());
    }

    #[test]
    fn test_take_overflow() {
        let scream = std::iter::once("a").cycle().take(128).collect::<String>();
        let bytes = ByteBuffer::copy_from(scream.as_bytes());
        let offsets = buffer![0u8, 128u8].into_array();

        let array = VarBinArray::new(
            offsets,
            bytes,
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        );

        let indices = buffer![0u32; 3].into_array();
        let taken = array.take(indices).unwrap();

        let expected = VarBinViewArray::from_iter(
            [Some(scream.clone()), Some(scream.clone()), Some(scream)],
            DType::Utf8(Nullability::NonNullable),
        );
        assert_arrays_eq!(expected, taken);
    }
}
