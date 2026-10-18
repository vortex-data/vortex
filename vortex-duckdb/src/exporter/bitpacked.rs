// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exports bit-packed integers by unpacking each exported range straight into the vector.

use std::mem::MaybeUninit;

use num_traits::WrappingAdd;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::match_each_integer_ptype;
use vortex::buffer::BufferMut;
use vortex::dtype::NativePType;
use vortex::dtype::PhysicalPType;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::encodings::fastlanes::BitWidthsView;
use vortex::encodings::fastlanes::FL_CHUNK_SIZE;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArrayExt;
use vortex::encodings::fastlanes::FoRArraySlotsExt;
use vortex::encodings::fastlanes::FoRReferences;
use vortex::encodings::fastlanes::unpack_iter::BitPacked as UnpackType;
use vortex::encodings::fastlanes::unpack_iter::BitPackingStrategy;
use vortex::encodings::fastlanes::unpack_iter::UnpackedChunks;
use vortex::error::VortexResult;

use crate::duckdb::VectorBuffer;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::validity;

/// Exports a bit-packed array, possibly under a frame of reference, without decoding it into a
/// buffer first.
///
/// Decoding a whole array allocates and writes a buffer as large as the array, which DuckDB then
/// reads a vector at a time. Unpacking each vector's rows into the vector itself touches only
/// memory that stays in cache.
struct BitPackedExporter<T: UnpackType> {
    /// Keeps the packed buffer alive.
    _array: ArrayRef,
    packed: *const T::Physical,
    packed_len: usize,
    bit_width: usize,
    /// The position of the array's first row within its first chunk.
    offset: usize,
    len: usize,
    reference: T,
}

// SAFETY: the exporter only reads the packed buffer, which `_array` keeps alive.
unsafe impl<T: UnpackType> Send for BitPackedExporter<T> {}

/// An exporter for `array` if it is bit-packed at one bit width without patches, possibly under a
/// frame of reference with one reference.
pub(crate) fn try_new_exporter(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn ColumnExporter>>> {
    let (packed_array, reference) = match array.as_opt::<FoR>() {
        Some(for_array) => {
            let FoRReferences::Global(reference) = for_array.references() else {
                return Ok(None);
            };
            (for_array.encoded().clone(), Some(reference.clone()))
        }
        None => (array.clone(), None),
    };
    let Some(bitpacked) = packed_array.as_opt::<BitPacked>() else {
        return Ok(None);
    };
    let BitWidthsView::Global(bit_width) = bitpacked.bit_widths() else {
        return Ok(None);
    };
    // A frame of reference may pack unsigned offsets of signed values, so the values take the
    // type of the exported array, which must have the width of the packed values.
    if bitpacked.patches().is_some()
        || !array.dtype().is_int()
        || array.dtype().as_ptype().byte_width() != bitpacked.dtype().as_ptype().byte_width()
    {
        return Ok(None);
    }
    let validity = array.validity()?;
    if validity.definitely_all_null() {
        return Ok(None);
    }

    let len = array.len();
    let offset = usize::from(bitpacked.offset());
    match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
        let reference = match reference {
            Some(reference) => match reference.as_primitive().typed_value::<T>() {
                Some(reference) => reference,
                None => return Ok(None),
            },
            None => T::default(),
        };
        let packed = bitpacked.packed_slice::<<T as PhysicalPType>::Physical>();
        let exporter = Box::new(BitPackedExporter::<T> {
            packed: packed.as_ptr(),
            packed_len: packed.len(),
            _array: packed_array.clone(),
            bit_width: usize::from(bit_width),
            offset,
            len,
            reference,
        });
        validity::new_exporter(validity, len, exporter, ctx).map(Some)
    })
}

impl<T: UnpackType + NativePType + WrappingAdd> ColumnExporter for BitPackedExporter<T> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        assert!(self.len >= offset + len);
        let start = self.offset + offset;
        let elems_per_chunk = FL_CHUNK_SIZE * self.bit_width / (8 * size_of::<T>());
        let first_chunk = start / FL_CHUNK_SIZE;
        // SAFETY: `packed` and `packed_len` describe the packed buffer, which `_array` keeps alive.
        let packed = unsafe { std::slice::from_raw_parts(self.packed, self.packed_len) };
        let num_chunks = (start % FL_CHUNK_SIZE + len).div_ceil(FL_CHUNK_SIZE);
        let packed = &packed[first_chunk * elems_per_chunk..][..num_chunks * elems_per_chunk];

        let mut scratch = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
        let mut chunks = UnpackedChunks::try_new_with_strategy(
            BitPackingStrategy,
            packed,
            self.bit_width,
            start % FL_CHUNK_SIZE,
            len,
            &mut scratch,
        )?;
        // The vector may still point at the memory of an array exported before, so the values
        // are unpacked into a buffer of their own that the vector then shares.
        let mut values = BufferMut::<T>::with_capacity(len);
        chunks.decode_into(&mut values.spare_capacity_mut()[..len]);
        // SAFETY: `decode_into` wrote all `len` values.
        unsafe { values.set_len(len) };
        if self.reference != T::default() {
            for value in values.iter_mut() {
                *value = value.wrapping_add(&self.reference);
            }
        }
        let values = values.freeze();
        let data = values.as_ptr();
        // SAFETY: the vector buffer keeps `values` alive for as long as the vector uses them.
        unsafe {
            vector.set_vector_buffer(&VectorBuffer::new(values));
            // While we are setting a *mut T this is an artifact of the C API, this is in fact const.
            vector.set_data_ptr(data.cast_mut());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::encodings::fastlanes::BitPackedData;
    use vortex::encodings::fastlanes::FoR;
    use vortex::encodings::fastlanes::FoRArraySlotsExt;
    use vortex::encodings::fastlanes::FoRData;
    use vortex::error::VortexResult;
    use vortex::scalar::Scalar;

    use super::try_new_exporter;
    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    fn exported(
        array: &vortex::array::ArrayRef,
        offset: usize,
        len: usize,
    ) -> VortexResult<Vec<i64>> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BIGINT)]);
        try_new_exporter(array, &mut ctx)?
            .expect("bit-packed arrays have an exporter")
            .export(offset, len, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(len);
        Ok(chunk.get_vector(0).as_slice_with_len::<i64>(len).to_vec())
    }

    #[rstest]
    #[case::aligned(0, 2048)]
    #[case::unaligned(1000, 2048)]
    #[case::tail(4000, 1000)]
    fn exports_bit_packed_ranges(#[case] offset: usize, #[case] len: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..5000i64).map(|i| (i * 7) % 1000));
        let packed = BitPackedData::encode(&values.into_array(), 10, &mut ctx)?.into_array();
        let expected: Vec<i64> = (offset as i64..(offset + len) as i64)
            .map(|i| (i * 7) % 1000)
            .collect();
        assert_eq!(exported(&packed, offset, len)?, expected);

        let sliced = packed.execute_slice(300..5000, &mut ctx)?;
        let expected: Vec<i64> = ((offset + 300) as i64
            ..(offset + 300 + len.min(4700 - offset)) as i64)
            .map(|i| (i * 7) % 1000)
            .collect();
        assert_eq!(exported(&sliced, offset, expected.len())?, expected);
        Ok(())
    }

    #[test]
    fn exports_frame_of_reference_of_signed_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..3000i64).map(|i| -500 + (i * 13) % 1000);
        let for_array = FoRData::encode(PrimitiveArray::from_iter(values.clone()), &mut ctx)?;
        let encoded = for_array.encoded().clone();
        let encoded = BitPackedData::encode(&encoded, 10, &mut ctx)?.into_array();
        let for_array = FoR::try_new(encoded, Scalar::from(-500i64))?.into_array();
        let expected: Vec<i64> = values.skip(700).take(2048).collect();
        assert_eq!(exported(&for_array, 700, 2048)?, expected);
        Ok(())
    }

    #[test]
    fn exports_frame_of_reference() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = PrimitiveArray::from_iter((0..3000i64).map(|i| i % 500)).into_array();
        let encoded = BitPackedData::encode(&encoded, 9, &mut ctx)?.into_array();
        let for_array = FoR::try_new(encoded, Scalar::from(1_000_000i64))?.into_array();
        let expected: Vec<i64> = (100..2148i64).map(|i| 1_000_000 + i % 500).collect();
        assert_eq!(exported(&for_array, 100, 2048)?, expected);
        Ok(())
    }
}
