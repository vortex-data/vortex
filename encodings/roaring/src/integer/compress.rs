use croaring::Bitmap;
use num_traits::NumCast;
use vortex_array::array::PrimitiveArray;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_dtype::{NativePType, PType};
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::RoaringIntArray;

pub fn roaring_int_encode(parray: PrimitiveArray) -> VortexResult<RoaringIntArray> {
    match parray.ptype() {
        PType::U8 => roaring_encode_primitive::<u8>(parray.maybe_null_slice()),
        PType::U16 => roaring_encode_primitive::<u16>(parray.maybe_null_slice()),
        PType::U32 => roaring_encode_primitive::<u32>(parray.maybe_null_slice()),
        PType::U64 => roaring_encode_primitive::<u64>(parray.maybe_null_slice()),
        _ => vortex_bail!("Unsupported PType {}", parray.ptype()),
    }
}

fn roaring_encode_primitive<T: NumCast + NativePType>(
    values: &[T],
) -> VortexResult<RoaringIntArray> {
    let mut bitmap = Bitmap::new();
    bitmap.extend(
        values
            .iter()
            .map(|i| {
                i.to_u32()
                    .ok_or_else(|| vortex_err!("Failed to cast value {} to u32", i))
            })
            .collect::<VortexResult<Vec<u32>>>()?,
    );
    bitmap.run_optimize();
    bitmap.shrink_to_fit();
    RoaringIntArray::try_new(bitmap, T::PTYPE)
}
