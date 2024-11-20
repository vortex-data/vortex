use croaring::Bitmap;
use vortex_array::array::BoolArray;
use vortex_array::ArrayLen;
use vortex_error::VortexResult;

use crate::RoaringBoolArray;

pub fn roaring_bool_encode(bool_array: BoolArray) -> VortexResult<RoaringBoolArray> {
    let mut bitmap = Bitmap::new();
    bitmap.extend(bool_array.boolean_buffer().set_indices().map(|i| i as u32));
    bitmap.run_optimize();
    bitmap.shrink_to_fit();

    RoaringBoolArray::try_new(bitmap, bool_array.len())
}
