use arrow_array::builder::make_view;
use arrow_buffer::Buffer;
use vortex_array::array::{PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::{
    ArrayDType, ArrayData, Canonical, IntoArrayData, IntoArrayVariant, IntoCanonical,
};
use vortex_error::VortexResult;

use crate::FSSTArray;

impl IntoCanonical for FSSTArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        self.with_decompressor(|decompressor| {
            // FSSTArray has two child arrays:
            //
            //  1. A VarBinArray, which holds the string heap of the compressed codes.
            //  2. An uncompressed_lengths primitive array, storing the length of each original
            //     string element.
            //
            // To speed up canonicalization, we can decompress the entire string-heap in a single
            // call. We then turn our uncompressed_lengths into an offsets buffer
            // necessary for a VarBinViewArray and construct the canonical array.

            let compressed_bytes = VarBinArray::try_from(self.codes())?
                .sliced_bytes()?
                .into_primitive()?;

            // Bulk-decompress the entire array.
            let uncompressed_bytes =
                decompressor.decompress(compressed_bytes.maybe_null_slice::<u8>());

            let uncompressed_lens_array = self
                .uncompressed_lengths()
                .into_canonical()?
                .into_primitive()?;
            let uncompressed_lens_slice = uncompressed_lens_array.maybe_null_slice::<i32>();

            // Directly create the binary views.
            let views: Vec<u128> = uncompressed_lens_slice
                .iter()
                .scan(0, |offset, len| {
                    let str_start = *offset;
                    let str_end = *offset + len;

                    *offset += len;

                    Some(make_view(
                        &uncompressed_bytes[(str_start as usize)..(str_end as usize)],
                        0u32,
                        str_start as u32,
                    ))
                })
                .collect();

            let views_array: ArrayData = Buffer::from(views).into();
            let uncompressed_bytes_array = PrimitiveArray::from(uncompressed_bytes).into_array();

            VarBinViewArray::try_new(
                views_array,
                vec![uncompressed_bytes_array],
                self.dtype().clone(),
                self.validity(),
            )
            .map(Canonical::VarBinView)
        })
    }
}
