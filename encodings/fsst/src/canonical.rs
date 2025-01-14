use arrow_array::builder::make_view;
use vortex_array::array::{BinaryView, VarBinArray, VarBinViewArray};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayLen, Canonical, IntoCanonical};
use vortex_buffer::{BufferMut, ByteBuffer};
use vortex_dtype::match_each_integer_ptype;
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

            let bytes = VarBinArray::try_from(self.codes())?.sliced_bytes();

            // Bulk-decompress the entire array.
            let uncompressed_bytes = decompressor.decompress(bytes.as_slice());

            let uncompressed_lens_array = self
                .uncompressed_lengths()
                .into_canonical()?
                .into_primitive()?;

            // Directly create the binary views.
            let mut views = BufferMut::<BinaryView>::with_capacity(uncompressed_lens_array.len());

            match_each_integer_ptype!(uncompressed_lens_array.ptype(), |$P| {
                let mut offset = 0;
                for len in uncompressed_lens_array.as_slice::<$P>() {
                    let len = *len as usize;
                    let view = make_view(
                        &uncompressed_bytes[offset..][..len],
                        0u32,
                        offset as u32,
                    );
                    // SAFETY: we reserved the right capacity beforehand
                    unsafe { views.push_unchecked(view.into()) };
                    offset += len;
                }
            });

            let views = views.freeze();
            let uncompressed_bytes_array = ByteBuffer::from(uncompressed_bytes);

            VarBinViewArray::try_new(
                views,
                vec![uncompressed_bytes_array],
                self.dtype().clone(),
                self.validity(),
            )
            .map(Canonical::VarBinView)
        })
    }
}
