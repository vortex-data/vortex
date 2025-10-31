// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use flatbuffers::root;
use vortex_alp::{ALPEncoding, ALPFloat, ALPVTable, match_each_alp_float_ptype};
use vortex_array::flatbuffers::ArrayNode;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{ArrayContext, DeserializeMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, NativePType};
use vortex_error::VortexExpect;
use vortex_fastlanes::{BitPackedEncoding, BitPackedVTable, FoREncoding, FoRVTable};
use vortex_flatbuffers::array::Array;
use vortex_scalar::Scalar;

use crate::CudaByteBuffer;
use crate::jit::{AlpEncodingTree, BitPackedEncodingTree, EncodingTreeRef, FoREncodingTree};

pub struct GpuArrayParts<'a> {
    buffers: Vec<Option<CudaByteBuffer>>,
    ctx: ArrayContext,
    array: Array<'a>,
}

impl<'a> GpuArrayParts<'a> {
    pub fn new(
        node_bytes: &'a ByteBuffer,
        buffer_slice: CudaByteBuffer,
        ctx: ArrayContext,
    ) -> Self {
        let array = root::<Array>(node_bytes).vortex_expect("flatbuffer root");

        let mut offset = 0;

        let stream = buffer_slice.stream();

        let buffers: Vec<Option<CudaByteBuffer>> = array
            .buffers()
            .unwrap_or_default()
            .iter()
            .map(|fb_buffer| {
                // Skip padding
                offset += fb_buffer.padding() as usize;

                let buffer_len = fb_buffer.length() as usize;

                // Extract a buffer and ensure it's aligned, copying if necessary
                let view = buffer_slice.slice(offset..(offset + buffer_len));

                let mut buffer = unsafe { stream.alloc(view.len()) }
                    .ok()
                    .vortex_expect("alloc");
                stream
                    .memcpy_dtod(&view, &mut buffer)
                    .ok()
                    .vortex_expect("memcpy");

                offset += buffer_len;
                Some(buffer)
            })
            .collect();

        GpuArrayParts {
            buffers,
            ctx: ctx.clone(),
            array,
        }
    }

    pub fn create_array(&mut self, dtype: &DType, len: usize) -> EncodingTreeRef {
        let fb_array = self.array;
        let array_node = fb_array.root().vortex_expect("must have root");

        self.create_array2(array_node, dtype, len)
    }

    fn create_array2(
        &mut self,
        array_node: ArrayNode,
        dtype: &DType,
        len: usize,
    ) -> EncodingTreeRef {
        let enc = self
            .ctx
            .lookup_encoding(array_node.encoding())
            .vortex_expect("ctx not found");
        if enc.id() == FoREncoding.id() {
            let deser =
                <<FoRVTable as SerdeVTable<FoRVTable>>::Metadata as DeserializeMetadata>::deserialize(
                    array_node.metadata().vortex_expect("md").bytes(),
                )
                    .vortex_expect("deser");
            let child = self.create_array2(
                array_node.children().vortex_expect("for has child").get(0),
                dtype,
                len,
            );
            let reference = Scalar::new(dtype.clone(), deser);
            return Arc::new(FoREncodingTree { reference, child }) as EncodingTreeRef;
        } else if enc.id() == BitPackedEncoding.id() {
            assert!(array_node.children().unwrap_or_default().is_empty());
            let deser =
                <<BitPackedVTable as SerdeVTable<BitPackedVTable>>::Metadata as DeserializeMetadata>::deserialize(
                    array_node.metadata().vortex_expect("md exists").bytes(),
                )
                    .vortex_expect("deser");
            let ptype = dtype.as_ptype();

            let buffer_handle = self.buffers[array_node
                .buffers()
                .vortex_expect("bitpacking has a buffer")
                .get(0) as usize]
                .take()
                .vortex_expect("missing buffer");
            return Arc::new(BitPackedEncodingTree {
                bit_width: u8::try_from(deser.bit_width).vortex_expect("bit width not u8"),
                output_type: ptype,
                buffer_handle,
            });
        } else if enc.id() == ALPEncoding.id() {
            let deser =
                <<ALPVTable as SerdeVTable<ALPVTable>>::Metadata as DeserializeMetadata>::deserialize(
                    array_node.metadata().vortex_expect("md alp").bytes(),
                )
                    .vortex_expect("deser");
            let ptype = dtype.as_ptype();

            let child_node = array_node.children().vortex_expect("for has child").get(0);

            let child = match_each_alp_float_ptype!(ptype, |P| {
                self.create_array2(
                    child_node,
                    &DType::Primitive(
                        <<P as ALPFloat>::ALPInt as NativePType>::PTYPE,
                        dtype.nullability(),
                    ),
                    len,
                )
            });
            return Arc::new(AlpEncodingTree {
                float_type: dtype.as_ptype(),
                child,
                f: deser.exp_f,
                e: deser.exp_e,
            });
        }

        todo!()
    }
}
