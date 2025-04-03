use fsst::{Compressor, Symbol};
use serde::{Deserialize, Serialize};
use vortex_array::arrays::VarBinArray;
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef,
    ArrayVisitorImpl, Canonical, DeserializeMetadata, Encoding, EncodingId, SerdeMetadata,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::{FSSTArray, FSSTEncoding, fsst_compress, fsst_train_compressor};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FSSTMetadata {
    uncompressed_lengths_ptype: PType,
}

impl EncodingVTable for FSSTEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.fsst")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = SerdeMetadata::<FSSTMetadata>::deserialize(parts.metadata())?;

        if parts.nbuffers() != 2 {
            vortex_bail!(InvalidArgument: "Expected 2 buffers, got {}", parts.nbuffers());
        }
        let symbols = Buffer::<Symbol>::from_byte_buffer(parts.buffer(0)?);
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(parts.buffer(1)?);

        if parts.nchildren() != 2 {
            vortex_bail!(InvalidArgument: "Expected 2 children, got {}", parts.nchildren());
        }
        let codes = parts
            .child(0)
            .decode(ctx, DType::Binary(dtype.nullability()), len)?
            .as_opt::<VarBinArray>()
            .ok_or_else(|| {
                vortex_err!(
                    "Expected VarBinArray for codes, got {:?}",
                    ctx.lookup_encoding(parts.child(0).encoding_id())
                )
            })?
            .clone();
        let uncompressed_lengths = parts.child(1).decode(
            ctx,
            DType::Primitive(
                metadata.uncompressed_lengths_ptype,
                Nullability::NonNullable,
            ),
            len,
        )?;

        Ok(
            FSSTArray::try_new(dtype, symbols, symbol_lengths, codes, uncompressed_lengths)?
                .into_array(),
        )
    }

    fn encode(
        &self,
        input: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let like = like
            .map(|like| {
                like.as_opt::<<Self as Encoding>::Array>().ok_or_else(|| {
                    vortex_err!(
                        "Expected {} encoded array but got {}",
                        self.id(),
                        like.encoding()
                    )
                })
            })
            .transpose()?;

        let array = input.clone().into_varbinview()?;

        let compressor = match like {
            Some(like) => Compressor::rebuild_from(like.symbols(), like.symbol_lengths()),
            None => fsst_train_compressor(&array)?,
        };

        let fsst = fsst_compress(&array, &compressor)?;

        Ok(Some(fsst.into_array()))
    }
}

impl ArrayVisitorImpl<SerdeMetadata<FSSTMetadata>> for FSSTArray {
    fn _visit_buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&self.symbols().clone().into_byte_buffer());
        visitor.visit_buffer(&self.symbol_lengths().clone().into_byte_buffer());
    }

    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", self.codes());
        visitor.visit_child("uncompressed_lengths", self.uncompressed_lengths());
    }

    fn _metadata(&self) -> SerdeMetadata<FSSTMetadata> {
        SerdeMetadata(FSSTMetadata {
            uncompressed_lengths_ptype: PType::try_from(self.uncompressed_lengths().dtype())
                .vortex_expect("Must be a valid PType"),
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_array::SerdeMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::serde::FSSTMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fsst_metadata() {
        check_metadata(
            "fsst.metadata",
            SerdeMetadata(FSSTMetadata {
                uncompressed_lengths_ptype: PType::U64,
            }),
        );
    }
}
