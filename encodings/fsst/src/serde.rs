use fsst::{Compressor, Symbol};
use vortex_array::arrays::VarBinVTable;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::{FSSTArray, FSSTEncoding, FSSTVTable, fsst_compress, fsst_train_compressor};

#[derive(Clone, prost::Message)]
pub struct FSSTMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    uncompressed_lengths_ptype: i32,
}

impl SerdeVTable<FSSTVTable> for FSSTVTable {
    type Metadata = ProstMetadata<FSSTMetadata>;

    fn metadata(array: &FSSTArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(FSSTMetadata {
            uncompressed_lengths_ptype: PType::try_from(array.uncompressed_lengths().dtype())
                .vortex_expect("Must be a valid PType")
                as i32,
        })))
    }

    fn build(
        _encoding: &FSSTEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FSSTArray> {
        if buffers.len() != 2 {
            vortex_bail!(InvalidArgument: "Expected 2 buffers, got {}", buffers.len());
        }
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[0].clone());
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[1].clone());

        if children.len() != 2 {
            vortex_bail!(InvalidArgument: "Expected 2 children, got {}", children.len());
        }
        let codes = children.get(0, &DType::Binary(dtype.nullability()), len)?;
        let codes = codes
            .as_opt::<VarBinVTable>()
            .ok_or_else(|| {
                vortex_err!(
                    "Expected VarBinArray for codes, got {}",
                    codes.encoding_id()
                )
            })?
            .clone();
        let uncompressed_lengths = children.get(
            1,
            &DType::Primitive(
                metadata.uncompressed_lengths_ptype(),
                Nullability::NonNullable,
            ),
            len,
        )?;

        FSSTArray::try_new(
            dtype.clone(),
            symbols,
            symbol_lengths,
            codes,
            uncompressed_lengths,
        )
    }
}

impl EncodeVTable<FSSTVTable> for FSSTVTable {
    fn encode(
        _encoding: &FSSTEncoding,
        canonical: &Canonical,
        like: Option<&FSSTArray>,
    ) -> VortexResult<Option<FSSTArray>> {
        let array = canonical.clone().into_varbinview()?;

        let compressor = match like {
            Some(like) => Compressor::rebuild_from(like.symbols(), like.symbol_lengths()),
            None => fsst_train_compressor(array.as_ref())?,
        };

        Ok(Some(fsst_compress(array.as_ref(), &compressor)?))
    }
}

impl VisitorVTable<FSSTVTable> for FSSTVTable {
    fn visit_buffers(array: &FSSTArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&array.symbols().clone().into_byte_buffer());
        visitor.visit_buffer(&array.symbol_lengths().clone().into_byte_buffer());
    }

    fn visit_children(array: &FSSTArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", array.codes().as_ref());
        visitor.visit_child("uncompressed_lengths", array.uncompressed_lengths());
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::serde::FSSTMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fsst_metadata() {
        check_metadata(
            "fsst.metadata",
            ProstMetadata(FSSTMetadata {
                uncompressed_lengths_ptype: PType::U64 as i32,
            }),
        );
    }
}
