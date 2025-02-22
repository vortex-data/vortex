use serde::{Deserialize, Serialize};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, DeserializeMetadata,
    SerdeMetadata,
};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult};

use crate::array::{SYMBOLS_DTYPE, SYMBOL_LENS_DTYPE};
use crate::{FSSTArray, FSSTEncoding};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FSSTMetadata {
    symbols_len: usize,
    codes_nullability: Nullability,
    uncompressed_lengths_ptype: PType,
}

impl ArrayVisitorImpl<SerdeMetadata<FSSTMetadata>> for FSSTArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("symbols", self.symbols());
        visitor.visit_child("symbol_lengths", self.symbol_lengths());
        visitor.visit_child("codes", self.codes());
        visitor.visit_child("uncompressed_lengths", self.uncompressed_lengths());
    }

    fn _metadata(&self) -> SerdeMetadata<FSSTMetadata> {
        SerdeMetadata(FSSTMetadata {
            symbols_len: self.symbols().len(),
            codes_nullability: self.codes().dtype().nullability(),
            uncompressed_lengths_ptype: PType::try_from(self.uncompressed_lengths().dtype())
                .vortex_expect("Must be a valid PType"),
        })
    }
}

impl SerdeVTable<&FSSTArray> for FSSTEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = SerdeMetadata::<FSSTMetadata>::deserialize(parts.metadata())?;

        let symbols = parts
            .child(0)
            .decode(ctx, SYMBOLS_DTYPE.clone(), metadata.symbols_len)?;
        let symbol_lengths =
            parts
                .child(1)
                .decode(ctx, SYMBOL_LENS_DTYPE.clone(), metadata.symbols_len)?;
        let codes = parts
            .child(2)
            .decode(ctx, DType::Binary(metadata.codes_nullability), len)?;
        let uncompressed_lengths = parts.child(3).decode(
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
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_array::SerdeMetadata;
    use vortex_dtype::{Nullability, PType};

    use crate::serde::FSSTMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fsst_metadata() {
        check_metadata(
            "fsst.metadata",
            SerdeMetadata(FSSTMetadata {
                symbols_len: usize::MAX,
                codes_nullability: Nullability::Nullable,
                uncompressed_lengths_ptype: PType::U64,
            }),
        );
    }
}
