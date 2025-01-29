use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::ids;
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{
    CanonicalVTable, StatisticsVTable, ValidateVTable, ValidityVTable, VariantsVTable,
    VisitorVTable,
};
use vortex_array::{impl_encoding, ArrayData, Canonical, IntoArrayData, SerdeMetadata};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::alp::{alp_encode, decompress, Exponents};

impl_encoding!("vortex.alp", ids::ALP, ALP, SerdeMetadata<ALPMetadata>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPMetadata {
    pub(crate) exponents: Exponents,
    pub(crate) patches: Option<PatchesMetadata>,
}

impl ALPArray {
    pub fn try_new(
        encoded: ArrayData,
        exponents: Exponents,
        patches: Option<Patches>,
    ) -> VortexResult<Self> {
        let dtype = match encoded.dtype() {
            DType::Primitive(PType::I32, nullability) => DType::Primitive(PType::F32, *nullability),
            DType::Primitive(PType::I64, nullability) => DType::Primitive(PType::F64, *nullability),
            d => vortex_bail!(MismatchedTypes: "int32 or int64", d),
        };

        let length = encoded.len();

        let mut children = Vec::with_capacity(2);
        children.push(encoded);
        if let Some(patches) = &patches {
            children.push(patches.indices().clone());
            children.push(patches.values().clone());
        }

        let patches = patches
            .as_ref()
            .map(|p| p.to_metadata(length, &dtype))
            .transpose()?;

        Self::try_from_parts(
            dtype,
            length,
            SerdeMetadata(ALPMetadata { exponents, patches }),
            None,
            Some(children.into()),
            Default::default(),
        )
    }

    pub fn encode(array: ArrayData) -> VortexResult<ArrayData> {
        if let Some(parray) = PrimitiveArray::maybe_from(array) {
            Ok(alp_encode(&parray)?.into_array())
        } else {
            vortex_bail!("ALP can only encode primitive arrays");
        }
    }

    pub fn encoded(&self) -> ArrayData {
        self.as_ref()
            .child(0, &self.encoded_dtype(), self.len())
            .vortex_expect("Missing encoded child in ALPArray")
    }

    #[inline]
    pub fn exponents(&self) -> Exponents {
        self.metadata().exponents
    }

    pub fn patches(&self) -> Option<Patches> {
        self.metadata().patches.as_ref().map(|p| {
            Patches::new(
                self.len(),
                self.as_ref()
                    .child(1, &p.indices_dtype(), p.len())
                    .vortex_expect("ALPArray: patch indices"),
                self.as_ref()
                    .child(2, self.dtype(), p.len())
                    .vortex_expect("ALPArray: patch values"),
            )
        })
    }

    #[inline]
    fn encoded_dtype(&self) -> DType {
        match self.dtype() {
            DType::Primitive(PType::F32, _) => {
                DType::Primitive(PType::I32, self.dtype().nullability())
            }
            DType::Primitive(PType::F64, _) => {
                DType::Primitive(PType::I64, self.dtype().nullability())
            }
            d => vortex_panic!(MismatchedTypes: "f32 or f64", d),
        }
    }
}

impl ValidateVTable<ALPArray> for ALPEncoding {}

impl VariantsVTable<ALPArray> for ALPEncoding {
    fn as_primitive_array<'a>(&self, array: &'a ALPArray) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }
}

impl PrimitiveArrayTrait for ALPArray {}

impl ValidityVTable<ALPArray> for ALPEncoding {
    fn is_valid(&self, array: &ALPArray, index: usize) -> VortexResult<bool> {
        array.encoded().is_valid(index)
    }

    fn logical_validity(&self, array: &ALPArray) -> VortexResult<Mask> {
        array.encoded().logical_validity()
    }
}

impl CanonicalVTable<ALPArray> for ALPEncoding {
    fn into_canonical(&self, array: ALPArray) -> VortexResult<Canonical> {
        decompress(array).map(Canonical::Primitive)
    }
}

impl VisitorVTable<ALPArray> for ALPEncoding {
    fn accept(&self, array: &ALPArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("encoded", &array.encoded())?;
        if let Some(patches) = array.patches().as_ref() {
            visitor.visit_patches(patches)?;
        }
        Ok(())
    }
}

impl StatisticsVTable<ALPArray> for ALPEncoding {}

#[cfg(test)]
mod tests {
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::SerdeMetadata;
    use vortex_dtype::PType;

    use crate::{ALPMetadata, Exponents};

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alp_metadata() {
        check_metadata(
            "alp.metadata",
            SerdeMetadata(ALPMetadata {
                patches: Some(PatchesMetadata::new(usize::MAX, PType::U64)),
                exponents: Exponents {
                    e: u8::MAX,
                    f: u8::MAX,
                },
            }),
        );
    }
}
