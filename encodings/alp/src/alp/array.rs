use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::ids;
use vortex_array::stats::StatisticsVTable;
use vortex_array::validity::{ArrayValidity, LogicalValidity, ValidityVTable};
use vortex_array::variants::{ArrayVariants, PrimitiveArrayTrait};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData,
    IntoCanonical,
};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};

use crate::alp::{alp_encode, decompress, Exponents};

impl_encoding!("vortex.alp", ids::ALP, ALP);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPMetadata {
    exponents: Exponents,
}

impl Display for ALPMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ALPArray {
    pub fn try_new(
        encoded: ArrayData,
        exponents: Exponents,
        patches: Option<ArrayData>,
    ) -> VortexResult<Self> {
        let dtype = match encoded.dtype() {
            DType::Primitive(PType::I32, nullability) => DType::Primitive(PType::F32, *nullability),
            DType::Primitive(PType::I64, nullability) => DType::Primitive(PType::F64, *nullability),
            d => vortex_bail!(MismatchedTypes: "int32 or int64", d),
        };

        let length = encoded.len();
        if let Some(parray) = patches.as_ref() {
            if parray.len() != length {
                vortex_bail!(
                    "Mismatched length in ALPArray between encoded({}) {} and it's patches({}) {}",
                    encoded.encoding().id(),
                    encoded.len(),
                    parray.encoding().id(),
                    parray.len()
                )
            }
        }

        let mut children = Vec::with_capacity(2);
        children.push(encoded);
        if let Some(patch) = patches {
            if !patch.dtype().eq_ignore_nullability(&dtype) || !patch.dtype().is_nullable() {
                vortex_bail!(
                    "ALP patches dtype, {}, must be nullable version of array dtype, {}",
                    patch.dtype(),
                    dtype,
                );
            }
            children.push(patch);
        }

        Self::try_from_parts(
            dtype,
            length,
            ALPMetadata { exponents },
            children.into(),
            Default::default(),
        )
    }

    pub fn encode(array: ArrayData) -> VortexResult<ArrayData> {
        if let Ok(parray) = PrimitiveArray::try_from(array) {
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

    pub fn patches(&self) -> Option<ArrayData> {
        (self.as_ref().nchildren() > 1).then(|| {
            self.as_ref()
                .child(1, &self.patches_dtype(), self.len())
                .vortex_expect("Missing patches child in ALPArray")
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

    #[inline]
    fn patches_dtype(&self) -> DType {
        self.dtype().as_nullable()
    }
}

impl ArrayTrait for ALPArray {}

impl ArrayVariants for ALPArray {
    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for ALPArray {}

impl ValidityVTable<ALPArray> for ALPEncoding {
    fn is_valid(&self, array: &ALPArray, index: usize) -> bool {
        array.encoded().is_valid(index)
    }

    fn logical_validity(&self, array: &ALPArray) -> LogicalValidity {
        array.encoded().logical_validity()
    }
}

impl IntoCanonical for ALPArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        decompress(self).map(Canonical::Primitive)
    }
}

impl VisitorVTable<ALPArray> for ALPEncoding {
    fn accept(&self, array: &ALPArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("encoded", &array.encoded())?;
        if let Some(patches) = array.patches().as_ref() {
            visitor.visit_child("patches", patches)?;
        }
        Ok(())
    }
}

impl StatisticsVTable<ALPArray> for ALPEncoding {}
