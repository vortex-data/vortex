use std::fmt::Display;

use serde::{Deserialize, Serialize};
use vortex_array::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::ids;
use vortex_array::stats::{ArrayStatisticsCompute, StatsSet};
use vortex_array::validity::{ArrayValidity, LogicalValidity};
use vortex_array::variants::{ArrayVariants, PrimitiveArrayTrait};
use vortex_array::{
    impl_encoding, Array, ArrayDType, ArrayTrait, Canonical, IntoArray, IntoArrayVariant,
    IntoCanonical,
};
use vortex_dtype::{DType, PType};
use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexExpect as _, VortexResult, VortexUnwrap as _,
};

use crate::compress::zigzag_encode;
use crate::zigzag_decode;

impl_encoding!("vortex.zigzag", ids::ZIGZAG, ZigZag);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZigZagMetadata;

impl Display for ZigZagMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ZigZagMetadata")
    }
}

impl ZigZagArray {
    pub fn try_new(encoded: Array) -> VortexResult<Self> {
        let encoded_dtype = encoded.dtype().clone();
        if !encoded_dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }

        let dtype = DType::from(PType::try_from(&encoded_dtype)?.to_signed())
            .with_nullability(encoded_dtype.nullability());

        let len = encoded.len();
        let children = [encoded];

        Self::try_from_parts(dtype, len, ZigZagMetadata, children.into(), StatsSet::new())
    }

    pub fn encode(array: &Array) -> VortexResult<Array> {
        PrimitiveArray::try_from(array)
            .map_err(|_| vortex_err!("ZigZag can only encoding primitive arrays"))
            .and_then(zigzag_encode)
            .map(|a| a.into_array())
    }

    pub fn encoded(&self) -> Array {
        let ptype = PType::try_from(self.dtype()).unwrap_or_else(|err| {
            vortex_panic!(err, "Failed to convert DType {} to PType", self.dtype())
        });
        let encoded = DType::from(ptype.to_unsigned()).with_nullability(self.dtype().nullability());
        self.as_ref()
            .child(0, &encoded, self.len())
            .vortex_expect("ZigZagArray is missing its encoded child array")
    }

    pub fn ptype(&self) -> PType {
        PType::try_from(self.dtype()).vortex_unwrap()
    }
}

impl ArrayTrait for ZigZagArray {}

impl ArrayVariants for ZigZagArray {
    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for ZigZagArray {}

impl ArrayValidity for ZigZagArray {
    fn is_valid(&self, index: usize) -> bool {
        self.encoded().with_dyn(|a| a.is_valid(index))
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.encoded().with_dyn(|a| a.logical_validity())
    }
}

impl AcceptArrayVisitor for ZigZagArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("encoded", &self.encoded())
    }
}

impl ArrayStatisticsCompute for ZigZagArray {}

impl IntoCanonical for ZigZagArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        zigzag_decode(self.encoded().into_primitive()?).map(Canonical::Primitive)
    }
}
