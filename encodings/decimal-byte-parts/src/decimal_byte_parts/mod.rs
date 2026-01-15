// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compute;
mod rules;

use std::hash::Hash;
use std::ops::Range;

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::ToCanonical;
use vortex_array::arrays::DecimalArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ArrayVTable;
use vortex_array::vtable::ArrayVTableExt;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityHelper;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_array::vtable::VisitorVTable;
use vortex_dtype::DType;
use vortex_dtype::DecimalDType;
use vortex_dtype::PType;
use vortex_dtype::match_each_signed_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_scalar::DecimalValue;
use vortex_scalar::Scalar;

use crate::decimal_byte_parts::rules::PARENT_RULES;

vtable!(DecimalByteParts);

#[derive(Clone, prost::Message)]
pub struct DecimalBytesPartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    lower_part_count: u32,
}

impl VTable for DecimalBytePartsVTable {
    type Array = DecimalBytePartsArray;

    type Metadata = ProstMetadata<DecimalBytesPartsMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.decimal_byte_parts")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        DecimalBytePartsVTable.as_vtable()
    }

    fn metadata(array: &DecimalBytePartsArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DecimalBytesPartsMetadata {
            zeroth_child_ptype: PType::try_from(array.msp.dtype())? as i32,
            lower_part_count: 0,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DecimalBytesPartsMetadata::decode(buffer)?))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DecimalBytePartsArray> {
        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("decoding decimal but given non decimal dtype {}", dtype)
        };

        let encoded_dtype = DType::Primitive(metadata.zeroth_child_ptype(), dtype.nullability());

        let msp = children.get(0, &encoded_dtype, len)?;

        assert_eq!(
            metadata.lower_part_count, 0,
            "lower_part_count > 0 not currently supported"
        );

        DecimalBytePartsArray::try_new(msp, *decimal_dtype)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "DecimalBytePartsArray expects exactly 1 child (msp), got {}",
            children.len()
        );
        array.msp = children.into_iter().next().vortex_expect("checked");
        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing encoded MSP does not change the encoded values
        Ok(Some(unsafe {
            DecimalBytePartsArray::new_unchecked(array.msp.slice(range), *array.decimal_dtype())
                .into_array()
        }))
    }
}

/// This array encodes decimals as between 1-4 columns of primitive typed children.
/// The most significant part (msp) sorting the most significant decimal bits.
/// This array must be signed and is nullable iff the decimal is nullable.
///
/// e.g. for a decimal i128 \[ 127..64 | 64..0 \] msp = 127..64 and lower_part\[0\] = 64..0
#[derive(Clone, Debug)]
pub struct DecimalBytePartsArray {
    msp: ArrayRef,
    // NOTE: the lower_parts is currently unused, we reserve this field so that it is properly
    //  read/written during serde, but provide no constructor to initialize this to anything
    //  other than the empty Vec.
    _lower_parts: Vec<ArrayRef>,
    dtype: DType,
    stats_set: ArrayStats,
}

impl DecimalBytePartsArray {
    pub fn try_new(msp: ArrayRef, decimal_dtype: DecimalDType) -> VortexResult<Self> {
        if !msp.dtype().is_signed_int() {
            vortex_bail!("decimal bytes parts, first part must be a signed array")
        }

        let nullable = msp.dtype().nullability();
        Ok(Self {
            msp,
            _lower_parts: Vec::new(),
            dtype: DType::Decimal(decimal_dtype, nullable),
            stats_set: Default::default(),
        })
    }

    pub(crate) unsafe fn new_unchecked(msp: ArrayRef, decimal_dtype: DecimalDType) -> Self {
        let nullable = msp.dtype().nullability();
        Self {
            msp,
            _lower_parts: Vec::new(),
            dtype: DType::Decimal(decimal_dtype, nullable),
            stats_set: Default::default(),
        }
    }

    pub fn decimal_dtype(&self) -> &DecimalDType {
        self.dtype
            .as_decimal_opt()
            .vortex_expect("must be a decimal dtype")
    }

    pub(crate) fn msp(&self) -> &ArrayRef {
        &self.msp
    }
}

#[derive(Debug)]
pub struct DecimalBytePartsVTable;

impl BaseArrayVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn len(array: &DecimalBytePartsArray) -> usize {
        array.msp.len()
    }

    fn dtype(array: &DecimalBytePartsArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DecimalBytePartsArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &DecimalBytePartsArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.msp.array_hash(state, precision);
    }

    fn array_eq(
        array: &DecimalBytePartsArray,
        other: &DecimalBytePartsArray,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype && array.msp.array_eq(&other.msp, precision)
    }
}

impl CanonicalVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn canonicalize(array: &DecimalBytePartsArray) -> Canonical {
        // TODO(joe): support parts len != 1
        let prim = array.msp.to_primitive();
        // Depending on the decimal type and the min/max of the primitive array we can choose
        // the correct buffer size

        match_each_signed_integer_ptype!(prim.ptype(), |P| {
            // SAFETY: The primitive array's buffer is already validated with correct type.
            // The decimal dtype matches the array's dtype, and validity is preserved.
            Canonical::Decimal(unsafe {
                DecimalArray::new_unchecked(
                    prim.to_buffer::<P>(),
                    *array.decimal_dtype(),
                    prim.validity().clone(),
                )
            })
        })
    }
}

impl OperationsVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn scalar_at(array: &DecimalBytePartsArray, index: usize) -> Scalar {
        // TODO(joe): support parts len != 1
        let scalar = array.msp.scalar_at(index);

        // Note. values in msp, can only be signed integers upto size i64.
        let primitive_scalar = scalar.as_primitive();
        // TODO(joe): extend this to support multiple parts.
        let value = primitive_scalar.as_::<i64>().vortex_expect("non-null");
        Scalar::new(array.dtype.clone(), DecimalValue::I64(value).into())
    }
}

impl ValidityChild<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn validity_child(array: &DecimalBytePartsArray) -> &ArrayRef {
        // validity stored in 0th child
        &array.msp
    }
}

impl VisitorVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn visit_buffers(_array: &DecimalBytePartsArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DecimalBytePartsArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("msp", &array.msp);
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::Array;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::DecimalDType;
    use vortex_dtype::Nullability;
    use vortex_scalar::DecimalValue;
    use vortex_scalar::Scalar;

    use crate::DecimalBytePartsArray;

    #[test]
    fn test_scalar_at_decimal_parts() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let array = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(
                buffer![100i32, 200i32, 400i32],
                Validity::Array(BoolArray::from_iter(vec![false, true, true]).to_array()),
            )
            .to_array(),
            decimal_dtype,
        )
        .unwrap()
        .to_array();

        assert_eq!(Scalar::null(dtype.clone()), array.scalar_at(0));
        assert_eq!(
            Scalar::new(dtype.clone(), DecimalValue::I64(200).into()),
            array.scalar_at(1)
        );
        assert_eq!(
            Scalar::new(dtype, DecimalValue::I64(400).into()),
            array.scalar_at(2)
        );
    }
}
