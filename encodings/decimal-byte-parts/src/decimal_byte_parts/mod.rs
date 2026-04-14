// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayView;
pub(crate) mod compute;
mod rules;
mod slice;

use prost::Message as _;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::PType;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::decimal_byte_parts::compute::kernel::PARENT_KERNELS;
use crate::decimal_byte_parts::rules::PARENT_RULES;

/// A [`DecimalByteParts`]-encoded Vortex array.
pub type DecimalBytePartsArray = Array<DecimalByteParts>;

impl ArrayHash for DecimalBytePartsData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for DecimalBytePartsData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

#[derive(Clone, prost::Message)]
pub struct DecimalBytesPartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    lower_part_count: u32,
}

impl VTable for DecimalByteParts {
    type ArrayData = DecimalBytePartsData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.decimal_byte_parts");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("expected decimal dtype, got {}", dtype)
        };
        let msp = slots[MSP_SLOT]
            .as_ref()
            .vortex_expect("DecimalBytePartsArray msp slot");
        DecimalBytePartsData::validate(msp, *decimal_dtype, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DecimalBytePartsArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("DecimalBytePartsArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DecimalBytesPartsMetadata {
                zeroth_child_ptype: PType::try_from(array.msp().dtype())? as i32,
                lower_part_count: 0,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = DecimalBytesPartsMetadata::decode(metadata)?;
        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("decoding decimal but given non decimal dtype {}", dtype)
        };

        let encoded_dtype = DType::Primitive(metadata.zeroth_child_ptype(), dtype.nullability());

        let msp = children.get(0, &encoded_dtype, len)?;

        assert_eq!(
            metadata.lower_part_count, 0,
            "lower_part_count > 0 not currently supported"
        );

        let slots = vec![Some(msp.clone())];
        let data = DecimalBytePartsData::try_new(msp.dtype(), msp.len(), *decimal_dtype)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        to_canonical_decimal(&array, ctx).map(ExecutionResult::done)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

/// The most significant parts of the decimal values.
pub(super) const MSP_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["msp"];

/// This array encodes decimals as between 1-4 columns of primitive typed children.
/// The most significant part (msp) sorting the most significant decimal bits.
/// This array must be signed and is nullable iff the decimal is nullable.
///
/// e.g. for a decimal i128 \[ 127..64 | 64..0 \] msp = 127..64 and lower_part\[0\] = 64..0
#[derive(Clone, Debug)]
pub struct DecimalBytePartsData {
    // NOTE: the lower_parts is currently unused, we reserve this field so that it is properly
    //  read/written during serde, but provide no constructor to initialize this to anything
    //  other than the empty Vec.
    _lower_parts: Vec<ArrayRef>,
}

impl Display for DecimalBytePartsData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub struct DecimalBytePartsDataParts {
    pub msp: ArrayRef,
}

pub trait DecimalBytePartsArrayExt: TypedArrayRef<DecimalByteParts> {
    fn msp(&self) -> &ArrayRef {
        self.as_ref().slots()[MSP_SLOT]
            .as_ref()
            .vortex_expect("DecimalBytePartsArray msp slot")
    }
}

impl<T: TypedArrayRef<DecimalByteParts>> DecimalBytePartsArrayExt for T {}

impl DecimalBytePartsData {
    pub fn validate(
        msp: &ArrayRef,
        decimal_dtype: DecimalDType,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        if !msp.dtype().is_signed_int() {
            vortex_bail!("decimal bytes parts, first part must be a signed array")
        }

        let expected_dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        vortex_ensure!(
            dtype == &expected_dtype,
            "expected dtype {expected_dtype}, got {dtype}"
        );
        vortex_ensure!(msp.len() == len, "expected len {len}, got {}", msp.len());
        Ok(())
    }

    pub(crate) fn try_new(
        msp_dtype: &DType,
        msp_len: usize,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<Self> {
        let expected_dtype = DType::Decimal(decimal_dtype, msp_dtype.nullability());
        vortex_ensure!(
            msp_dtype.is_signed_int(),
            "decimal bytes parts, first part must be a signed array"
        );
        let _ = msp_len;
        drop(expected_dtype);
        Ok(Self {
            _lower_parts: Vec::new(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct DecimalByteParts;

impl DecimalByteParts {
    /// Construct a new [`DecimalBytePartsArray`] from an MSP array and decimal dtype.
    pub fn try_new(
        msp: ArrayRef,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<DecimalBytePartsArray> {
        let len = msp.len();
        let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        let slots = vec![Some(msp.clone())];
        let data = DecimalBytePartsData::try_new(msp.dtype(), msp.len(), decimal_dtype)?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(DecimalByteParts, dtype, len, data).with_slots(slots),
            )
        })
    }
}

/// Converts a DecimalBytePartsArray to its canonical DecimalArray representation.
fn to_canonical_decimal(
    array: &DecimalBytePartsArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // TODO(joe): support parts len != 1
    let prim = array.msp().clone().execute::<PrimitiveArray>(ctx)?;
    // Depending on the decimal type and the min/max of the primitive array we can choose
    // the correct buffer size

    Ok(match_each_signed_integer_ptype!(prim.ptype(), |P| {
        // SAFETY: The primitive array's buffer is already validated with correct type.
        // The decimal dtype matches the array's dtype, and validity is preserved.
        unsafe {
            DecimalArray::new_unchecked(
                prim.to_buffer::<P>(),
                *array
                    .dtype()
                    .as_decimal_opt()
                    .vortex_expect("must be a decimal dtype"),
                prim.validity()?,
            )
        }
        .into_array()
    }))
}

impl OperationsVTable<DecimalByteParts> for DecimalByteParts {
    fn scalar_at(
        array: ArrayView<'_, DecimalByteParts>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // TODO(joe): support parts len != 1
        let scalar = array.msp().scalar_at(index)?;

        // Note. values in msp, can only be signed integers upto size i64.
        let primitive_scalar = scalar.as_primitive();
        // TODO(joe): extend this to support multiple parts.
        let value = primitive_scalar.as_::<i64>().vortex_expect("non-null");
        Scalar::try_new(
            array.dtype().clone(),
            Some(ScalarValue::Decimal(DecimalValue::I64(value))),
        )
    }
}

impl ValidityChild<DecimalByteParts> for DecimalByteParts {
    fn validity_child(array: ArrayView<'_, DecimalByteParts>) -> ArrayRef {
        // validity stored in 0th child
        array.msp().clone()
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::DecimalByteParts;

    #[test]
    fn test_scalar_at_decimal_parts() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let array = DecimalByteParts::try_new(
            PrimitiveArray::new(
                buffer![100i32, 200i32, 400i32],
                Validity::Array(BoolArray::from_iter(vec![false, true, true]).into_array()),
            )
            .into_array(),
            decimal_dtype,
        )
        .unwrap()
        .into_array();

        assert_eq!(Scalar::null(dtype.clone()), array.scalar_at(0).unwrap());
        assert_eq!(
            Scalar::try_new(
                dtype.clone(),
                Some(ScalarValue::Decimal(DecimalValue::I64(200)))
            )
            .unwrap(),
            array.scalar_at(1).unwrap()
        );
        assert_eq!(
            Scalar::try_new(dtype, Some(ScalarValue::Decimal(DecimalValue::I64(400)))).unwrap(),
            array.scalar_at(2).unwrap()
        );
    }
}
