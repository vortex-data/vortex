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
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
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
    type TypedArrayData = DecimalBytePartsData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.decimal_byte_parts");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
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
        let lower = slots.get(LOWER_SLOT).and_then(Option::as_ref);
        DecimalBytePartsData::validate(msp, lower, *decimal_dtype, dtype, len)
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
                lower_part_count: u32::from(array.lower().is_some()),
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

        let slots = match metadata.lower_part_count {
            0 => smallvec![Some(msp.clone())],
            1 => {
                let lower_dtype = DType::Primitive(LOWER_PTYPE, Nullability::NonNullable);
                let lower = children.get(1, &lower_dtype, len)?;
                smallvec![Some(msp.clone()), Some(lower)]
            }
            n => vortex_bail!("decimal byte parts supports at most one lower limb, got {n}"),
        };

        let lower = slots.as_slice().get(LOWER_SLOT).and_then(Option::as_ref);
        DecimalBytePartsData::validate(&msp, lower, *decimal_dtype, dtype, len)?;
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, DecimalBytePartsData)
                .with_slots(slots),
        )
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

/// The most significant part (high limb) of the decimal values.
pub(super) const MSP_SLOT: usize = 0;
/// The single lower limb, present only for the two-limb i128 representation.
pub(super) const LOWER_SLOT: usize = 1;
/// The maximum number of children an array of this encoding can hold.
pub(super) const MAX_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; MAX_SLOTS] = ["msp", "lower"];

/// The physical type of the lower limb in the two-limb representation.
pub(super) const LOWER_PTYPE: PType = PType::U64;

/// This array encodes decimals as 1 or 2 columns of primitive typed children.
/// The most significant part (msp) stores the most significant decimal bits and is always a
/// signed integer, nullable iff the decimal is nullable.
///
/// With a single child the decimal value is exactly the (sign-extended) msp. With two children the
/// value is reconstructed as `(msp as i128) << 64 | (lower as u64 as i128)`, i.e. the msp is the
/// signed high 64-bit limb and `lower` is the unsigned low 64-bit limb. The lower limb is always a
/// non-nullable `u64`; validity is carried solely by the msp.
///
/// e.g. for a decimal i128 \[ 127..64 | 64..0 \] msp = 127..64 and lower = 64..0
#[derive(Clone, Debug)]
pub struct DecimalBytePartsData;

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

    /// The lower (low 64-bit) limb, present only for the two-limb i128 representation.
    fn lower(&self) -> Option<&ArrayRef> {
        self.as_ref()
            .slots()
            .get(LOWER_SLOT)
            .and_then(Option::as_ref)
    }
}

impl<T: TypedArrayRef<DecimalByteParts>> DecimalBytePartsArrayExt for T {}

impl DecimalBytePartsData {
    pub fn validate(
        msp: &ArrayRef,
        lower: Option<&ArrayRef>,
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

        if let Some(lower) = lower {
            vortex_ensure!(
                matches!(msp.dtype(), DType::Primitive(PType::I64, _)),
                "two-limb decimal byte parts requires an i64 high limb, got {}",
                msp.dtype()
            );
            vortex_ensure!(
                lower.dtype() == &DType::Primitive(LOWER_PTYPE, Nullability::NonNullable),
                "decimal byte parts lower limb must be a non-nullable {LOWER_PTYPE}, got {}",
                lower.dtype()
            );
            vortex_ensure!(
                lower.len() == len,
                "lower limb length {} != array length {len}",
                lower.len()
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DecimalByteParts;

impl DecimalByteParts {
    /// Construct a new single-limb [`DecimalBytePartsArray`] from an MSP array and decimal dtype.
    pub fn try_new(
        msp: ArrayRef,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<DecimalBytePartsArray> {
        let len = msp.len();
        let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        DecimalBytePartsData::validate(&msp, None, decimal_dtype, &dtype, len)?;
        let slots = smallvec![Some(msp)];
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(DecimalByteParts, dtype, len, DecimalBytePartsData)
                    .with_slots(slots),
            )
        })
    }

    /// Construct a two-limb [`DecimalBytePartsArray`] representing an i128 decimal.
    ///
    /// `msp` is the signed high 64-bit limb (carrying validity); `lower` is the non-nullable
    /// unsigned low 64-bit limb. The decimal value at index `i` is
    /// `(msp[i] as i128) << 64 | (lower[i] as u64 as i128)`.
    pub fn try_new_with_lower(
        msp: ArrayRef,
        lower: ArrayRef,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<DecimalBytePartsArray> {
        let len = msp.len();
        let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        DecimalBytePartsData::validate(&msp, Some(&lower), decimal_dtype, &dtype, len)?;
        let slots = smallvec![Some(msp), Some(lower)];
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(DecimalByteParts, dtype, len, DecimalBytePartsData)
                    .with_slots(slots),
            )
        })
    }
}

/// Converts a DecimalBytePartsArray to its canonical DecimalArray representation.
fn to_canonical_decimal(
    array: &DecimalBytePartsArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let decimal_dtype = *array
        .dtype()
        .as_decimal_opt()
        .vortex_expect("must be a decimal dtype");
    let msp = array.msp().clone().execute::<PrimitiveArray>(ctx)?;

    let Some(lower) = array.lower() else {
        // Single-limb: the decimal is exactly the sign-extended msp.
        return Ok(match_each_signed_integer_ptype!(msp.ptype(), |P| {
            // SAFETY: The primitive array's buffer is already validated with correct type.
            // The decimal dtype matches the array's dtype, and validity is preserved.
            unsafe {
                DecimalArray::new_unchecked(msp.to_buffer::<P>(), decimal_dtype, msp.validity()?)
            }
            .into_array()
        }));
    };

    // Two-limb: reconstruct each i128 as `(high as i128) << 64 | low`.
    let lower = lower.clone().execute::<PrimitiveArray>(ctx)?;
    let validity = msp.validity()?;
    let low = lower.as_slice::<u64>();
    let values: Buffer<i128> = match_each_signed_integer_ptype!(msp.ptype(), |P| {
        msp.as_slice::<P>()
            .iter()
            .zip(low)
            .map(|(&high, &low)| ((high as i128) << 64) | i128::from(low))
            .collect()
    });

    // SAFETY: validity comes from the msp and the reconstructed values match the decimal dtype.
    Ok(unsafe { DecimalArray::new_unchecked(values, decimal_dtype, validity) }.into_array())
}

impl OperationsVTable<DecimalByteParts> for DecimalByteParts {
    fn scalar_at(
        array: ArrayView<'_, DecimalByteParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Null indices are short-circuited by the validity child (the msp), so the msp scalar
        // here is always non-null.
        let high = array
            .msp()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("non-null");

        let decimal_value = match array.lower() {
            None => DecimalValue::I64(high),
            Some(lower) => {
                let low = lower
                    .execute_scalar(index, ctx)?
                    .as_primitive()
                    .as_::<u64>()
                    .vortex_expect("lower limb is non-nullable");
                DecimalValue::I128((i128::from(high) << 64) | i128::from(low))
            }
        };
        Scalar::try_new(
            array.dtype().clone(),
            Some(ScalarValue::Decimal(decimal_value)),
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
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
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

        assert_eq!(
            Scalar::null(dtype.clone()),
            array
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            Scalar::try_new(
                dtype.clone(),
                Some(ScalarValue::Decimal(DecimalValue::I64(200)))
            )
            .unwrap(),
            array
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            Scalar::try_new(dtype, Some(ScalarValue::Decimal(DecimalValue::I64(400)))).unwrap(),
            array
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
    }
}
