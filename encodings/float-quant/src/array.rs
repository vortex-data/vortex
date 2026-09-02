// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
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
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::bitpack_decompress::unpack_map;
use vortex_fastlanes::bitpack_decompress::unpack_pair_map;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::rules::RULES;

const METADATA_VERSION: u8 = 1;
const METADATA_LEN: usize = 2;

/// A lossless float split with quantized high bits and low-bit adjustments.
pub type FloatQuantArray = Array<FloatQuant>;

#[array_slots(FloatQuant)]
pub struct FloatQuantSlots {
    /// Ordered float bits after the low `k` bits are removed.
    #[slot(0)]
    pub primary: ArrayRef,
    /// Sign-normalized low `k` bits.
    #[slot(1)]
    pub secondary: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct FloatQuantData {
    pub(crate) k: u8,
}

impl Display for FloatQuantData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "k: {}", self.k)
    }
}

impl ArrayHash for FloatQuantData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.k.hash(state);
    }
}

impl ArrayEq for FloatQuantData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.k == other.k
    }
}

#[derive(Clone, Debug)]
pub struct FloatQuant;

impl VTable for FloatQuant {
    type TypedArrayData = FloatQuantData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.float_quant");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let ptype = PType::try_from(dtype)?;
        let latent_ptype = latent_ptype(ptype)?;
        let precision_bits = precision_bits(ptype)?;
        vortex_ensure!(
            data.k > 0 && data.k <= precision_bits,
            "FloatQuant k {} exceeds {ptype} precision {precision_bits}",
            data.k
        );

        let slots = FloatQuantSlotsView::from_slots(slots);
        let expected_primary = DType::Primitive(latent_ptype, dtype.nullability());
        vortex_ensure!(
            slots.primary.dtype() == &expected_primary,
            "expected primary dtype {expected_primary}, got {}",
            slots.primary.dtype()
        );
        vortex_ensure!(
            slots.primary.len() == len,
            "FloatQuant primary length differs"
        );
        if let Some(secondary) = slots.secondary {
            let expected_secondary = DType::Primitive(latent_ptype, NonNullable);
            vortex_ensure!(
                secondary.dtype() == &expected_secondary,
                "expected secondary dtype {expected_secondary}, got {}",
                secondary.dtype()
            );
            vortex_ensure!(
                secondary.len() == len,
                "FloatQuant secondary length differs"
            );
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FloatQuantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("FloatQuantArray buffer_name index {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![METADATA_VERSION, array.data().k]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(buffers.is_empty(), "FloatQuant expects no buffers");
        vortex_ensure!(
            metadata.len() == METADATA_LEN,
            "FloatQuant metadata requires {METADATA_LEN} bytes"
        );
        vortex_ensure!(
            metadata[0] == METADATA_VERSION,
            "unsupported FloatQuant metadata version {}",
            metadata[0]
        );
        vortex_ensure!(
            matches!(children.len(), 1 | 2),
            "FloatQuant requires one or two children"
        );

        let ptype = PType::try_from(dtype)?;
        let latent_ptype = latent_ptype(ptype)?;
        let primary_dtype = DType::Primitive(latent_ptype, dtype.nullability());
        let primary = children.get(0, &primary_dtype, len)?;
        let secondary = if children.len() == 2 {
            let secondary_dtype = DType::Primitive(latent_ptype, NonNullable);
            Some(children.get(1, &secondary_dtype, len)?)
        } else {
            None
        };
        let slots = FloatQuantSlots { primary, secondary }.into_slots();
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            FloatQuantData { k: metadata[1] },
        )
        .with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        FloatQuantSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            decode(array.as_view(), ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<FloatQuant> for FloatQuant {
    fn scalar_at(
        array: ArrayView<'_, FloatQuant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let primary = array.primary().execute_scalar(index, ctx)?;
        if primary.is_null() {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        let k = array.data().k;
        Ok(match PType::try_from(array.dtype())? {
            PType::F16 => Scalar::primitive(
                join_f16(
                    primary
                        .as_primitive()
                        .typed_value::<u16>()
                        .vortex_expect("validated primary scalar"),
                    array
                        .secondary()
                        .map(|secondary| {
                            secondary
                                .execute_scalar(index, ctx)?
                                .as_primitive()
                                .typed_value::<u16>()
                                .ok_or_else(|| {
                                    vortex_error::vortex_err!("validated secondary scalar is null")
                                })
                        })
                        .transpose()?
                        .unwrap_or(0),
                    k,
                ),
                array.dtype().nullability(),
            ),
            PType::F32 => Scalar::primitive(
                join_f32(
                    primary
                        .as_primitive()
                        .typed_value::<u32>()
                        .vortex_expect("validated primary scalar"),
                    array
                        .secondary()
                        .map(|secondary| {
                            secondary
                                .execute_scalar(index, ctx)?
                                .as_primitive()
                                .typed_value::<u32>()
                                .ok_or_else(|| {
                                    vortex_error::vortex_err!("validated secondary scalar is null")
                                })
                        })
                        .transpose()?
                        .unwrap_or(0),
                    k,
                ),
                array.dtype().nullability(),
            ),
            PType::F64 => Scalar::primitive(
                join_f64(
                    primary
                        .as_primitive()
                        .typed_value::<u64>()
                        .vortex_expect("validated primary scalar"),
                    array
                        .secondary()
                        .map(|secondary| {
                            secondary
                                .execute_scalar(index, ctx)?
                                .as_primitive()
                                .typed_value::<u64>()
                                .ok_or_else(|| {
                                    vortex_error::vortex_err!("validated secondary scalar is null")
                                })
                        })
                        .transpose()?
                        .unwrap_or(0),
                    k,
                ),
                array.dtype().nullability(),
            ),
            ptype => vortex_panic!("unsupported FloatQuant ptype {ptype}"),
        })
    }
}

impl ValidityChild<FloatQuant> for FloatQuant {
    fn validity_child(array: ArrayView<'_, FloatQuant>) -> ArrayRef {
        array.primary().clone()
    }
}

pub trait FloatQuantArrayExt: TypedArrayRef<FloatQuant> + FloatQuantArraySlotsExt {
    /// Return the number of split low bits.
    fn k(&self) -> u8 {
        self.deref().k
    }
}

impl<T: TypedArrayRef<FloatQuant>> FloatQuantArrayExt for T {}

impl FloatQuant {
    /// Construct a float quantization array from one or two latent children.
    pub fn try_new(
        primary: ArrayRef,
        secondary: Option<ArrayRef>,
        float_ptype: PType,
        k: u8,
    ) -> VortexResult<FloatQuantArray> {
        let dtype = DType::Primitive(float_ptype, primary.dtype().nullability());
        let len = primary.len();
        let slots = FloatQuantSlots { primary, secondary }.into_slots();
        Array::try_from_parts(
            ArrayParts::new(FloatQuant, dtype, len, FloatQuantData { k }).with_slots(slots),
        )
    }

    /// Split a canonical float array into two unsigned latent children.
    pub fn from_primitive(array: ArrayView<'_, Primitive>, k: u8) -> VortexResult<FloatQuantArray> {
        let validity = array.validity()?;
        match array.ptype() {
            PType::F16 => {
                let (primary, secondary) = split_f16(array.as_slice::<f16>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    Some(
                        PrimitiveArray::new(Buffer::from(secondary), NonNullable.into())
                            .into_array(),
                    ),
                    PType::F16,
                    k,
                )
            }
            PType::F32 => {
                let (primary, secondary) = split_f32(array.as_slice::<f32>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    Some(
                        PrimitiveArray::new(Buffer::from(secondary), NonNullable.into())
                            .into_array(),
                    ),
                    PType::F32,
                    k,
                )
            }
            PType::F64 => {
                let (primary, secondary) = split_f64(array.as_slice::<f64>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    Some(
                        PrimitiveArray::new(Buffer::from(secondary), NonNullable.into())
                            .into_array(),
                    ),
                    PType::F64,
                    k,
                )
            }
            ptype => vortex_bail!("FloatQuant requires f16, f32, or f64, got {ptype}"),
        }
    }

    /// Split floats whose lowest `k` fraction bits are zero.
    pub fn from_primitive_constant_secondary(
        array: ArrayView<'_, Primitive>,
        k: u8,
    ) -> VortexResult<FloatQuantArray> {
        let validity = array.validity()?;
        match array.ptype() {
            PType::F16 => {
                let primary = split_primary_f16(array.as_slice::<f16>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    None,
                    PType::F16,
                    k,
                )
            }
            PType::F32 => {
                let primary = split_primary_f32(array.as_slice::<f32>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    None,
                    PType::F32,
                    k,
                )
            }
            PType::F64 => {
                let primary = split_primary_f64(array.as_slice::<f64>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    None,
                    PType::F64,
                    k,
                )
            }
            ptype => vortex_bail!("FloatQuant requires f16, f32, or f64, got {ptype}"),
        }
    }

    /// Split a constant-secondary float array into frame-of-reference primary values.
    pub fn primary_for_primitive(
        array: ArrayView<'_, Primitive>,
        k: u8,
        primary_min: u64,
    ) -> VortexResult<PrimitiveArray> {
        let validity = array.validity()?;
        match array.ptype() {
            PType::F16 => {
                let primary_min = u16::try_from(primary_min)?;
                let primary = split_primary_for_f16(array.as_slice::<f16>(), k, primary_min)?;
                Ok(PrimitiveArray::new(Buffer::from(primary), validity))
            }
            PType::F32 => {
                let primary_min = u32::try_from(primary_min)?;
                let primary = split_primary_for_f32(array.as_slice::<f32>(), k, primary_min)?;
                Ok(PrimitiveArray::new(Buffer::from(primary), validity))
            }
            PType::F64 => {
                let primary = split_primary_for_f64(array.as_slice::<f64>(), k, primary_min)?;
                Ok(PrimitiveArray::new(Buffer::from(primary), validity))
            }
            ptype => vortex_bail!("FloatQuant requires f16, f32, or f64, got {ptype}"),
        }
    }
}

/// Compression facts derived during FloatQuant split selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatQuantAnalysis {
    /// Selected low-bit width.
    pub k: u8,
    /// Bit width of the frame-of-reference primary values.
    pub primary_bit_width: u8,
    /// Minimum primary value before frame-of-reference subtraction.
    pub primary_min: u64,
    /// Bit width of the secondary values. Zero identifies an implicit-zero child.
    pub secondary_bit_width: u8,
}

/// Analyze a canonical float array for a FloatQuant split.
pub fn analyze_float_quant(array: ArrayView<'_, Primitive>) -> Option<FloatQuantAnalysis> {
    match array.ptype() {
        PType::F16 => analyze_f16(array.as_slice::<f16>()),
        PType::F32 => analyze_f32(array.as_slice::<f32>()),
        PType::F64 => analyze_f64(array.as_slice::<f64>()),
        _ => None,
    }
}

fn analyze_bits(
    low_bits_or: u64,
    precision_bits: u8,
    len: usize,
    primary_min: u64,
    primary_max: u64,
) -> Option<FloatQuantAnalysis> {
    if len == 0 {
        return None;
    }

    let mut best = None;
    for k in 1..=precision_bits {
        let low_mask = (1_u64 << k) - 1;
        let secondary_bit_width =
            u8::try_from(u64::BITS - (low_bits_or & low_mask).leading_zeros()).unwrap_or(u8::MAX);
        if k - secondary_bit_width < 2 {
            continue;
        }

        let shifted_min = primary_min >> k;
        let shifted_max = primary_max >> k;
        let primary_bit_width =
            u8::try_from(u64::BITS - (shifted_max - shifted_min).leading_zeros())
                .unwrap_or(u8::MAX);
        let total_bit_width = primary_bit_width + secondary_bit_width;
        let candidate = (
            total_bit_width,
            secondary_bit_width,
            k,
            primary_bit_width,
            shifted_min,
        );
        if best.is_none_or(|current| candidate < current) {
            best = Some(candidate);
        }
    }

    let (_, secondary_bit_width, k, primary_bit_width, primary_min) = best?;
    Some(FloatQuantAnalysis {
        k,
        primary_bit_width,
        primary_min,
        secondary_bit_width,
    })
}

fn analyze_f16(values: &[f16]) -> Option<FloatQuantAnalysis> {
    let mut minimum = u16::MAX;
    let mut maximum = u16::MIN;
    let mut low_bits_or = 0_u16;
    for value in values {
        let bits = value.to_bits();
        let ordered = ordered_u16(bits);
        minimum = minimum.min(ordered);
        maximum = maximum.max(ordered);
        low_bits_or |= bits;
    }
    analyze_bits(
        u64::from(low_bits_or),
        10,
        values.len(),
        u64::from(minimum),
        u64::from(maximum),
    )
}

fn analyze_f32(values: &[f32]) -> Option<FloatQuantAnalysis> {
    let mut minimum = u32::MAX;
    let mut maximum = u32::MIN;
    let mut low_bits_or = 0_u32;
    for value in values {
        let bits = value.to_bits();
        let ordered = ordered_u32(bits);
        minimum = minimum.min(ordered);
        maximum = maximum.max(ordered);
        low_bits_or |= bits;
    }
    analyze_bits(
        u64::from(low_bits_or),
        23,
        values.len(),
        u64::from(minimum),
        u64::from(maximum),
    )
}

fn analyze_f64(values: &[f64]) -> Option<FloatQuantAnalysis> {
    let mut minimum = u64::MAX;
    let mut maximum = u64::MIN;
    let mut low_bits_or = 0_u64;
    for value in values {
        let bits = value.to_bits();
        let ordered = ordered_u64(bits);
        minimum = minimum.min(ordered);
        maximum = maximum.max(ordered);
        low_bits_or |= bits;
    }
    analyze_bits(low_bits_or, 52, values.len(), minimum, maximum)
}

fn latent_ptype(ptype: PType) -> VortexResult<PType> {
    match ptype {
        PType::F16 => Ok(PType::U16),
        PType::F32 => Ok(PType::U32),
        PType::F64 => Ok(PType::U64),
        _ => vortex_bail!("FloatQuant requires f16, f32, or f64, got {ptype}"),
    }
}

fn precision_bits(ptype: PType) -> VortexResult<u8> {
    match ptype {
        PType::F16 => Ok(10),
        PType::F32 => Ok(23),
        PType::F64 => Ok(52),
        _ => vortex_bail!("FloatQuant requires f16, f32, or f64, got {ptype}"),
    }
}

fn ordered_u16(bits: u16) -> u16 {
    if bits & (1_u16 << 15) == 0 {
        bits ^ (1_u16 << 15)
    } else {
        !bits
    }
}

fn ordered_u32(bits: u32) -> u32 {
    if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

fn ordered_u64(bits: u64) -> u64 {
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn split_f16(values: &[f16], k: u8) -> VortexResult<(Vec<u16>, Vec<u16>)> {
    vortex_ensure!(k > 0 && k <= 10, "FloatQuant f16 k must be in 1..=10");
    let low_mask = (1_u16 << k) - 1;
    let mut primary = Vec::with_capacity(values.len());
    let mut secondary = Vec::with_capacity(values.len());
    for &value in values {
        let bits = value.to_bits();
        let ordered = ordered_u16(bits);
        primary.push(ordered >> k);
        let low = ordered & low_mask;
        secondary.push(if bits & (1_u16 << 15) == 0 {
            low
        } else {
            low_mask - low
        });
    }
    Ok((primary, secondary))
}

fn split_f32(values: &[f32], k: u8) -> VortexResult<(Vec<u32>, Vec<u32>)> {
    vortex_ensure!(k > 0 && k <= 23, "FloatQuant f32 k must be in 1..=23");
    let low_mask = (1_u32 << k) - 1;
    let mut primary = Vec::with_capacity(values.len());
    let mut secondary = Vec::with_capacity(values.len());
    for &value in values {
        let bits = value.to_bits();
        let ordered = ordered_u32(bits);
        primary.push(ordered >> k);
        let low = ordered & low_mask;
        secondary.push(if bits & (1_u32 << 31) == 0 {
            low
        } else {
            low_mask - low
        });
    }
    Ok((primary, secondary))
}

fn split_f64(values: &[f64], k: u8) -> VortexResult<(Vec<u64>, Vec<u64>)> {
    vortex_ensure!(k > 0 && k <= 52, "FloatQuant f64 k must be in 1..=52");
    let low_mask = (1_u64 << k) - 1;
    let mut primary = Vec::with_capacity(values.len());
    let mut secondary = Vec::with_capacity(values.len());
    for &value in values {
        let bits = value.to_bits();
        let ordered = ordered_u64(bits);
        primary.push(ordered >> k);
        let low = ordered & low_mask;
        secondary.push(if bits & (1_u64 << 63) == 0 {
            low
        } else {
            low_mask - low
        });
    }
    Ok((primary, secondary))
}

fn split_primary_f32(values: &[f32], k: u8) -> VortexResult<Vec<u32>> {
    vortex_ensure!(k > 0 && k <= 23, "FloatQuant f32 k must be in 1..=23");
    let low_mask = (1_u32 << k) - 1;
    vortex_ensure!(
        values.iter().all(|value| value.to_bits() & low_mask == 0),
        "FloatQuant constant secondary requires zero low bits"
    );
    Ok(values
        .iter()
        .map(|value| ordered_u32(value.to_bits()) >> k)
        .collect())
}

fn split_primary_f16(values: &[f16], k: u8) -> VortexResult<Vec<u16>> {
    vortex_ensure!(k > 0 && k <= 10, "FloatQuant f16 k must be in 1..=10");
    let low_mask = (1_u16 << k) - 1;
    vortex_ensure!(
        values.iter().all(|value| value.to_bits() & low_mask == 0),
        "FloatQuant constant secondary requires zero low bits"
    );
    Ok(values
        .iter()
        .map(|value| ordered_u16(value.to_bits()) >> k)
        .collect())
}

fn split_primary_f64(values: &[f64], k: u8) -> VortexResult<Vec<u64>> {
    vortex_ensure!(k > 0 && k <= 52, "FloatQuant f64 k must be in 1..=52");
    let low_mask = (1_u64 << k) - 1;
    vortex_ensure!(
        values.iter().all(|value| value.to_bits() & low_mask == 0),
        "FloatQuant constant secondary requires zero low bits"
    );
    Ok(values
        .iter()
        .map(|value| ordered_u64(value.to_bits()) >> k)
        .collect())
}

fn split_primary_for_f32(values: &[f32], k: u8, primary_min: u32) -> VortexResult<Vec<u32>> {
    vortex_ensure!(k > 0 && k <= 23, "FloatQuant f32 k must be in 1..=23");
    let low_mask = (1_u32 << k) - 1;
    vortex_ensure!(
        values.iter().all(|value| value.to_bits() & low_mask == 0),
        "FloatQuant constant secondary requires zero low bits"
    );
    values
        .iter()
        .map(|value| {
            (ordered_u32(value.to_bits()) >> k)
                .checked_sub(primary_min)
                .ok_or_else(|| vortex_error::vortex_err!("FloatQuant primary minimum is invalid"))
        })
        .collect::<VortexResult<Vec<_>>>()
}

fn split_primary_for_f16(values: &[f16], k: u8, primary_min: u16) -> VortexResult<Vec<u16>> {
    vortex_ensure!(k > 0 && k <= 10, "FloatQuant f16 k must be in 1..=10");
    let low_mask = (1_u16 << k) - 1;
    vortex_ensure!(
        values.iter().all(|value| value.to_bits() & low_mask == 0),
        "FloatQuant constant secondary requires zero low bits"
    );
    values
        .iter()
        .map(|value| {
            (ordered_u16(value.to_bits()) >> k)
                .checked_sub(primary_min)
                .ok_or_else(|| vortex_error::vortex_err!("FloatQuant primary minimum is invalid"))
        })
        .collect::<VortexResult<Vec<_>>>()
}

fn split_primary_for_f64(values: &[f64], k: u8, primary_min: u64) -> VortexResult<Vec<u64>> {
    vortex_ensure!(k > 0 && k <= 52, "FloatQuant f64 k must be in 1..=52");
    let low_mask = (1_u64 << k) - 1;
    vortex_ensure!(
        values.iter().all(|value| value.to_bits() & low_mask == 0),
        "FloatQuant constant secondary requires zero low bits"
    );
    values
        .iter()
        .map(|value| {
            (ordered_u64(value.to_bits()) >> k)
                .checked_sub(primary_min)
                .ok_or_else(|| vortex_error::vortex_err!("FloatQuant primary minimum is invalid"))
        })
        .collect::<VortexResult<Vec<_>>>()
}

fn join_f16(primary: u16, secondary: u16, k: u8) -> f16 {
    let low_mask = (1_u16 << k) - 1;
    let sign_cutoff = (1_u16 << 15) >> k;
    let low = if primary >= sign_cutoff {
        secondary
    } else {
        low_mask.wrapping_sub(secondary)
    };
    let ordered = (primary << k).wrapping_add(low);
    let bits = if ordered & (1_u16 << 15) == 0 {
        !ordered
    } else {
        ordered ^ (1_u16 << 15)
    };
    f16::from_bits(bits)
}

fn join_f32(primary: u32, secondary: u32, k: u8) -> f32 {
    let low_mask = (1_u32 << k) - 1;
    let sign_cutoff = (1_u32 << 31) >> k;
    let low = if primary >= sign_cutoff {
        secondary
    } else {
        low_mask.wrapping_sub(secondary)
    };
    let ordered = (primary << k).wrapping_add(low);
    let bits = if ordered & (1_u32 << 31) == 0 {
        !ordered
    } else {
        ordered ^ (1_u32 << 31)
    };
    f32::from_bits(bits)
}

fn join_f64(primary: u64, secondary: u64, k: u8) -> f64 {
    let low_mask = (1_u64 << k) - 1;
    let sign_cutoff = (1_u64 << 63) >> k;
    let low = if primary >= sign_cutoff {
        secondary
    } else {
        low_mask.wrapping_sub(secondary)
    };
    let ordered = (primary << k).wrapping_add(low);
    let bits = if ordered & (1_u64 << 63) == 0 {
        !ordered
    } else {
        ordered ^ (1_u64 << 63)
    };
    f64::from_bits(bits)
}

fn join_zero_f32(primary: u32, k: u8) -> f32 {
    let low_mask = (1_u32 << k) - 1;
    let sign_cutoff = (1_u32 << 31) >> k;
    let low = if primary >= sign_cutoff { 0 } else { low_mask };
    let ordered = (primary << k).wrapping_add(low);
    let bits = if ordered & (1_u32 << 31) == 0 {
        !ordered
    } else {
        ordered ^ (1_u32 << 31)
    };
    f32::from_bits(bits)
}

fn join_zero_f16(primary: u16, k: u8) -> f16 {
    let low_mask = (1_u16 << k) - 1;
    let sign_cutoff = (1_u16 << 15) >> k;
    let low = if primary >= sign_cutoff { 0 } else { low_mask };
    let ordered = (primary << k).wrapping_add(low);
    let bits = if ordered & (1_u16 << 15) == 0 {
        !ordered
    } else {
        ordered ^ (1_u16 << 15)
    };
    f16::from_bits(bits)
}

fn join_zero_f64(primary: u64, k: u8) -> f64 {
    let low_mask = (1_u64 << k) - 1;
    let sign_cutoff = (1_u64 << 63) >> k;
    let low = if primary >= sign_cutoff { 0 } else { low_mask };
    let ordered = (primary << k).wrapping_add(low);
    let bits = if ordered & (1_u64 << 63) == 0 {
        !ordered
    } else {
        ordered ^ (1_u64 << 63)
    };
    f64::from_bits(bits)
}

fn decode(
    array: ArrayView<'_, FloatQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    if array.dtype().as_ptype() == PType::F64
        && let Some(decoded) = decode_fastlanes_zero(array, ctx)?
    {
        return Ok(decoded);
    }
    if let Some(decoded) = decode_fastlanes_pair(array)? {
        return Ok(decoded);
    }

    let primary = array.primary().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = primary.validity()?;
    let k = array.data().k;
    let Some(secondary) = array.secondary() else {
        return Ok(match PType::try_from(array.dtype())? {
            PType::F16 => PrimitiveArray::new(
                primary
                    .into_buffer::<u16>()
                    .map_each_in_place(|primary| join_zero_f16(primary, k))
                    .freeze(),
                validity,
            ),
            PType::F32 => PrimitiveArray::new(
                primary
                    .into_buffer::<u32>()
                    .map_each_in_place(|primary| join_zero_f32(primary, k))
                    .freeze(),
                validity,
            ),
            PType::F64 => PrimitiveArray::new(
                primary
                    .into_buffer::<u64>()
                    .map_each_in_place(|primary| join_zero_f64(primary, k))
                    .freeze(),
                validity,
            ),
            ptype => vortex_panic!("unsupported FloatQuant ptype {ptype}"),
        });
    };
    let secondary = secondary.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match PType::try_from(array.dtype())? {
        PType::F16 => {
            let secondary_values = secondary.as_slice::<u16>();
            let mut index = 0;
            let values = primary
                .into_buffer::<u16>()
                .map_each_in_place(|primary| {
                    let value = join_f16(primary, secondary_values[index], k);
                    index += 1;
                    value
                })
                .freeze();
            PrimitiveArray::new(values, validity)
        }
        PType::F32 => {
            let secondary_values = secondary.as_slice::<u32>();
            let mut index = 0;
            let values = primary
                .into_buffer::<u32>()
                .map_each_in_place(|primary| {
                    let value = join_f32(primary, secondary_values[index], k);
                    index += 1;
                    value
                })
                .freeze();
            PrimitiveArray::new(values, validity)
        }
        PType::F64 => {
            let secondary_values = secondary.as_slice::<u64>();
            let mut index = 0;
            let values = primary
                .into_buffer::<u64>()
                .map_each_in_place(|primary| {
                    let value = join_f64(primary, secondary_values[index], k);
                    index += 1;
                    value
                })
                .freeze();
            PrimitiveArray::new(values, validity)
        }
        ptype => vortex_panic!("unsupported FloatQuant ptype {ptype}"),
    })
}

fn decode_fastlanes_zero(
    array: ArrayView<'_, FloatQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<PrimitiveArray>> {
    if array.secondary().is_some() {
        return Ok(None);
    }
    let Some(primary_for) = array.primary().as_opt::<FoR>() else {
        return Ok(None);
    };
    let Some(primary) = primary_for.encoded().as_opt::<BitPacked>() else {
        return Ok(None);
    };

    let k = array.data().k;
    let reference = primary_for
        .reference_scalar()
        .as_primitive()
        .typed_value::<u64>()
        .vortex_expect("validated f64 primary reference");
    Ok(Some(unpack_map::<u64, f64, _>(primary, ctx, |primary| {
        join_zero_f64(primary.wrapping_add(reference), k)
    })?))
}

fn decode_fastlanes_pair(array: ArrayView<'_, FloatQuant>) -> VortexResult<Option<PrimitiveArray>> {
    let Some(secondary) = array
        .secondary()
        .and_then(|child| child.as_opt::<BitPacked>())
    else {
        return Ok(None);
    };
    let Some(primary_for) = array.primary().as_opt::<FoR>() else {
        return Ok(None);
    };
    let Some(primary) = primary_for.encoded().as_opt::<BitPacked>() else {
        return Ok(None);
    };
    if primary.patches().is_some()
        || secondary.patches().is_some()
        || primary.offset() != secondary.offset()
    {
        return Ok(None);
    }

    let validity = primary.validity()?;
    let k = array.data().k;
    Ok(Some(match PType::try_from(array.dtype())? {
        PType::F16 => {
            let reference = primary_for
                .reference_scalar()
                .as_primitive()
                .typed_value::<u16>()
                .vortex_expect("validated f16 primary reference");
            PrimitiveArray::new(
                unpack_pair_map::<u16, f16, _>(primary, secondary, |primary, secondary| {
                    join_f16(primary.wrapping_add(reference), secondary, k)
                })?,
                validity,
            )
        }
        PType::F32 => {
            let reference = primary_for
                .reference_scalar()
                .as_primitive()
                .typed_value::<u32>()
                .vortex_expect("validated f32 primary reference");
            PrimitiveArray::new(
                unpack_pair_map::<u32, f32, _>(primary, secondary, |primary, secondary| {
                    join_f32(primary.wrapping_add(reference), secondary, k)
                })?,
                validity,
            )
        }
        PType::F64 => {
            let reference = primary_for
                .reference_scalar()
                .as_primitive()
                .typed_value::<u64>()
                .vortex_expect("validated f64 primary reference");
            PrimitiveArray::new(
                unpack_pair_map::<u64, f64, _>(primary, secondary, |primary, secondary| {
                    join_f64(primary.wrapping_add(reference), secondary, k)
                })?,
                validity,
            )
        }
        ptype => vortex_panic!("unsupported FloatQuant ptype {ptype}"),
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn f16_bit_patterns_roundtrip() -> VortexResult<()> {
        let values = [
            f16::from_bits(0xfc00),
            f16::from_f32(-1.5),
            f16::NEG_ZERO,
            f16::ZERO,
            f16::from_f32(1.5),
            f16::INFINITY,
            f16::from_bits(0x7e34),
            f16::from_bits(0xfe56),
        ];
        let array = PrimitiveArray::from_iter(values);
        let encoded = FloatQuant::from_primitive(array.as_view(), 5)?;
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(
            decoded
                .as_slice::<f16>()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values.map(f16::to_bits)
        );
        Ok(())
    }

    #[test]
    fn fixed_tree_analysis_accounts_for_secondary_width() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter((0_u32..4096).map(|index| {
            let value = f64::from(f32::from_bits(0x3f80_0000 | index.wrapping_mul(7_919)));
            if index % 10 == 0 {
                f64::from_bits(value.to_bits() | 1)
            } else {
                value
            }
        }));
        let analysis = analyze_float_quant(values.as_view()).vortex_expect("FloatQuant input");
        assert_eq!(analysis.k, 29);
        assert_eq!(analysis.secondary_bit_width, 1);

        let general =
            PrimitiveArray::from_iter((0_u32..4096).map(|index| {
                f32::from_bits(0x3f80_0000 | (index.wrapping_mul(7_919) & 0x007f_ffff))
            }));
        assert_eq!(analyze_float_quant(general.as_view()), None);
        Ok(())
    }

    #[test]
    fn float_bit_patterns_roundtrip() -> VortexResult<()> {
        let values = [
            f64::NEG_INFINITY,
            -1.5,
            -0.0,
            0.0,
            1.5,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::from_bits(0xfff8_0000_0000_5678),
        ];
        let array = PrimitiveArray::from_iter(values);
        let encoded = FloatQuant::from_primitive(array.as_view(), 29)?;
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(
            decoded
                .as_slice::<f64>()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn f32_bit_patterns_roundtrip() -> VortexResult<()> {
        let values = [
            f32::NEG_INFINITY,
            -1.5,
            -0.0,
            0.0,
            1.5,
            f32::INFINITY,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_5678),
        ];
        let array = PrimitiveArray::from_iter(values);
        let encoded = FloatQuant::from_primitive(array.as_view(), 8)?;
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(
            decoded
                .as_slice::<f32>()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values.map(f32::to_bits)
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_split_shapes() {
        let primary = PrimitiveArray::from_iter([0_u32, 1, 2]).into_array();
        assert!(FloatQuant::try_new(primary.clone(), None, PType::F32, 0).is_err());
        assert!(FloatQuant::try_new(primary.clone(), None, PType::F32, 24).is_err());
        assert!(FloatQuant::try_new(primary.clone(), None, PType::U32, 8).is_err());

        let wrong_ptype = PrimitiveArray::from_iter([0_u64, 1, 2]).into_array();
        assert!(FloatQuant::try_new(primary.clone(), Some(wrong_ptype), PType::F32, 8).is_err());
        let nullable = PrimitiveArray::from_option_iter([Some(0_u32), None, Some(2)]).into_array();
        assert!(FloatQuant::try_new(primary, Some(nullable), PType::F32, 8).is_err());
    }

    #[test]
    fn nullable_slice_and_scalar_access() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            Buffer::from(vec![1.25_f32, 0.0, -0.0, 42.5, -10.0]),
            Validity::from_iter([true, false, true, true, false]),
        );
        let encoded = FloatQuant::from_primitive(array.as_view(), 8)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(encoded, array, &mut ctx);
        assert_nth_scalar!(encoded, 3, 42.5_f32, &mut ctx);
        assert!(encoded.execute_scalar(1, &mut ctx)?.is_null());

        let sliced = encoded.into_array().slice(1..4)?;
        assert!(sliced.is::<FloatQuant>());
        assert_arrays_eq!(sliced, array.into_array().slice(1..4)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn implicit_zero_secondary_roundtrip() -> VortexResult<()> {
        let original = PrimitiveArray::from_option_iter([
            Some(f64::from(-10.5_f32)),
            None,
            Some(f64::from(-0.0_f32)),
            Some(f64::from(42.25_f32)),
        ]);
        let encoded = FloatQuant::from_primitive_constant_secondary(original.as_view(), 29)?;
        assert!(encoded.secondary().is_none());

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(encoded, original, &mut ctx);
        assert_nth_scalar!(encoded, 3, 42.25_f64, &mut ctx);

        let sliced = encoded.into_array().slice(1..4)?;
        let expected = original.into_array().slice(1..4)?;
        assert_arrays_eq!(sliced, expected, &mut ctx);

        let dtype = sliced.dtype().clone();
        let len = sliced.len();
        let array_context = ArrayContext::empty();
        let serialized =
            sliced.serialize(&array_context, &SESSION, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(array_context.to_ids()),
            &SESSION,
        )?;
        assert!(decoded.as_::<FloatQuant>().secondary().is_none());
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn implicit_zero_secondary_rejects_nonzero_low_bits() {
        let f32_values = PrimitiveArray::from_iter([f32::from_bits(1.0_f32.to_bits() | 1)]);
        assert!(FloatQuant::from_primitive_constant_secondary(f32_values.as_view(), 8).is_err());
        assert!(FloatQuant::primary_for_primitive(f32_values.as_view(), 8, 0).is_err());

        let f64_values = PrimitiveArray::from_iter([f64::from_bits(1.0_f64.to_bits() | 1)]);
        assert!(FloatQuant::from_primitive_constant_secondary(f64_values.as_view(), 29).is_err());
        assert!(FloatQuant::primary_for_primitive(f64_values.as_view(), 29, 0).is_err());
    }

    #[test]
    fn primary_for_primitive_rejects_invalid_reference() {
        let values = PrimitiveArray::from_iter([1.0_f32, 2.0]);
        assert!(
            FloatQuant::primary_for_primitive(values.as_view(), 1, u64::from(u32::MAX)).is_err()
        );
    }

    #[test]
    fn fastlanes_zero_decode_roundtrip_and_slice() -> VortexResult<()> {
        let original = PrimitiveArray::from_option_iter((0_u32..4097).map(|index| {
            (index % 17 != 0).then(|| {
                let mantissa = index.wrapping_mul(7_919) & 0x007f_ffff;
                f64::from(f32::from_bits(0x3f80_0000 | mantissa))
            })
        }));
        let analysis = analyze_float_quant(original.as_view()).vortex_expect("FloatQuant input");
        assert_eq!(analysis.secondary_bit_width, 0);
        let primary = FloatQuant::primary_for_primitive(
            original.as_view(),
            analysis.k,
            analysis.primary_min,
        )?;
        // SAFETY: The analysis computes the exact primary width.
        let primary = unsafe {
            vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked(
                primary,
                analysis.primary_bit_width,
            )?
        };
        let primary = FoR::try_new(primary.into_array(), Scalar::from(analysis.primary_min))?;
        let encoded =
            FloatQuant::try_new(primary.into_array(), None, PType::F64, analysis.k)?.into_array();

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(encoded, original, &mut ctx);
        assert_arrays_eq!(
            encoded.slice(3..2051)?,
            original.into_array().slice(3..2051)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn fastlanes_pair_decode_roundtrip_and_slice() -> VortexResult<()> {
        let original = PrimitiveArray::from_iter((0_u32..4097).map(|index| {
            let mantissa = index.wrapping_mul(7_919) & 0x007f_ffff;
            let value = f64::from(f32::from_bits(0x3f80_0000 | mantissa));
            if index % 10 == 0 {
                f64::from_bits(value.to_bits() | 1)
            } else {
                value
            }
        }));
        let analysis = analyze_float_quant(original.as_view()).vortex_expect("FloatQuant input");
        assert_eq!(analysis.secondary_bit_width, 1);
        let split = FloatQuant::from_primitive(original.as_view(), analysis.k)?;
        let primary = split
            .primary()
            .clone()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        let secondary = split
            .secondary()
            .vortex_expect("nonzero secondary")
            .clone()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        let biased_primary = PrimitiveArray::from_iter(
            primary
                .as_slice::<u64>()
                .iter()
                .map(|value| value - analysis.primary_min),
        );
        // SAFETY: The analysis computes the exact primary width.
        let primary = unsafe {
            vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked(
                biased_primary,
                analysis.primary_bit_width,
            )?
        };
        let primary = FoR::try_new(primary.into_array(), Scalar::from(analysis.primary_min))?;
        // SAFETY: The analysis computes the exact secondary width.
        let secondary = unsafe {
            vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked(
                secondary,
                analysis.secondary_bit_width,
            )?
        };
        let encoded = FloatQuant::try_new(
            primary.into_array(),
            Some(secondary.into_array()),
            PType::F64,
            analysis.k,
        )?
        .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(encoded, original, &mut ctx);
        assert_arrays_eq!(
            encoded.slice(3..2051)?,
            original.into_array().slice(3..2051)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn serialization_roundtrip() -> VortexResult<()> {
        let original = PrimitiveArray::from_option_iter([
            Some(f64::NEG_INFINITY),
            None,
            Some(-0.0),
            Some(42.25),
            Some(f64::from_bits(0x7ff8_0000_0000_1234)),
        ]);
        let encoded = FloatQuant::from_primitive(original.as_view(), 29)?;
        let sliced = encoded.into_array().slice(1..5)?;
        let dtype = sliced.dtype().clone();
        let len = sliced.len();
        let array_context = ArrayContext::empty();
        let serialized =
            sliced.serialize(&array_context, &SESSION, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }

        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(array_context.to_ids()),
            &SESSION,
        )?;
        assert!(decoded.is::<FloatQuant>());
        assert_arrays_eq!(
            decoded,
            original.into_array().slice(1..5)?,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn conformance() -> VortexResult<()> {
        let explicit = PrimitiveArray::from_option_iter([
            Some(f32::NEG_INFINITY),
            None,
            Some(-0.0),
            Some(0.0),
            Some(42.25),
            Some(f32::INFINITY),
        ]);
        let explicit = FloatQuant::from_primitive(explicit.as_view(), 8)?.into_array();
        let implicit = PrimitiveArray::from_option_iter([
            Some(f64::from(-10.5_f32)),
            None,
            Some(f64::from(-0.0_f32)),
            Some(f64::from(42.25_f32)),
        ]);
        let implicit =
            FloatQuant::from_primitive_constant_secondary(implicit.as_view(), 29)?.into_array();
        let mut ctx = SESSION.create_execution_ctx();

        for array in [explicit, implicit] {
            test_array_consistency(&array, &mut ctx);
        }
        Ok(())
    }
}
