// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::mem::align_of;
use std::mem::size_of;
use std::mem::size_of_val;
use std::ptr;
use std::slice;
use std::sync::Arc;

use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::Extension;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::ListView;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::MapArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::decimal::DecimalArrayExt;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::listview::ListViewArrayExt;
use vortex::array::arrays::listview::ListViewArraySlotsExt;
use vortex::array::arrays::map::MapArrayExt;
use vortex::array::arrays::map::MapArraySlotsExt;
use vortex::array::arrays::primitive::PrimitiveArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::DecimalType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::PType;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TimeUnit;
use vortex::mask::Mask;
use vortex_array::ArrayView;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::FL_CHUNK_SIZE;

use super::*;
use crate::array::ArrowMemoryReservation;
use crate::array::conservative_export_reservation;
use crate::array::parse_memory_callbacks;
use crate::array::vx_velox_arrow_memory_callbacks;
use crate::ffi::try_or;
use crate::ffi::vx_array_ref;
use crate::ffi::vx_session_ref;
use crate::ffi::vx_velox_array;
use crate::ffi::vx_velox_error;
use crate::ffi::vx_velox_session;

fn primitive_type_id(value: PType) -> vx_velox_primitive_type {
    match value {
        PType::U8 => VX_VELOX_PRIMITIVE_U8,
        PType::U16 => VX_VELOX_PRIMITIVE_U16,
        PType::U32 => VX_VELOX_PRIMITIVE_U32,
        PType::U64 => VX_VELOX_PRIMITIVE_U64,
        PType::I8 => VX_VELOX_PRIMITIVE_I8,
        PType::I16 => VX_VELOX_PRIMITIVE_I16,
        PType::I32 => VX_VELOX_PRIMITIVE_I32,
        PType::I64 => VX_VELOX_PRIMITIVE_I64,
        PType::F16 => VX_VELOX_PRIMITIVE_F16,
        PType::F32 => VX_VELOX_PRIMITIVE_F32,
        PType::F64 => VX_VELOX_PRIMITIVE_F64,
    }
}

/// Retains one prepared Vortex array across several Velox output windows.
#[repr(C)]
pub struct vx_velox_export_cursor {
    export: CursorExport,
}

enum CursorExport {
    Primitive(PrimitiveExport),
    Bool(BoolExport),
    VarBin(VarBinExport),
    Dictionary(DictionaryExport),
    Constant(ConstantExport),
    Struct(StructExport),
    List(ListExport),
    Map(MapExport),
}

struct PackedBits(Box<[u64]>);

impl PackedBits {
    fn try_new(bits: vortex::buffer::BitBuffer) -> VortexResult<(Self, usize)> {
        let compact = bits
            .chunks()
            .iter_padded()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let allocation = size_of_val(compact.as_ref());
        Ok((Self(compact), allocation))
    }

    fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr().cast()
    }

    fn len(&self) -> usize {
        size_of_val(self.0.as_ref())
    }
}

struct BoolOwner {
    values: PackedBits,
    validity: Option<PackedBits>,
    retained_bytes: usize,
    memory_reservation: Option<ArrowMemoryReservation>,
}

impl BoolOwner {
    fn try_new(
        values: vortex::buffer::BitBuffer,
        validity: Option<vortex::buffer::BitBuffer>,
    ) -> VortexResult<Self> {
        let (values, values_allocation) = PackedBits::try_new(values)?;
        let (validity, validity_allocation) = match validity {
            Some(validity) => {
                let (validity, allocation) = PackedBits::try_new(validity)?;
                (Some(validity), allocation)
            }
            None => (None, 0),
        };
        let retained_bytes = values_allocation
            .checked_add(validity_allocation)
            .ok_or_else(|| vortex_err!("Boolean visitor retained byte count overflow"))?;
        Ok(Self {
            values,
            validity,
            retained_bytes,
            memory_reservation: None,
        })
    }

    fn set_memory_reservation(&mut self, reservation: ArrowMemoryReservation) {
        self.memory_reservation = Some(reservation);
    }
}

enum PrimitiveValues {
    Compact64(Box<[MaybeUninit<u64>]>),
    Compact128(Box<[MaybeUninit<u128>]>),
    Retained(ByteBuffer),
}

impl PrimitiveValues {
    fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Compact64(values) => values.as_ptr().cast(),
            Self::Compact128(values) => values.as_ptr().cast(),
            Self::Retained(values) => values.as_ptr(),
        }
    }
}

struct PrimitiveOwner {
    values: PrimitiveValues,
    values_length: usize,
    validity: Option<PackedBits>,
    retained_bytes: usize,
    memory_reservation: Option<ArrowMemoryReservation>,
}

enum RetainedBytes {
    Retained(ByteBuffer),
    Compact(Box<[u8]>),
}

impl RetainedBytes {
    fn try_new(handle: BufferHandle) -> VortexResult<(Self, usize)> {
        let buffer = handle.try_into_host_sync()?;
        let length = buffer.len();
        match buffer.try_into_mut() {
            Ok(buffer) => {
                let allocation_size = buffer.allocation_size();
                Ok((Self::Retained(buffer.freeze()), allocation_size))
            }
            Err(buffer) => {
                let compact = buffer.as_slice().to_vec().into_boxed_slice();
                Ok((Self::Compact(compact), length))
            }
        }
    }

    fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Retained(buffer) => buffer.as_ptr(),
            Self::Compact(buffer) => buffer.as_ptr(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Retained(buffer) => buffer.len(),
            Self::Compact(buffer) => buffer.len(),
        }
    }
}

enum RetainedViews {
    Retained(ByteBuffer),
    Compact(Box<[vx_velox_binary_view]>),
}

impl RetainedViews {
    fn try_new(handle: BufferHandle) -> VortexResult<(Self, usize)> {
        let buffer = handle.try_into_host_sync()?;
        if !buffer
            .len()
            .is_multiple_of(size_of::<vx_velox_binary_view>())
        {
            vortex_bail!(
                "Vortex variable-width view buffer has an invalid byte length: {}",
                buffer.len()
            );
        }
        match buffer.try_into_mut() {
            Ok(buffer) => {
                let allocation_size = buffer.allocation_size();
                Ok((Self::Retained(buffer.freeze()), allocation_size))
            }
            Err(buffer) => {
                let length = buffer.len() / size_of::<vx_velox_binary_view>();
                let mut compact = vec![
                    vx_velox_binary_view {
                        length: 0,
                        data: [0; 12],
                    };
                    length
                ]
                .into_boxed_slice();
                if !buffer.is_empty() {
                    // SAFETY: Both byte ranges have the checked identical size.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            buffer.as_ptr(),
                            compact.as_mut_ptr().cast::<u8>(),
                            buffer.len(),
                        )
                    };
                }
                let allocation = size_of_val(compact.as_ref());
                Ok((Self::Compact(compact), allocation))
            }
        }
    }

    fn as_ptr(&self) -> *const vx_velox_binary_view {
        match self {
            Self::Retained(buffer) => buffer.as_ptr().cast(),
            Self::Compact(buffer) => buffer.as_ptr(),
        }
    }
}

struct VarBinOwner {
    views: RetainedViews,
    _data: Box<[RetainedBytes]>,
    descriptors: Box<[vx_velox_byte_buffer_view]>,
    validity: Option<PackedBits>,
    retained_bytes: usize,
    memory_reservation: Option<ArrowMemoryReservation>,
}

// SAFETY: The owner never mutates its buffers or pointer descriptors after construction.
// Every descriptor points into an immutable allocation that the same owner retains.
unsafe impl Send for VarBinOwner {}
// SAFETY: Shared access only reads immutable buffers and descriptors retained by this owner.
unsafe impl Sync for VarBinOwner {}

impl VarBinOwner {
    fn try_new(
        views: BufferHandle,
        mut buffers: Arc<[BufferHandle]>,
        validity: Option<vortex::buffer::BitBuffer>,
        length: usize,
    ) -> VortexResult<Self> {
        let (views, views_allocation) = RetainedViews::try_new(views)?;
        let handles = if let Some(handles) = Arc::get_mut(&mut buffers) {
            handles
                .iter_mut()
                .map(|handle| {
                    std::mem::replace(handle, BufferHandle::new_host(ByteBuffer::empty()))
                })
                .collect::<Vec<_>>()
        } else {
            buffers.iter().cloned().collect::<Vec<_>>()
        };
        let mut data_allocation = 0usize;
        let data = handles
            .into_iter()
            .map(|handle| {
                let (buffer, allocation) = RetainedBytes::try_new(handle)?;
                data_allocation = data_allocation
                    .checked_add(allocation)
                    .ok_or_else(|| vortex_err!("Vortex string payload allocation overflow"))?;
                Ok(buffer)
            })
            .collect::<VortexResult<Vec<_>>>()?
            .into_boxed_slice();
        let descriptors = data
            .iter()
            .map(|buffer| vx_velox_byte_buffer_view {
                data: buffer.as_ptr(),
                length: buffer.len(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let descriptor_allocation = size_of_val(descriptors.as_ref());
        let (validity, validity_allocation) = retain_validity(validity, length)?;
        let retained_bytes = views_allocation
            .checked_add(data_allocation)
            .and_then(|bytes| bytes.checked_add(descriptor_allocation))
            .and_then(|bytes| bytes.checked_add(validity_allocation))
            .ok_or_else(|| vortex_err!("Vortex string retained byte count overflow"))?;
        Ok(Self {
            views,
            _data: data,
            descriptors,
            validity,
            retained_bytes,
            memory_reservation: None,
        })
    }

    fn set_memory_reservation(&mut self, reservation: ArrowMemoryReservation) {
        self.memory_reservation = Some(reservation);
    }
}

fn retain_validity(
    validity: Option<vortex::buffer::BitBuffer>,
    length: usize,
) -> VortexResult<(Option<PackedBits>, usize)> {
    let Some(validity) = validity else {
        return Ok((None, 0));
    };
    if validity.len() < length {
        vortex_bail!(
            "Vortex validity length is too small: {} for {length} values",
            validity.len()
        );
    }
    let validity = if validity.len() == length {
        validity
    } else {
        validity.slice(..length)
    };
    let (validity, allocation) = PackedBits::try_new(validity)?;
    Ok((Some(validity), allocation))
}

impl PrimitiveOwner {
    fn try_allocate(
        values_length: usize,
        values_alignment: usize,
        validity: Option<vortex::buffer::BitBuffer>,
        length: usize,
    ) -> VortexResult<Self> {
        let (values, values_allocation) = if values_alignment > align_of::<u64>() {
            if values_alignment > align_of::<u128>() {
                vortex_bail!(
                    "Primitive visitor does not support value alignment {values_alignment}"
                );
            }
            let values =
                vec![MaybeUninit::<u128>::uninit(); values_length.div_ceil(size_of::<u128>())]
                    .into_boxed_slice();
            let allocation = values
                .len()
                .checked_mul(size_of::<u128>())
                .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
            (PrimitiveValues::Compact128(values), allocation)
        } else {
            let values =
                vec![MaybeUninit::<u64>::uninit(); values_length.div_ceil(size_of::<u64>())]
                    .into_boxed_slice();
            let allocation = values
                .len()
                .checked_mul(size_of::<u64>())
                .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
            (PrimitiveValues::Compact64(values), allocation)
        };
        let (validity, validity_allocation) = retain_validity(validity, length)?;
        let retained_bytes = values_allocation
            .checked_add(validity_allocation)
            .ok_or_else(|| vortex_err!("Primitive visitor retained byte count overflow"))?;
        Ok(Self {
            values,
            values_length,
            validity,
            retained_bytes,
            memory_reservation: None,
        })
    }

    fn try_new(
        host_values: ByteBuffer,
        values_alignment: usize,
        validity: Option<vortex::buffer::BitBuffer>,
        length: usize,
        retain_values: bool,
    ) -> VortexResult<Self> {
        let values_length = host_values.len();
        let host_values = if retain_values {
            match host_values.try_into_mut() {
                Ok(values) => {
                    let values_allocation = values.allocation_size();
                    let (validity, validity_allocation) = retain_validity(validity, length)?;
                    let retained_bytes = values_allocation
                        .checked_add(validity_allocation)
                        .ok_or_else(|| {
                            vortex_err!("Primitive visitor retained byte count overflow")
                        })?;
                    return Ok(Self {
                        values: PrimitiveValues::Retained(values.freeze()),
                        values_length,
                        validity,
                        retained_bytes,
                        memory_reservation: None,
                    });
                }
                Err(values) => values,
            }
        } else {
            host_values
        };
        let mut owner = Self::try_allocate(values_length, values_alignment, validity, length)?;
        if !host_values.is_empty() {
            let (values_pointer, values_capacity) = match &mut owner.values {
                PrimitiveValues::Compact64(values) => (
                    values.as_mut_ptr().cast::<u8>(),
                    values.len() * size_of::<u64>(),
                ),
                PrimitiveValues::Compact128(values) => (
                    values.as_mut_ptr().cast::<u8>(),
                    values.len() * size_of::<u128>(),
                ),
                PrimitiveValues::Retained(_) => {
                    unreachable!("a newly allocated primitive owner must be compact")
                }
            };
            // SAFETY: The byte view spans the complete compact allocation.
            let values_bytes =
                unsafe { slice::from_raw_parts_mut(values_pointer, values_capacity) };
            values_bytes[..values_length].copy_from_slice(host_values.as_slice());
        }
        Ok(owner)
    }

    fn try_new_bitpacked_i64(
        array: ArrayView<'_, BitPacked>,
        validity: Option<vortex::buffer::BitBuffer>,
    ) -> VortexResult<Self> {
        let values_length = array
            .len()
            .checked_mul(size_of::<i64>())
            .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
        let mut owner =
            Self::try_allocate(values_length, align_of::<i64>(), validity, array.len())?;
        // SAFETY: The allocation uses `u64` alignment and contains at least `values_length` bytes.
        // The output slice covers exactly `array.len()` values and remains uniquely borrowed.
        let output = unsafe {
            slice::from_raw_parts_mut(
                match &mut owner.values {
                    PrimitiveValues::Compact64(values) => {
                        values.as_mut_ptr().cast::<MaybeUninit<i64>>()
                    }
                    PrimitiveValues::Compact128(_) | PrimitiveValues::Retained(_) => {
                        unreachable!("a newly allocated primitive owner must be compact")
                    }
                },
                array.len(),
            )
        };
        let mut scratch = [const { MaybeUninit::<i64>::uninit() }; FL_CHUNK_SIZE];
        array.unpacked_chunks(&mut scratch)?.decode_into(output);
        Ok(owner)
    }

    fn values(&self) -> *const u8 {
        if self.values_length == 0 {
            ptr::null()
        } else {
            self.values.as_ptr()
        }
    }

    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    fn set_memory_reservation(&mut self, reservation: ArrowMemoryReservation) {
        self.memory_reservation = Some(reservation);
    }
}

fn pointer_alignment(pointer: *const u8) -> usize {
    if pointer.is_null() {
        return 0;
    }
    1usize << pointer.addr().trailing_zeros()
}

fn primitive_width(primitive_type: vx_velox_primitive_type) -> VortexResult<usize> {
    Ok(match primitive_type {
        VX_VELOX_PRIMITIVE_U8 | VX_VELOX_PRIMITIVE_I8 => 1,
        VX_VELOX_PRIMITIVE_U16 | VX_VELOX_PRIMITIVE_I16 | VX_VELOX_PRIMITIVE_F16 => 2,
        VX_VELOX_PRIMITIVE_U32 | VX_VELOX_PRIMITIVE_I32 | VX_VELOX_PRIMITIVE_F32 => 4,
        VX_VELOX_PRIMITIVE_U64 | VX_VELOX_PRIMITIVE_I64 | VX_VELOX_PRIMITIVE_F64 => 8,
        VX_VELOX_PRIMITIVE_I128 => 16,
        _ => vortex_bail!("Unknown Vortex Velox primitive type: {primitive_type}"),
    })
}

fn cast_decimal_values<T, S>(values: Buffer<S>, validity: &Mask) -> VortexResult<ByteBuffer>
where
    T: NativeDecimalType,
    S: NativeDecimalType,
{
    let mut output = BufferMut::<T>::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        if !validity.value(index) {
            output.push(T::default());
            continue;
        }
        output.push(<T as vortex::dtype::BigCast>::from(value).ok_or_else(|| {
            vortex_err!(
                "Decimal value cannot be represented as {}",
                std::any::type_name::<T>()
            )
        })?);
    }
    Ok(output.freeze().into_byte_buffer())
}

fn normalized_decimal_values<T>(array: &DecimalArray, validity: &Mask) -> VortexResult<ByteBuffer>
where
    T: NativeDecimalType,
{
    if array.values_type() == T::DECIMAL_TYPE {
        return array.buffer_handle().clone().try_into_host_sync();
    }
    match array.values_type() {
        DecimalType::I8 => cast_decimal_values::<T, i8>(array.buffer::<i8>(), validity),
        DecimalType::I16 => cast_decimal_values::<T, i16>(array.buffer::<i16>(), validity),
        DecimalType::I32 => cast_decimal_values::<T, i32>(array.buffer::<i32>(), validity),
        DecimalType::I64 => cast_decimal_values::<T, i64>(array.buffer::<i64>(), validity),
        DecimalType::I128 => cast_decimal_values::<T, i128>(array.buffer::<i128>(), validity),
        DecimalType::I256 => cast_decimal_values::<T, vortex::dtype::i256>(
            array.buffer::<vortex::dtype::i256>(),
            validity,
        ),
    }
}

struct PrimitiveExport {
    primitive_type: vx_velox_primitive_type,
    decimal_precision: u32,
    decimal_scale: i32,
    length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<PrimitiveOwner>,
}

impl PrimitiveExport {
    fn try_new_decimal(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let retain_values = memory_callbacks.is_some();
        let mut execution = session.create_execution_ctx();
        let mut memory_reservation = match memory_callbacks {
            Some(callbacks) => Some(ArrowMemoryReservation::try_new(
                callbacks,
                conservative_export_reservation(&array, &mut execution)?,
            )?),
            None => None,
        };
        let is_nullable = array.dtype().is_nullable();
        let decimal = array.execute::<DecimalArray>(&mut execution)?;
        let decimal_precision = u32::from(decimal.precision());
        let decimal_scale = i32::from(decimal.scale());
        let length = decimal.len();
        let mask = decimal
            .as_ref()
            .validity()?
            .execute_mask(length, &mut execution)?;
        let (primitive_type, host_values) = match decimal.precision() {
            1..=18 => (
                VX_VELOX_PRIMITIVE_I64,
                normalized_decimal_values::<i64>(&decimal, &mask)?,
            ),
            19..=38 => (
                VX_VELOX_PRIMITIVE_I128,
                normalized_decimal_values::<i128>(&decimal, &mask)?,
            ),
            precision => {
                vortex_bail!("Vortex Velox visitor does not support decimal precision {precision}")
            }
        };
        let (validity_kind, validity) = exported_validity(is_nullable, mask);
        let mut owner = PrimitiveOwner::try_new(
            host_values,
            primitive_width(primitive_type)?,
            validity,
            length,
            retain_values,
        )?;
        if let Some(mut reservation) = memory_reservation.take() {
            reservation.reconcile(owner.retained_bytes())?;
            owner.set_memory_reservation(reservation);
        }
        Ok(Self {
            primitive_type,
            decimal_precision,
            decimal_scale,
            length,
            validity_kind,
            owner: Arc::new(owner),
        })
    }

    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let retain_values = memory_callbacks.is_some();
        let direct_bitpacked = array.as_opt::<BitPacked>().filter(|bitpacked| {
            array.dtype().as_ptype() == PType::I64 && bitpacked.patches().is_none()
        });
        let values_length =
            array
                .len()
                .checked_mul(array.dtype().element_size().ok_or_else(|| {
                    vortex_err!("Primitive visitor received a variable-width array")
                })?)
                .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
        let values_allocation = values_length
            .checked_add(size_of::<u64>() - 1)
            .ok_or_else(|| vortex_err!("Primitive visitor value allocation overflow"))?
            / size_of::<u64>()
            * size_of::<u64>();
        let validity_allocation = if array.dtype().is_nullable() {
            array
                .len()
                .div_ceil(u64::BITS as usize)
                .checked_mul(size_of::<u64>())
                .ok_or_else(|| vortex_err!("Primitive visitor validity allocation overflow"))?
        } else {
            0
        };
        let peak_reservation =
            if direct_bitpacked.is_some() {
                values_allocation.checked_add(validity_allocation.checked_mul(2).ok_or_else(
                    || vortex_err!("Primitive visitor validity reservation overflow"),
                )?)
            } else {
                values_allocation
                    .checked_add(validity_allocation)
                    .and_then(|bytes| bytes.checked_mul(2))
            }
            .ok_or_else(|| vortex_err!("Primitive visitor memory reservation overflow"))?;
        let mut memory_reservation = match (memory_callbacks, peak_reservation) {
            (Some(callbacks), bytes) if bytes != 0 => {
                Some(ArrowMemoryReservation::try_new(callbacks, bytes)?)
            }
            _ => None,
        };

        let mut execution = session.create_execution_ctx();
        let (primitive_type, length, validity_kind, mut owner) = if let Some(bitpacked) =
            direct_bitpacked
        {
            let primitive_type = primitive_type_id(array.dtype().as_ptype());
            let length = array.len();
            let mask = bitpacked.validity()?.execute_mask(length, &mut execution)?;
            let (validity_kind, validity) = exported_validity(array.dtype().is_nullable(), mask);
            let owner = PrimitiveOwner::try_new_bitpacked_i64(bitpacked, validity)?;
            (primitive_type, length, validity_kind, owner)
        } else {
            let Canonical::Primitive(primitive) = array.execute::<Canonical>(&mut execution)?
            else {
                vortex_bail!("Primitive visitor received a non-primitive array");
            };
            let primitive_type = primitive_type_id(primitive.ptype());
            let length = primitive.len();
            let mask = primitive.validity()?.execute_mask(length, &mut execution)?;
            let (validity_kind, validity) =
                exported_validity(primitive.dtype().is_nullable(), mask);
            let host_values = primitive.into_data_parts().buffer.try_into_host_sync()?;
            let owner = PrimitiveOwner::try_new(
                host_values,
                primitive_width(primitive_type)?,
                validity,
                length,
                retain_values,
            )?;
            (primitive_type, length, validity_kind, owner)
        };
        if let Some(mut reservation) = memory_reservation.take() {
            reservation.reconcile(owner.retained_bytes())?;
            owner.set_memory_reservation(reservation);
        }
        Ok(Self {
            primitive_type,
            decimal_precision: 0,
            decimal_scale: 0,
            length,
            validity_kind,
            owner: Arc::new(owner),
        })
    }

    fn view(&self, offset: usize, length: usize) -> VortexResult<vx_velox_primitive_view> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let width = primitive_width(self.primitive_type)?;
        let byte_offset = offset
            .checked_mul(width)
            .ok_or_else(|| vortex_err!("Vortex Velox value offset overflow"))?;
        let values_length = length
            .checked_mul(width)
            .ok_or_else(|| vortex_err!("Vortex Velox value length overflow"))?;
        let values = if values_length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked export range lies within the retained primitive buffer.
            unsafe { self.owner.values().add(byte_offset) }
        };
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("Primitive validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        Ok(vx_velox_primitive_view {
            struct_size: size_of::<vx_velox_primitive_view>(),
            primitive_type: self.primitive_type,
            decimal_precision: self.decimal_precision,
            decimal_scale: self.decimal_scale,
            length,
            values,
            values_length,
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_primitive_owner),
                release: Some(release_primitive_owner),
                retained_bytes: self.owner.retained_bytes(),
            },
            values_alignment: pointer_alignment(values),
            validity_alignment: pointer_alignment(validity),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let view = self.view(offset, length)?;
        let callback = visitor
            .visit_primitive
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a primitive callback"))?;
        // SAFETY: The cursor retains every buffer in the view through this callback.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct BoolExport {
    length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<BoolOwner>,
}

impl BoolExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let mut execution = session.create_execution_ctx();
        let mut memory_reservation = match memory_callbacks {
            Some(callbacks) => Some(ArrowMemoryReservation::try_new(
                callbacks,
                conservative_export_reservation(&array, &mut execution)?,
            )?),
            None => None,
        };
        let is_nullable = array.dtype().is_nullable();
        let Canonical::Bool(boolean) = array.execute::<Canonical>(&mut execution)? else {
            vortex_bail!("Boolean visitor received a non-Boolean array");
        };
        let length = boolean.len();
        let mask = boolean.validity()?.execute_mask(length, &mut execution)?;
        let (validity_kind, validity) = exported_validity(is_nullable, mask);
        let mut owner = BoolOwner::try_new(boolean.into_bit_buffer(), validity)?;
        if let Some(mut reservation) = memory_reservation.take() {
            reservation.reconcile(owner.retained_bytes)?;
            owner.set_memory_reservation(reservation);
        }
        Ok(Self {
            length,
            validity_kind,
            owner: Arc::new(owner),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let (values, values_length, values_bit_offset) =
            packed_bits_window(&self.owner.values, offset, length)?;
        let (validity, validity_length, validity_bit_offset) = match &self.owner.validity {
            Some(validity) => packed_bits_window(validity, offset, length)?,
            None => (ptr::null(), 0, 0),
        };
        let view = vx_velox_bool_view {
            struct_size: size_of::<vx_velox_bool_view>(),
            length,
            values,
            values_length,
            values_bit_offset,
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_bool_owner),
                release: Some(release_bool_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            values_alignment: pointer_alignment(values),
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor
            .visit_bool
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a Boolean callback"))?;
        // SAFETY: The cursor retains every buffer in the view through this callback.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

fn packed_bits_window(
    bits: &PackedBits,
    offset: usize,
    length: usize,
) -> VortexResult<(*const u8, usize, usize)> {
    if length == 0 {
        return Ok((ptr::null(), 0, 0));
    }
    let word_bits = u64::BITS as usize;
    let byte_offset = offset / word_bits * size_of::<u64>();
    let bit_offset = offset % word_bits;
    let required_length = bit_offset
        .checked_add(length)
        .ok_or_else(|| vortex_err!("Packed Boolean window overflow"))?
        .div_ceil(u8::BITS as usize);
    let byte_length = bits
        .len()
        .checked_sub(byte_offset)
        .ok_or_else(|| vortex_err!("Packed Boolean window exceeds its owner"))?;
    if byte_length < required_length {
        vortex_bail!("Packed Boolean window exceeds its readable bytes");
    }
    // SAFETY: The caller validated the logical window against the owner length.
    let values = unsafe { bits.as_ptr().add(byte_offset) };
    Ok((values, byte_length, bit_offset))
}

struct VarBinExport {
    kind: vx_velox_varbin_kind,
    length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<VarBinOwner>,
}

impl VarBinExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let mut execution = session.create_execution_ctx();
        let mut memory_reservation = match memory_callbacks {
            Some(callbacks) => Some(ArrowMemoryReservation::try_new(
                callbacks,
                conservative_export_reservation(&array, &mut execution)?,
            )?),
            None => None,
        };
        let is_nullable = array.dtype().is_nullable();
        let varbin = array.execute::<VarBinViewArray>(&mut execution)?;
        let length = varbin.len();
        let parts = varbin.into_data_parts();
        let kind = match parts.dtype {
            DType::Utf8(_) => VX_VELOX_VARBIN_UTF8,
            DType::Binary(_) => VX_VELOX_VARBIN_BINARY,
            dtype => vortex_bail!("Variable-width visitor received an invalid type: {dtype}"),
        };
        let mask = parts.validity.execute_mask(length, &mut execution)?;
        let (validity_kind, validity) = exported_validity(is_nullable, mask);
        let mut owner = VarBinOwner::try_new(parts.views, parts.buffers, validity, length)?;
        if let Some(mut reservation) = memory_reservation.take() {
            reservation.reconcile(owner.retained_bytes)?;
            owner.set_memory_reservation(reservation);
        }
        Ok(Self {
            kind,
            length,
            validity_kind,
            owner: Arc::new(owner),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let view_byte_offset = offset
            .checked_mul(size_of::<vx_velox_binary_view>())
            .ok_or_else(|| vortex_err!("Vortex string view offset overflow"))?;
        let views_length = length
            .checked_mul(size_of::<vx_velox_binary_view>())
            .ok_or_else(|| vortex_err!("Vortex string view length overflow"))?;
        let views = if views_length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked export range lies within the retained view buffer.
            unsafe {
                self.owner
                    .views
                    .as_ptr()
                    .cast::<u8>()
                    .add(view_byte_offset)
                    .cast()
            }
        };
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("String validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        let data_buffers = if self.owner.descriptors.is_empty() {
            ptr::null()
        } else {
            self.owner.descriptors.as_ptr()
        };
        let view = vx_velox_varbin_view {
            struct_size: size_of::<vx_velox_varbin_view>(),
            kind: self.kind,
            length,
            views,
            views_length,
            data_buffers,
            data_buffer_count: self.owner.descriptors.len(),
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_varbin_owner),
                release: Some(release_varbin_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            views_alignment: pointer_alignment(views.cast()),
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor.visit_varbin.ok_or_else(|| {
            vortex_err!("Vortex Velox visitor requires a variable-width callback")
        })?;
        // SAFETY: The cursor retains every buffer in the view through this callback.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct DictionaryExport {
    codes: PrimitiveExport,
    values_length: usize,
    values: Box<vx_velox_export_cursor>,
}

impl DictionaryExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let dictionary = array.as_::<Dict>();
        let values = dictionary.values().clone();
        Ok(Self {
            codes: PrimitiveExport::try_new(dictionary.codes().clone(), session, memory_callbacks)?,
            values_length: values.len(),
            values: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new_canonical(values, session, memory_callbacks)?,
            }),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let codes = self.codes.view(offset, length)?;
        let view = vx_velox_dictionary_view {
            struct_size: size_of::<vx_velox_dictionary_view>(),
            length,
            codes,
            values: &raw const *self.values,
            values_length: self.values_length,
        };
        let callback = visitor
            .visit_dictionary
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a dictionary callback"))?;
        // SAFETY: The borrowed child cursor and every code buffer remain live through this call.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct ConstantExport {
    length: usize,
    value: Box<vx_velox_export_cursor>,
}

impl ConstantExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let length = array.len();
        let scalar = array.as_::<Constant>().scalar().clone();
        let value = ConstantArray::new(scalar, 1).into_array();
        Ok(Self {
            length,
            value: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new_canonical(value, session, memory_callbacks)?,
            }),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let view = vx_velox_constant_view {
            struct_size: size_of::<vx_velox_constant_view>(),
            length,
            value: &raw const *self.value,
        };
        let callback = visitor
            .visit_constant
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a constant callback"))?;
        // SAFETY: The borrowed child cursor remains live through this call.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct StructOwner {
    validity: Option<PackedBits>,
    retained_bytes: usize,
    _memory_reservation: Option<ArrowMemoryReservation>,
}

struct StructExport {
    length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<StructOwner>,
    fields: Box<[vx_velox_export_cursor]>,
    field_pointers: Box<[*const vx_velox_export_cursor]>,
}

impl StructExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let is_nullable = array.dtype().is_nullable();
        let mut execution = session.create_execution_ctx();
        let struct_array = array.execute::<StructArray>(&mut execution)?;
        let length = struct_array.len();
        let mask = struct_array
            .struct_validity()
            .execute_mask(length, &mut execution)?;
        let validity_reservation = if matches!(mask, Mask::Values(_)) {
            length
                .div_ceil(u64::BITS as usize)
                .checked_mul(size_of::<u64>())
                .ok_or_else(|| vortex_err!("Struct validity reservation overflow"))?
        } else {
            0
        };
        let mut memory_reservation = match (memory_callbacks, validity_reservation) {
            (Some(callbacks), bytes) if bytes != 0 => {
                Some(ArrowMemoryReservation::try_new(callbacks, bytes)?)
            }
            _ => None,
        };
        let (validity_kind, validity) = exported_validity(is_nullable, mask);
        let (validity, retained_bytes) = retain_validity(validity, length)?;
        if let Some(reservation) = memory_reservation.as_mut() {
            reservation.reconcile(retained_bytes)?;
        }
        let owner = Arc::new(StructOwner {
            validity,
            retained_bytes,
            _memory_reservation: memory_reservation,
        });
        let fields = struct_array
            .iter_unmasked_fields()
            .map(|field| {
                Ok(vx_velox_export_cursor {
                    export: CursorExport::try_new(field.clone(), session, memory_callbacks)?,
                })
            })
            .collect::<VortexResult<Vec<_>>>()?
            .into_boxed_slice();
        let field_pointers = fields
            .iter()
            .map(|field| field as *const vx_velox_export_cursor)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            length,
            validity_kind,
            owner,
            fields,
            field_pointers,
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("Struct validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        let view = vx_velox_struct_view {
            struct_size: size_of::<vx_velox_struct_view>(),
            length,
            offset,
            fields: if self.field_pointers.is_empty() {
                ptr::null()
            } else {
                self.field_pointers.as_ptr()
            },
            field_count: self.fields.len(),
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_struct_owner),
                release: Some(release_struct_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor
            .visit_struct
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a struct callback"))?;
        // SAFETY: The borrowed field cursors and parent validity remain live through this call.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct ListOwner {
    offsets: Box<[i32]>,
    sizes: Box<[i32]>,
    validity: Option<PackedBits>,
    retained_bytes: usize,
    _memory_reservation: Option<ArrowMemoryReservation>,
}

struct ListMetadata {
    length: usize,
    elements_length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<ListOwner>,
}

struct ListExport {
    length: usize,
    elements_length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<ListOwner>,
    elements: Box<vx_velox_export_cursor>,
}

fn list_metadata_value<T>(value: T, name: &str) -> VortexResult<i32>
where
    T: Copy + std::fmt::Display,
    i32: TryFrom<T>,
{
    i32::try_from(value)
        .map_err(|_| vortex_err!("Vortex list {name} exceeds the Velox vector limit: {value}"))
}

fn list_metadata_values(values: PrimitiveArray, name: &str) -> VortexResult<Box<[i32]>> {
    let values = values.reinterpret_cast(values.ptype().to_unsigned());
    match_each_unsigned_integer_ptype!(values.ptype(), |P| {
        values
            .as_slice::<P>()
            .iter()
            .map(|&value| list_metadata_value(value, name))
            .collect::<VortexResult<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    })
}

fn prepare_list_metadata(
    list: &ListViewArray,
    session: &vortex::session::VortexSession,
    memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
) -> VortexResult<ListMetadata> {
    let is_nullable = list.dtype().is_nullable();
    let mut execution = session.create_execution_ctx();
    let length = list.len();
    let elements_length = list.elements().len();
    if elements_length > i32::MAX as usize {
        vortex_bail!("Vortex list elements exceed the Velox vector limit: {elements_length}");
    }
    let mask = list
        .listview_validity()
        .execute_mask(length, &mut execution)?;
    let validity_reservation = if matches!(mask, Mask::Values(_)) {
        length
            .div_ceil(u64::BITS as usize)
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| vortex_err!("List validity reservation overflow"))?
    } else {
        0
    };
    let metadata_reservation = length
        .checked_mul(2 * size_of::<i32>())
        .ok_or_else(|| vortex_err!("List metadata reservation overflow"))?;
    let reservation = metadata_reservation
        .checked_add(validity_reservation)
        .ok_or_else(|| vortex_err!("List retained byte count overflow"))?;
    let mut memory_reservation = match (memory_callbacks, reservation) {
        (Some(callbacks), bytes) if bytes != 0 => {
            Some(ArrowMemoryReservation::try_new(callbacks, bytes)?)
        }
        _ => None,
    };
    let offsets = list_metadata_values(
        list.offsets()
            .clone()
            .execute::<PrimitiveArray>(&mut execution)?,
        "offset",
    )?;
    let sizes = list_metadata_values(
        list.sizes()
            .clone()
            .execute::<PrimitiveArray>(&mut execution)?,
        "size",
    )?;
    let (validity_kind, validity) = exported_validity(is_nullable, mask);
    let (validity, validity_allocation) = retain_validity(validity, length)?;
    let retained_bytes = size_of_val(offsets.as_ref())
        .checked_add(size_of_val(sizes.as_ref()))
        .and_then(|bytes| bytes.checked_add(validity_allocation))
        .ok_or_else(|| vortex_err!("List retained byte count overflow"))?;
    if let Some(reservation) = memory_reservation.as_mut() {
        reservation.reconcile(retained_bytes)?;
    }
    Ok(ListMetadata {
        length,
        elements_length,
        validity_kind,
        owner: Arc::new(ListOwner {
            offsets,
            sizes,
            validity,
            retained_bytes,
            _memory_reservation: memory_reservation,
        }),
    })
}

impl ListExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let mut execution = session.create_execution_ctx();
        let list = array.execute::<ListViewArray>(&mut execution)?;
        let elements = list.elements().clone();
        let metadata = prepare_list_metadata(&list, session, memory_callbacks)?;
        Ok(Self {
            length: metadata.length,
            elements_length: metadata.elements_length,
            validity_kind: metadata.validity_kind,
            owner: metadata.owner,
            elements: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new(elements, session, memory_callbacks)?,
            }),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("List validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        let offsets = if length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked range lies within the metadata arrays.
            unsafe { self.owner.offsets.as_ptr().add(offset) }
        };
        let sizes = if length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked range lies within the metadata arrays.
            unsafe { self.owner.sizes.as_ptr().add(offset) }
        };
        let view = vx_velox_list_view {
            struct_size: size_of::<vx_velox_list_view>(),
            length,
            offsets,
            sizes,
            elements: &raw const *self.elements,
            elements_length: self.elements_length,
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_list_owner),
                release: Some(release_list_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            offsets_alignment: pointer_alignment(offsets.cast()),
            sizes_alignment: pointer_alignment(sizes.cast()),
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor
            .visit_list
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a list callback"))?;
        // SAFETY: The borrowed element cursor and parent buffers remain live through this call.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct MapExport {
    length: usize,
    entries_length: usize,
    keys_sorted: bool,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<ListOwner>,
    keys: Box<vx_velox_export_cursor>,
    values: Box<vx_velox_export_cursor>,
}

impl MapExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let mut execution = session.create_execution_ctx();
        let map = array.execute::<MapArray>(&mut execution)?;
        let keys_sorted = map.keys_sorted();
        let entries = map.entries().clone().downcast::<ListView>();
        let entry_values = entries.elements().clone();
        let entry_struct = entry_values.execute::<StructArray>(&mut execution)?;
        let fields = entry_struct.iter_unmasked_fields().collect::<Vec<_>>();
        if fields.len() != 2 {
            vortex_bail!(
                "Vortex map entries require two fields, got {}",
                fields.len()
            );
        }
        let metadata = prepare_list_metadata(&entries, session, memory_callbacks)?;
        Ok(Self {
            length: metadata.length,
            entries_length: metadata.elements_length,
            keys_sorted,
            validity_kind: metadata.validity_kind,
            owner: metadata.owner,
            keys: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new(fields[0].clone(), session, memory_callbacks)?,
            }),
            values: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new(fields[1].clone(), session, memory_callbacks)?,
            }),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("Map validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        let offsets = if length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked range lies within the metadata arrays.
            unsafe { self.owner.offsets.as_ptr().add(offset) }
        };
        let sizes = if length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked range lies within the metadata arrays.
            unsafe { self.owner.sizes.as_ptr().add(offset) }
        };
        let view = vx_velox_map_view {
            struct_size: size_of::<vx_velox_map_view>(),
            length,
            offsets,
            sizes,
            keys: &raw const *self.keys,
            values: &raw const *self.values,
            entries_length: self.entries_length,
            keys_sorted: self.keys_sorted,
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_list_owner),
                release: Some(release_list_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            offsets_alignment: pointer_alignment(offsets.cast()),
            sizes_alignment: pointer_alignment(sizes.cast()),
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor
            .visit_map
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a map callback"))?;
        // SAFETY: The borrowed child cursors and parent buffers remain live through this callback.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

impl CursorExport {
    fn date_storage(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
    ) -> VortexResult<Option<vortex::array::ArrayRef>> {
        let DType::Extension(ext_dtype) = array.dtype() else {
            return Ok(None);
        };
        let Some(time_unit) = ext_dtype.metadata_opt::<Date>() else {
            return Ok(None);
        };
        if *time_unit != TimeUnit::Days {
            vortex_bail!(
                "Vortex Velox visitor does not support date unit {time_unit}; Velox DATE uses days"
            );
        }

        if let Some(extension) = array.as_opt::<Extension>() {
            return Ok(Some(extension.storage_array().clone()));
        }
        let mut execution = session.create_execution_ctx();
        let extension = array.execute::<ExtensionArray>(&mut execution)?;
        Ok(Some(extension.storage_array().clone()))
    }

    fn try_new_canonical(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        if matches!(array.dtype(), DType::Map(..)) {
            Ok(Self::Map(MapExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::List(..)) {
            Ok(Self::List(ListExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::Struct(..)) {
            Ok(Self::Struct(StructExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::Decimal(..)) {
            Ok(Self::Primitive(PrimitiveExport::try_new_decimal(
                array,
                session,
                memory_callbacks,
            )?))
        } else if let Some(storage) = Self::date_storage(array.clone(), session)? {
            Ok(Self::Primitive(PrimitiveExport::try_new(
                storage,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::Bool(_)) {
            Ok(Self::Bool(BoolExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::Utf8(_) | DType::Binary(_)) {
            Ok(Self::VarBin(VarBinExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else {
            Ok(Self::Primitive(PrimitiveExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        }
    }

    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        if array.is::<Dict>() {
            Ok(Self::Dictionary(DictionaryExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if array.is::<Constant>() {
            Ok(Self::Constant(ConstantExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else {
            Self::try_new_canonical(array, session, memory_callbacks)
        }
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        match self {
            Self::Primitive(export) => export.visit(offset, length, visitor),
            Self::Bool(export) => export.visit(offset, length, visitor),
            Self::VarBin(export) => export.visit(offset, length, visitor),
            Self::Dictionary(export) => export.visit(offset, length, visitor),
            Self::Constant(export) => export.visit(offset, length, visitor),
            Self::Struct(export) => export.visit(offset, length, visitor),
            Self::List(export) => export.visit(offset, length, visitor),
            Self::Map(export) => export.visit(offset, length, visitor),
        }
    }
}

fn exported_validity(
    is_nullable: bool,
    mask: Mask,
) -> (vx_velox_validity_kind, Option<vortex::buffer::BitBuffer>) {
    if !is_nullable {
        return (VX_VELOX_VALIDITY_NON_NULLABLE, None);
    }
    match mask {
        Mask::AllTrue(_) => (VX_VELOX_VALIDITY_ALL_VALID, None),
        Mask::AllFalse(_) => (VX_VELOX_VALIDITY_ALL_INVALID, None),
        Mask::Values(values) => (VX_VELOX_VALIDITY_BITMAP, Some(values.bit_buffer().clone())),
    }
}

unsafe extern "C" fn retain_primitive_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<PrimitiveOwner>()) };
}

unsafe extern "C" fn release_primitive_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<PrimitiveOwner>()) });
}

unsafe extern "C" fn retain_bool_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<BoolOwner>()) };
}

unsafe extern "C" fn release_bool_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<BoolOwner>()) });
}

unsafe extern "C" fn retain_varbin_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<VarBinOwner>()) };
}

unsafe extern "C" fn release_varbin_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<VarBinOwner>()) });
}

unsafe extern "C" fn retain_struct_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<StructOwner>()) };
}

unsafe extern "C" fn release_struct_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<StructOwner>()) });
}

unsafe extern "C" fn retain_list_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<ListOwner>()) };
}

unsafe extern "C" fn release_list_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<ListOwner>()) });
}

fn validate_visitor(visitor: &vx_velox_visitor) -> VortexResult<()> {
    if visitor.struct_size < size_of::<vx_velox_visitor>() {
        vortex_bail!(
            "Vortex Velox visitor structure is too small: expected at least {}, got {}",
            size_of::<vx_velox_visitor>(),
            visitor.struct_size
        );
    }
    if visitor.abi_version != crate::VX_VELOX_ABI_VERSION {
        vortex_bail!(
            "Unsupported Vortex Velox ABI version: expected {}, got {}",
            crate::VX_VELOX_ABI_VERSION,
            visitor.abi_version
        );
    }
    Ok(())
}

fn callback_error(visitor: &vx_velox_visitor, status: i32) -> String {
    let Some(last_error) = visitor.last_error else {
        return format!("Velox visitor failed with status {status}");
    };
    // SAFETY: The callback contract returns null or a valid null-terminated string.
    let message = unsafe { last_error(visitor.context) };
    if message.is_null() {
        return format!("Velox visitor failed with status {status}");
    }
    // SAFETY: The callback keeps the string valid until the next callback.
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

fn selected_array(
    array: &vortex::array::ArrayRef,
    request: &vx_velox_visit_request,
) -> VortexResult<vortex::array::ArrayRef> {
    if request.rows.is_null() {
        if request.row_count != 0 {
            vortex_bail!("A null visitor row pointer requires a zero row count");
        }
        return Ok(array.clone());
    }
    // SAFETY: The caller supplies `row_count` readable positions.
    let rows = unsafe { slice::from_raw_parts(request.rows, request.row_count) };
    let mut previous = None;
    for row in rows {
        let position = usize::try_from(*row)
            .map_err(|_| vortex_err!("Visitor row does not fit usize: {}", row))?;
        if position >= array.len() {
            vortex_bail!(
                "Visitor row is out of bounds: row {}, array length {}",
                row,
                array.len()
            );
        }
        if previous.is_some_and(|previous| previous >= *row) {
            vortex_bail!("Visitor rows must be unique and increasing");
        }
        previous = Some(*row);
    }
    let dense = rows.len() == array.len()
        && rows
            .iter()
            .enumerate()
            .all(|(position, row)| *row == position as u64);
    if dense {
        return Ok(array.clone());
    }
    array.take(PrimitiveArray::from_iter(rows.iter().copied()).into_array())
}

fn visit_array(
    array: vortex::array::ArrayRef,
    session: &vortex::session::VortexSession,
    visitor: &vx_velox_visitor,
) -> VortexResult<()> {
    let length = array.len();
    CursorExport::try_new_canonical(array, session, None)?.visit(0, length, visitor)
}

/// Create one export cursor for several Velox output windows.
///
/// # Safety
///
/// The session and array pointers must identify live handles.
/// The memory callbacks must identify a complete, thread-safe callback table.
/// `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_export_cursor_new(
    session: *const vx_velox_session,
    array: *const vx_velox_array,
    memory_callbacks: *const vx_velox_arrow_memory_callbacks,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_export_cursor {
    try_or(error_out, ptr::null_mut(), || {
        let session = unsafe { vx_session_ref(session)? };
        let array = unsafe { vx_array_ref(array)? };
        let memory_callbacks = unsafe { parse_memory_callbacks(memory_callbacks)? };
        Ok(Box::into_raw(Box::new(vx_velox_export_cursor {
            export: CursorExport::try_new(array.clone(), session, Some(memory_callbacks))?,
        })))
    })
}

/// Free one export cursor.
///
/// # Safety
///
/// The pointer must be null or come from [`vx_velox_export_cursor_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_export_cursor_free(cursor: *mut vx_velox_export_cursor) {
    if !cursor.is_null() {
        // SAFETY: The pointer came from `Box::into_raw` and is freed once.
        drop(unsafe { Box::from_raw(cursor) });
    }
}

/// Visit one contiguous range from a retained export cursor.
///
/// # Safety
///
/// The cursor and visitor pointers must remain live until this call returns.
/// Concurrent calls are valid. The caller must not free the cursor before all calls return.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_export_cursor_visit(
    cursor: *const vx_velox_export_cursor,
    offset: usize,
    length: usize,
    visitor: *const vx_velox_visitor,
    error_out: *mut *mut vx_velox_error,
) -> i32 {
    try_or(error_out, 1, || {
        let cursor = unsafe {
            cursor
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox export cursor must not be null"))?
        };
        let visitor = unsafe {
            visitor
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox visitor must not be null"))?
        };
        validate_visitor(visitor)?;
        cursor.export.visit(offset, length, visitor)?;
        Ok(0)
    })
}

/// Visit one Vortex array through host semantic callbacks.
///
/// The request selects source positions once. Callback block positions are compact and follow the
/// request order.
///
/// # Safety
///
/// Every pointer must be null or valid for the documented access. The array and session handles
/// must remain live until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_visit(
    session: *const vx_velox_session,
    array: *const vx_velox_array,
    request: *const vx_velox_visit_request,
    visitor: *const vx_velox_visitor,
    error_out: *mut *mut vx_velox_error,
) -> i32 {
    try_or(error_out, 1, || {
        let session = unsafe { vx_session_ref(session)? };
        let array = unsafe { vx_array_ref(array)? };
        let request = unsafe {
            request
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox visit request must not be null"))?
        };
        if request.struct_size < size_of::<vx_velox_visit_request>() {
            vortex_bail!(
                "Vortex Velox visit request is too small: expected at least {}, got {}",
                size_of::<vx_velox_visit_request>(),
                request.struct_size
            );
        }
        let visitor = unsafe {
            visitor
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox visitor must not be null"))?
        };
        validate_visitor(visitor)?;
        visit_array(selected_array(array, request)?, session, visitor)?;
        Ok(0)
    })
}

#[cfg(test)]
mod tests;
