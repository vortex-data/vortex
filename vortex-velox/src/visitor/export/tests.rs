// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::align_of;
use std::ptr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::MapArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldNames;
use vortex::dtype::MapDType;
use vortex::dtype::Nullability;
use vortex::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_fastlanes::BitPackedData;

use super::*;
use crate::api::vx_velox_array_free;
use crate::ffi::vx_array_new_with;
use crate::ffi::vx_session_free;
use crate::ffi::vx_session_new_with;

#[derive(Default)]
struct TestMemory {
    retained_bytes: AtomicUsize,
}

unsafe extern "C" fn retain_test_memory(_context: *mut c_void) {}

unsafe extern "C" fn release_test_memory(_context: *mut c_void) {}

unsafe extern "C" fn reserve_test_memory(context: *mut c_void, bytes: usize) -> i32 {
    // SAFETY: The test context stays live through every callback.
    let memory = unsafe { &*context.cast::<TestMemory>() };
    memory.retained_bytes.fetch_add(bytes, Ordering::Relaxed);
    0
}

unsafe extern "C" fn free_test_memory(context: *mut c_void, bytes: usize) {
    // SAFETY: The test context stays live through every callback.
    let memory = unsafe { &*context.cast::<TestMemory>() };
    memory.retained_bytes.fetch_sub(bytes, Ordering::Relaxed);
}

fn test_memory_callbacks(memory: &mut TestMemory) -> vx_velox_arrow_memory_callbacks {
    vx_velox_arrow_memory_callbacks {
        struct_size: size_of::<vx_velox_arrow_memory_callbacks>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (memory as *mut TestMemory).cast(),
        retain_context: Some(retain_test_memory),
        release_context: Some(release_test_memory),
        report_allocation: Some(reserve_test_memory),
        report_free: Some(free_test_memory),
        last_error: None,
    }
}

#[rstest]
#[case(PType::U8, VX_VELOX_PRIMITIVE_U8)]
#[case(PType::U16, VX_VELOX_PRIMITIVE_U16)]
#[case(PType::U32, VX_VELOX_PRIMITIVE_U32)]
#[case(PType::U64, VX_VELOX_PRIMITIVE_U64)]
#[case(PType::I8, VX_VELOX_PRIMITIVE_I8)]
#[case(PType::I16, VX_VELOX_PRIMITIVE_I16)]
#[case(PType::I32, VX_VELOX_PRIMITIVE_I32)]
#[case(PType::I64, VX_VELOX_PRIMITIVE_I64)]
#[case(PType::F16, VX_VELOX_PRIMITIVE_F16)]
#[case(PType::F32, VX_VELOX_PRIMITIVE_F32)]
#[case(PType::F64, VX_VELOX_PRIMITIVE_F64)]
fn maps_primitive_types(#[case] input: PType, #[case] expected: vx_velox_primitive_type) {
    assert_eq!(primitive_type_id(input), expected);
}

#[test]
fn date_days_use_i32_storage_and_millisecond_dates_are_rejected() -> VortexResult<()> {
    let session = vortex::session::VortexSession::empty();
    let days = TemporalArray::new_date(
        PrimitiveArray::from_option_iter([Some(-1_i32), None, Some(19_000)]).into_array(),
        TimeUnit::Days,
    )
    .into_array();
    let CursorExport::Primitive(export) = CursorExport::try_new_canonical(days, &session, None)?
    else {
        vortex_bail!("date visitor did not produce primitive storage");
    };
    assert_eq!(export.primitive_type, VX_VELOX_PRIMITIVE_I32);
    let view = export.view(0, 3)?;
    // SAFETY: The export owns three readable i32 values.
    let values = unsafe { slice::from_raw_parts(view.values.cast::<i32>(), 3) };
    assert_eq!(values, [-1, 0, 19_000]);
    assert_eq!(view.validity_kind, VX_VELOX_VALIDITY_BITMAP);

    let milliseconds = TemporalArray::new_date(
        PrimitiveArray::from_iter([86_400_000_i64]).into_array(),
        TimeUnit::Milliseconds,
    )
    .into_array();
    let error = match CursorExport::try_new_canonical(milliseconds, &session, None) {
        Ok(_) => vortex_bail!("millisecond date visitor unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("Velox DATE uses days"));
    Ok(())
}

#[test]
fn decimals_normalize_to_velox_storage_widths() -> VortexResult<()> {
    let session = vortex::session::VortexSession::empty();
    let short = DecimalArray::new(
        buffer![1_i8, -2, 3],
        DecimalDType::new(18, 2),
        Validity::NonNullable,
    )
    .into_array();
    let short = PrimitiveExport::try_new_decimal(short, &session, None)?;
    assert_eq!(short.primitive_type, VX_VELOX_PRIMITIVE_I64);
    let short_view = short.view(0, 3)?;
    assert_eq!(short_view.decimal_precision, 18);
    assert_eq!(short_view.decimal_scale, 2);
    // SAFETY: The export owns three readable i64 values.
    let short_values = unsafe { slice::from_raw_parts(short_view.values.cast::<i64>(), 3) };
    assert_eq!(short_values, [1, -2, 3]);

    let nullable_short = DecimalArray::new(
        buffer![1_i128, i128::MAX],
        DecimalDType::new(18, 2),
        Validity::from_iter([true, false]),
    )
    .into_array();
    let nullable_short = PrimitiveExport::try_new_decimal(nullable_short, &session, None)?;
    let nullable_short_view = nullable_short.view(0, 2)?;
    // SAFETY: The export owns two readable i64 values.
    let nullable_short_values =
        unsafe { slice::from_raw_parts(nullable_short_view.values.cast::<i64>(), 2) };
    assert_eq!(nullable_short_values, [1, 0]);
    assert_eq!(nullable_short_view.validity_kind, VX_VELOX_VALIDITY_BITMAP);

    let long = DecimalArray::new(
        buffer![1_i64, -2, 3],
        DecimalDType::new(30, 4),
        Validity::NonNullable,
    )
    .into_array();
    let long = PrimitiveExport::try_new_decimal(long, &session, None)?;
    assert_eq!(long.primitive_type, VX_VELOX_PRIMITIVE_I128);
    let long_view = long.view(0, 3)?;
    assert_eq!(long_view.decimal_precision, 30);
    assert_eq!(long_view.decimal_scale, 4);
    // SAFETY: The export owns three readable i128 values.
    let long_values = unsafe { slice::from_raw_parts(long_view.values.cast::<i128>(), 3) };
    assert_eq!(long_values, [1, -2, 3]);

    let unsupported = DecimalArray::new(
        buffer![1_i8],
        DecimalDType::new(39, 0),
        Validity::NonNullable,
    )
    .into_array();
    let error = match PrimitiveExport::try_new_decimal(unsupported, &session, None) {
        Ok(_) => vortex_bail!("precision 39 decimal visitor unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("decimal precision 39"));
    Ok(())
}

#[test]
fn dictionary_export_preserves_code_width_and_nullable_children() -> VortexResult<()> {
    let session = vortex::session::VortexSession::empty();
    let code_cases: [(ArrayRef, vx_velox_primitive_type); 4] = [
        (buffer![0_u8, 1, 0].into_array(), VX_VELOX_PRIMITIVE_U8),
        (buffer![0_u16, 1, 0].into_array(), VX_VELOX_PRIMITIVE_U16),
        (buffer![0_u32, 1, 0].into_array(), VX_VELOX_PRIMITIVE_U32),
        (buffer![0_u64, 1, 0].into_array(), VX_VELOX_PRIMITIVE_U64),
    ];
    for (codes, expected_type) in code_cases {
        let dictionary = DictArray::try_new(codes, buffer![10_i64, 20].into_array())?;
        let CursorExport::Dictionary(export) =
            CursorExport::try_new(dictionary.into_array(), &session, None)?
        else {
            vortex_bail!("dictionary export lost its outer encoding");
        };
        assert_eq!(export.codes.primitive_type, expected_type);
        assert_eq!(export.values_length, 2);
        assert!(matches!(export.values.export, CursorExport::Primitive(_)));
    }

    let codes = PrimitiveArray::from_option_iter([Some(0_u8), None, Some(1)]).into_array();
    let values = PrimitiveArray::from_option_iter([Some(10_i64), None]).into_array();
    let dictionary = DictArray::try_new(codes, values)?;
    let CursorExport::Dictionary(export) =
        CursorExport::try_new(dictionary.into_array(), &session, None)?
    else {
        vortex_bail!("nullable dictionary export lost its outer encoding");
    };
    assert_eq!(export.codes.validity_kind, VX_VELOX_VALIDITY_BITMAP);
    let CursorExport::Primitive(values) = &export.values.export else {
        vortex_bail!("nullable dictionary values lost their primitive representation");
    };
    assert_eq!(values.validity_kind, VX_VELOX_VALIDITY_BITMAP);
    Ok(())
}

#[test]
fn constant_export_preserves_null_value() -> VortexResult<()> {
    let session = vortex::session::VortexSession::empty();
    let constant = ConstantArray::new(Scalar::null_native::<i64>(), 10).into_array();
    let CursorExport::Constant(export) = CursorExport::try_new(constant, &session, None)? else {
        vortex_bail!("constant export lost its outer encoding");
    };
    assert_eq!(export.length, 10);
    let CursorExport::Primitive(value) = &export.value.export else {
        vortex_bail!("null constant lost its primitive representation");
    };
    assert_eq!(value.length, 1);
    assert_eq!(value.validity_kind, VX_VELOX_VALIDITY_ALL_INVALID);
    Ok(())
}

#[test]
fn struct_export_preserves_children_and_nonzero_window() -> VortexResult<()> {
    #[derive(Default)]
    struct StructCapture {
        length: usize,
        offset: usize,
        fields: *const *const vx_velox_export_cursor,
        field_count: usize,
        validity: *const u8,
        validity_bit_offset: usize,
        owner: Option<vx_velox_buffer_owner>,
    }

    unsafe extern "C" fn capture_struct(
        context: *mut c_void,
        view: *const vx_velox_struct_view,
    ) -> i32 {
        if context.is_null() || view.is_null() {
            return 1;
        }
        // SAFETY: The test passes pointers to live capture and view objects.
        let (capture, view) = unsafe { (&mut *context.cast::<StructCapture>(), &*view) };
        let Some(retain) = view.buffers.retain else {
            return 2;
        };
        // SAFETY: The visitor owner is live for the callback.
        unsafe { retain(view.buffers.owner) };
        capture.length = view.length;
        capture.offset = view.offset;
        capture.fields = view.fields;
        capture.field_count = view.field_count;
        capture.validity = view.validity;
        capture.validity_bit_offset = view.validity_bit_offset;
        capture.owner = Some(view.buffers);
        0
    }

    let session = vortex::session::VortexSession::empty();
    let length: usize = 130;
    let dictionary = DictArray::try_new(
        PrimitiveArray::from_iter((0..length).map(|index| [0_u8, 1][index % 2])).into_array(),
        buffer![10_i64, 20].into_array(),
    )?
    .into_array();
    let constant = ConstantArray::new(Scalar::from(7_i64), length).into_array();
    let parent_validity = Validity::from_iter((0..length).map(|index| index % 9 != 0));
    let struct_array = StructArray::new(
        FieldNames::from(["dictionary", "constant"]),
        [dictionary, constant],
        length,
        parent_validity,
    )
    .into_array();
    let CursorExport::Struct(export) = CursorExport::try_new(struct_array, &session, None)? else {
        vortex_bail!("struct export lost its outer encoding");
    };
    assert!(matches!(
        export.fields[0].export,
        CursorExport::Dictionary(_)
    ));
    assert!(matches!(export.fields[1].export, CursorExport::Constant(_)));

    let mut capture = StructCapture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: None,
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: Some(capture_struct),
        visit_list: None,
        visit_map: None,
    };
    export.visit(65, 63, &visitor)?;
    assert_eq!(capture.length, 63);
    assert_eq!(capture.offset, 65);
    assert_eq!(capture.field_count, 2);
    assert_eq!(capture.validity_bit_offset, 1);
    // SAFETY: The export retains both field cursors until it is dropped below.
    assert_eq!(unsafe { *capture.fields }, &raw const export.fields[0]);
    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("struct callback returned no validity owner"))?;
    drop(export);
    // SAFETY: The callback retained the parent owner before the cursor was dropped.
    assert!(
        unsafe {
            *capture
                .validity
                .add(capture.validity_bit_offset / u8::BITS as usize)
        } != 0
    );
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("struct owner returned no release callback"))?;
    // SAFETY: This release matches the callback retain above.
    unsafe { release(owner.owner) };
    Ok(())
}

#[test]
fn list_export_preserves_elements_window_and_accounting() -> VortexResult<()> {
    #[derive(Default)]
    struct ListCapture {
        length: usize,
        offsets: *const i32,
        sizes: *const i32,
        elements_length: usize,
        validity: *const u8,
        validity_bit_offset: usize,
        owner: Option<vx_velox_buffer_owner>,
    }

    unsafe extern "C" fn capture_list(
        context: *mut c_void,
        view: *const vx_velox_list_view,
    ) -> i32 {
        if context.is_null() || view.is_null() {
            return 1;
        }
        // SAFETY: The test passes pointers to live capture and view objects.
        let (capture, view) = unsafe { (&mut *context.cast::<ListCapture>(), &*view) };
        let Some(retain) = view.buffers.retain else {
            return 2;
        };
        // SAFETY: The visitor owner is live for the callback.
        unsafe { retain(view.buffers.owner) };
        capture.length = view.length;
        capture.offsets = view.offsets;
        capture.sizes = view.sizes;
        capture.elements_length = view.elements_length;
        capture.validity = view.validity;
        capture.validity_bit_offset = view.validity_bit_offset;
        capture.owner = Some(view.buffers);
        0
    }

    let session = vortex::session::VortexSession::empty();
    let length = 130;
    let elements = DictArray::try_new(
        buffer![0_u8, 1, 0, 1, 0, 1].into_array(),
        PrimitiveArray::from_option_iter([Some(10_i64), None]).into_array(),
    )?
    .into_array();
    let offsets = PrimitiveArray::from_iter((0..length).map(|index| [0_u32, 2, 4][index % 3]));
    let sizes =
        PrimitiveArray::from_iter((0..length).map(|index| if index % 10 == 0 { 0 } else { 2 }));
    let validity = Validity::from_iter((0..length).map(|index| index % 9 != 0));
    let list = ListViewArray::new(elements, offsets.into_array(), sizes.into_array(), validity)
        .into_array();
    let mut memory = TestMemory::default();
    let CursorExport::List(export) =
        CursorExport::try_new(list, &session, Some(test_memory_callbacks(&mut memory)))?
    else {
        vortex_bail!("list export lost its outer encoding");
    };
    assert!(matches!(
        export.elements.export,
        CursorExport::Dictionary(_)
    ));
    let expected_parent_bytes =
        length * 2 * size_of::<i32>() + length.div_ceil(u64::BITS as usize) * size_of::<u64>();
    assert_eq!(export.owner.retained_bytes, expected_parent_bytes);

    let mut capture = ListCapture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: None,
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: Some(capture_list),
        visit_map: None,
    };
    export.visit(65, 63, &visitor)?;
    assert_eq!(capture.length, 63);
    assert_eq!(capture.elements_length, 6);
    assert_eq!(capture.validity_bit_offset, 1);
    // SAFETY: The retained owner keeps both metadata arrays live.
    assert_eq!(unsafe { *capture.offsets }, 4);
    // SAFETY: The retained owner keeps both metadata arrays live.
    assert_eq!(unsafe { *capture.sizes }, 2);
    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("list callback returned no owner"))?;
    drop(export);
    assert_eq!(
        memory.retained_bytes.load(Ordering::Relaxed),
        expected_parent_bytes
    );
    // SAFETY: The callback retained the owner before the export was dropped.
    assert_eq!(unsafe { *capture.offsets.add(1) }, 0);
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("list owner returned no release callback"))?;
    // SAFETY: This release matches the callback retain above.
    unsafe { release(owner.owner) };
    assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn map_export_preserves_children_window_and_accounting() -> VortexResult<()> {
    #[derive(Default)]
    struct MapCapture {
        length: usize,
        offsets: *const i32,
        sizes: *const i32,
        keys: *const vx_velox_export_cursor,
        values: *const vx_velox_export_cursor,
        entries_length: usize,
        keys_sorted: bool,
        validity_bit_offset: usize,
        owner: Option<vx_velox_buffer_owner>,
    }

    unsafe extern "C" fn capture_map(context: *mut c_void, view: *const vx_velox_map_view) -> i32 {
        if context.is_null() || view.is_null() {
            return 1;
        }
        // SAFETY: The test passes pointers to live capture and view objects.
        let (capture, view) = unsafe { (&mut *context.cast::<MapCapture>(), &*view) };
        let Some(retain) = view.buffers.retain else {
            return 2;
        };
        // SAFETY: The visitor owner is live for the callback.
        unsafe { retain(view.buffers.owner) };
        capture.length = view.length;
        capture.offsets = view.offsets;
        capture.sizes = view.sizes;
        capture.keys = view.keys;
        capture.values = view.values;
        capture.entries_length = view.entries_length;
        capture.keys_sorted = view.keys_sorted;
        capture.validity_bit_offset = view.validity_bit_offset;
        capture.owner = Some(view.buffers);
        0
    }

    let session = vortex::session::VortexSession::empty();
    let keys = DictArray::try_new(
        buffer![0_u8, 1, 0, 1, 0, 1].into_array(),
        buffer![10_i64, 20].into_array(),
    )?
    .into_array();
    let values = ConstantArray::new(Scalar::from(7_i64), 6).into_array();
    let entries = StructArray::new(
        FieldNames::from(["key", "value"]),
        [keys, values],
        6,
        Validity::NonNullable,
    )
    .into_array();
    let entry_lists = ListViewArray::new(
        entries,
        buffer![0_u32, 2, 4].into_array(),
        buffer![2_u32, 2, 2].into_array(),
        Validity::from_iter([true, false, true]),
    );
    let map_dtype = MapDType::try_new(
        DType::Primitive(PType::I64, Nullability::NonNullable),
        DType::Primitive(PType::I64, Nullability::NonNullable),
        true,
    )?;
    let map = MapArray::try_new(map_dtype, entry_lists)?.into_array();
    let mut memory = TestMemory::default();
    let CursorExport::Map(export) =
        CursorExport::try_new(map, &session, Some(test_memory_callbacks(&mut memory)))?
    else {
        vortex_bail!("map export lost its outer encoding");
    };
    assert!(matches!(export.keys.export, CursorExport::Dictionary(_)));
    assert!(matches!(export.values.export, CursorExport::Constant(_)));
    let expected_parent_bytes = 3 * 2 * size_of::<i32>() + size_of::<u64>();
    assert_eq!(export.owner.retained_bytes, expected_parent_bytes);

    let mut capture = MapCapture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: None,
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: Some(capture_map),
    };
    export.visit(1, 2, &visitor)?;
    assert_eq!(capture.length, 2);
    assert_eq!(capture.entries_length, 6);
    assert!(capture.keys_sorted);
    assert_eq!(capture.validity_bit_offset, 1);
    assert_eq!(capture.keys, &raw const *export.keys);
    assert_eq!(capture.values, &raw const *export.values);
    // SAFETY: The retained owner keeps both metadata arrays live.
    assert_eq!(unsafe { *capture.offsets }, 2);
    // SAFETY: The retained owner keeps both metadata arrays live.
    assert_eq!(unsafe { *capture.sizes }, 2);
    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("map callback returned no owner"))?;
    drop(export);
    assert_eq!(
        memory.retained_bytes.load(Ordering::Relaxed),
        expected_parent_bytes
    );
    // SAFETY: The callback retained the owner before the export was dropped.
    assert_eq!(unsafe { *capture.offsets.add(1) }, 4);
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("map owner returned no release callback"))?;
    // SAFETY: This release matches the callback retain above.
    unsafe { release(owner.owner) };
    assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
    Ok(())
}

#[derive(Default)]
struct Capture {
    primitive_type: Option<vx_velox_primitive_type>,
    length: usize,
    values: *const u8,
    values_length: usize,
    values_alignment: usize,
    validity: *const u8,
    validity_length: usize,
    validity_bit_offset: usize,
    validity_alignment: usize,
    retained_bytes: usize,
    validity_kind: Option<vx_velox_validity_kind>,
    owner: Option<vx_velox_buffer_owner>,
}

unsafe extern "C" fn capture_primitive(
    context: *mut c_void,
    view: *const vx_velox_primitive_view,
) -> i32 {
    if context.is_null() || view.is_null() {
        return 1;
    }
    // SAFETY: The test passes pointers to live `Capture` and view objects.
    let (capture, view) = unsafe { (&mut *context.cast::<Capture>(), &*view) };
    let Some(retain) = view.buffers.retain else {
        return 2;
    };
    // SAFETY: The visitor owner is live for the callback.
    unsafe { retain(view.buffers.owner) };
    capture.primitive_type = Some(view.primitive_type);
    capture.length = view.length;
    capture.values = view.values;
    capture.values_length = view.values_length;
    capture.values_alignment = view.values_alignment;
    capture.validity = view.validity;
    capture.validity_length = view.validity_length;
    capture.validity_bit_offset = view.validity_bit_offset;
    capture.validity_alignment = view.validity_alignment;
    capture.retained_bytes = view.buffers.retained_bytes;
    capture.validity_kind = Some(view.validity_kind);
    capture.owner = Some(view.buffers);
    0
}

fn release_capture(capture: &Capture) -> VortexResult<()> {
    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
    // SAFETY: This release matches the retain in `capture_primitive`.
    unsafe { release(owner.owner) };
    Ok(())
}

#[derive(Default)]
struct VarBinCapture {
    struct_size: usize,
    kind: Option<vx_velox_varbin_kind>,
    length: usize,
    views: *const vx_velox_binary_view,
    views_length: usize,
    views_alignment: usize,
    data_buffers: *const vx_velox_byte_buffer_view,
    data_buffer_count: usize,
    validity: *const u8,
    validity_length: usize,
    validity_bit_offset: usize,
    validity_alignment: usize,
    validity_kind: Option<vx_velox_validity_kind>,
    retained_bytes: usize,
    owner: Option<vx_velox_buffer_owner>,
}

unsafe extern "C" fn capture_varbin(
    context: *mut c_void,
    view: *const vx_velox_varbin_view,
) -> i32 {
    if context.is_null() || view.is_null() {
        return 1;
    }
    // SAFETY: The test passes pointers to live capture and view objects.
    let (capture, view) = unsafe { (&mut *context.cast::<VarBinCapture>(), &*view) };
    let Some(retain) = view.buffers.retain else {
        return 2;
    };
    // SAFETY: The visitor owner is live for the callback.
    unsafe { retain(view.buffers.owner) };
    capture.struct_size = view.struct_size;
    capture.kind = Some(view.kind);
    capture.length = view.length;
    capture.views = view.views;
    capture.views_length = view.views_length;
    capture.views_alignment = view.views_alignment;
    capture.data_buffers = view.data_buffers;
    capture.data_buffer_count = view.data_buffer_count;
    capture.validity = view.validity;
    capture.validity_length = view.validity_length;
    capture.validity_bit_offset = view.validity_bit_offset;
    capture.validity_alignment = view.validity_alignment;
    capture.validity_kind = Some(view.validity_kind);
    capture.retained_bytes = view.buffers.retained_bytes;
    capture.owner = Some(view.buffers);
    0
}

fn release_varbin_capture(capture: &VarBinCapture) -> VortexResult<()> {
    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("visitor did not return a retained string owner"))?;
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("string owner did not return a release callback"))?;
    // SAFETY: This release matches the retain in `capture_varbin`.
    unsafe { release(owner.owner) };
    Ok(())
}

#[derive(Default)]
struct BoolCapture {
    length: usize,
    values: *const u8,
    values_bit_offset: usize,
    validity: *const u8,
    validity_bit_offset: usize,
    validity_kind: Option<vx_velox_validity_kind>,
    retained_bytes: usize,
    owner: Option<vx_velox_buffer_owner>,
}

unsafe extern "C" fn capture_bool(context: *mut c_void, view: *const vx_velox_bool_view) -> i32 {
    if context.is_null() || view.is_null() {
        return 1;
    }
    // SAFETY: The test passes pointers to live capture and view objects.
    let (capture, view) = unsafe { (&mut *context.cast::<BoolCapture>(), &*view) };
    let Some(retain) = view.buffers.retain else {
        return 2;
    };
    // SAFETY: The visitor owner is live for the callback.
    unsafe { retain(view.buffers.owner) };
    capture.length = view.length;
    capture.values = view.values;
    capture.values_bit_offset = view.values_bit_offset;
    capture.validity = view.validity;
    capture.validity_bit_offset = view.validity_bit_offset;
    capture.validity_kind = Some(view.validity_kind);
    capture.retained_bytes = view.buffers.retained_bytes;
    capture.owner = Some(view.buffers);
    0
}

fn release_bool_capture(capture: &BoolCapture) -> VortexResult<()> {
    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("visitor did not return a retained Boolean owner"))?;
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("Boolean owner did not return a release callback"))?;
    // SAFETY: This release matches the retain in `capture_bool`.
    unsafe { release(owner.owner) };
    Ok(())
}

#[expect(
    clippy::host_endian_bytes,
    reason = "The Vortex binary-view fields use the host C ABI layout"
)]
unsafe fn captured_varbin_value(capture: &VarBinCapture, index: usize) -> Option<&[u8]> {
    if capture.validity_kind == Some(VX_VELOX_VALIDITY_BITMAP) {
        let bit_index = capture.validity_bit_offset + index;
        // SAFETY: The callback contract retains the bitmap for every captured row.
        let byte = unsafe { *capture.validity.add(bit_index / 8) };
        if byte & (1 << (bit_index % 8)) == 0 {
            return None;
        }
    }
    // SAFETY: The callback contract retains `length` readable views.
    let view = unsafe { &*capture.views.add(index) };
    let length = view.length as usize;
    const INLINE_LENGTH: usize = size_of::<vx_velox_binary_view>() - size_of::<u32>();
    if length <= INLINE_LENGTH {
        return Some(&view.data[..length]);
    }
    let buffer_index =
        u32::from_ne_bytes([view.data[4], view.data[5], view.data[6], view.data[7]]) as usize;
    let offset =
        u32::from_ne_bytes([view.data[8], view.data[9], view.data[10], view.data[11]]) as usize;
    // SAFETY: The callback contract retains all payload descriptors.
    let buffer = unsafe { &*capture.data_buffers.add(buffer_index) };
    // SAFETY: Canonical Vortex views contain validated payload ranges.
    Some(unsafe { slice::from_raw_parts(buffer.data.add(offset), length) })
}

#[rstest]
#[case(DType::Utf8(Nullability::Nullable), VX_VELOX_VARBIN_UTF8)]
#[case(DType::Binary(Nullability::Nullable), VX_VELOX_VARBIN_BINARY)]
fn varbin_cursor_retains_mixed_views_across_nonzero_window(
    #[case] dtype: DType,
    #[case] expected_kind: vx_velox_varbin_kind,
) -> VortexResult<()> {
    let utf8_expected: [Option<&[u8]>; 7] = [
        Some(b""),
        Some(b"a"),
        None,
        Some(b"abcdefghijkl"),
        Some(b"abcdefghijklm"),
        Some("vortex 🌀 outlined".as_bytes()),
        Some(b"tail"),
    ];
    let binary_expected: [Option<&[u8]>; 7] = [
        Some(b""),
        Some(b"\xff"),
        None,
        Some(b"abcdefghijkl"),
        Some(b"\x00abcdefghijklm"),
        Some(b"\xff\x00 binary outlined value"),
        Some(b"tail"),
    ];
    let expected = if matches!(dtype, DType::Utf8(_)) {
        utf8_expected
    } else {
        binary_expected
    };
    let session = vx_session_new_with(|session| session);
    let varbin = VarBinViewArray::from_iter(expected, dtype);
    let array = vx_array_new_with(varbin.into_array());
    let mut error = ptr::null_mut();
    let mut memory = TestMemory::default();
    let memory_callbacks = test_memory_callbacks(&mut memory);
    // SAFETY: The session and array handles remain live until cursor creation finishes.
    let cursor = unsafe {
        vx_velox_export_cursor_new(session, array, &raw const memory_callbacks, &raw mut error)
    };
    vortex_ensure!(!cursor.is_null(), "string cursor creation failed");
    vortex_ensure!(error.is_null(), "string cursor returned an error");

    let mut capture = VarBinCapture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: None,
        last_error: None,
        visit_varbin: Some(capture_varbin),
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    // SAFETY: The cursor and callback state remain live through the call.
    let status =
        unsafe { vx_velox_export_cursor_visit(cursor, 1, 5, &raw const visitor, &raw mut error) };
    assert_eq!(status, 0);
    vortex_ensure!(error.is_null(), "string export window returned an error");
    assert_eq!(capture.struct_size, size_of::<vx_velox_varbin_view>());
    assert_eq!(capture.kind, Some(expected_kind));
    assert_eq!(capture.length, 5);
    assert_eq!(capture.views_length, 5 * size_of::<vx_velox_binary_view>());
    assert!(capture.views_alignment >= align_of::<vx_velox_binary_view>());
    assert_eq!(capture.views.addr() % align_of::<vx_velox_binary_view>(), 0);
    assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
    assert_eq!(capture.validity_bit_offset, 1);
    assert!(capture.validity_length >= 1);
    assert!(capture.validity_alignment >= align_of::<u64>());
    assert_eq!(capture.validity.addr() % align_of::<u64>(), 0);
    assert!(capture.data_buffer_count >= 1);
    assert!(!capture.data_buffers.is_null());
    assert_eq!(
        capture.retained_bytes,
        memory.retained_bytes.load(Ordering::Relaxed)
    );

    // SAFETY: Each owned handle is freed once. The callback retained the string owner.
    unsafe {
        vx_velox_export_cursor_free(cursor);
        vx_velox_array_free(array);
        vx_session_free(session);
    }
    assert_eq!(
        memory.retained_bytes.load(Ordering::Relaxed),
        capture.retained_bytes
    );
    for (index, expected) in expected[1..6].iter().enumerate() {
        // SAFETY: The retained owner keeps every captured pointer live.
        let actual = unsafe { captured_varbin_value(&capture, index) };
        assert_eq!(actual, *expected);
    }
    release_varbin_capture(&capture)?;
    assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn varbin_shared_buffers_compact_into_exact_owned_storage() -> VortexResult<()> {
    let length = 130_usize;
    let strings = VarBinViewArray::from_iter(
        (0..length)
            .map(|index| (index % 11 != 0).then(|| format!("outlined string value {index:03}"))),
        DType::Utf8(Nullability::Nullable),
    );
    let parts = strings.into_data_parts();
    let views_length = parts.views.try_to_host_sync()?.len();
    let data_length = parts
        .buffers
        .iter()
        .map(|buffer| Ok(buffer.try_to_host_sync()?.len()))
        .sum::<VortexResult<usize>>()?;
    let descriptor_length = parts.buffers.len() * size_of::<vx_velox_byte_buffer_view>();
    let validity_length = length.div_ceil(u64::BITS as usize) * size_of::<u64>();
    let expected_retained = views_length + data_length + descriptor_length + validity_length;

    let retained_views = parts.views.clone();
    let retained_buffers = Arc::<[BufferHandle]>::clone(&parts.buffers);
    let mut execution = vortex::session::VortexSession::empty().create_execution_ctx();
    let mask = parts.validity.execute_mask(length, &mut execution)?;
    let (_, validity) = exported_validity(true, mask);
    let owner = VarBinOwner::try_new(parts.views, parts.buffers, validity, length)?;

    assert!(matches!(owner.views, RetainedViews::Compact(_)));
    assert!(
        owner
            ._data
            .iter()
            .all(|buffer| matches!(buffer, RetainedBytes::Compact(_)))
    );
    assert_eq!(owner.retained_bytes, expected_retained);
    drop(retained_views);
    drop(retained_buffers);
    Ok(())
}

#[test]
fn retained_varbin_buffers_report_complete_unique_allocations() -> VortexResult<()> {
    let alignment = vortex::buffer::Alignment::new(256);
    let mut payload = BufferMut::<u8>::with_capacity_aligned(17, alignment);
    payload.extend(0..17);
    let expected_payload_allocation = payload.allocation_size();
    let (retained_payload, payload_allocation) =
        RetainedBytes::try_new(BufferHandle::new_host(payload.freeze()))?;
    assert!(matches!(retained_payload, RetainedBytes::Retained(_)));
    assert_eq!(payload_allocation, expected_payload_allocation);
    assert!(payload_allocation > 17);

    let mut views =
        BufferMut::<u8>::with_capacity_aligned(2 * size_of::<vx_velox_binary_view>(), alignment);
    views.extend(std::iter::repeat_n(
        0,
        2 * size_of::<vx_velox_binary_view>(),
    ));
    let expected_views_allocation = views.allocation_size();
    let (retained_views, views_allocation) =
        RetainedViews::try_new(BufferHandle::new_host(views.freeze()))?;
    assert!(matches!(retained_views, RetainedViews::Retained(_)));
    assert_eq!(views_allocation, expected_views_allocation);
    assert!(views_allocation > 2 * size_of::<vx_velox_binary_view>());
    Ok(())
}

#[test]
fn word_aligned_windows_rebase_validity_buffers() -> VortexResult<()> {
    let session = vortex::session::VortexSession::empty();
    let primitive = PrimitiveArray::from_option_iter(
        (0..130).map(|index| (index % 7 != 0).then_some(index as i64)),
    )
    .into_array();
    let primitive = PrimitiveExport::try_new(primitive, &session, None)?;
    let primitive_first = primitive.view(0, 64)?;
    let primitive_second = primitive.view(64, 64)?;
    assert_eq!(primitive_first.validity_bit_offset, 0);
    assert_eq!(primitive_second.validity_bit_offset, 0);
    // SAFETY: Both pointers lie in the retained validity allocation.
    assert_eq!(primitive_second.validity, unsafe {
        primitive_first.validity.add(size_of::<u64>())
    });

    let strings = VarBinViewArray::from_iter(
        (0..130).map(|index| (index % 11 != 0).then(|| format!("value-{index}"))),
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let strings = VarBinExport::try_new(strings, &session, None)?;
    let mut first = VarBinCapture::default();
    let first_visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut first).cast(),
        visit_primitive: None,
        last_error: None,
        visit_varbin: Some(capture_varbin),
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    strings.visit(0, 64, &first_visitor)?;

    let mut second = VarBinCapture::default();
    let second_visitor = vx_velox_visitor {
        context: (&raw mut second).cast(),
        ..first_visitor
    };
    strings.visit(64, 64, &second_visitor)?;
    assert_eq!(first.validity_bit_offset, 0);
    assert_eq!(second.validity_bit_offset, 0);
    // SAFETY: Both pointers lie in the retained validity allocation.
    assert_eq!(second.validity, unsafe {
        first.validity.add(size_of::<u64>())
    });
    release_varbin_capture(&first)?;
    release_varbin_capture(&second)?;
    Ok(())
}

#[test]
fn bool_cursor_retains_nonzero_window_and_exact_accounting() -> VortexResult<()> {
    let expected = (0..130)
        .map(|index| (index % 11 != 0).then_some(index % 3 == 0))
        .collect::<Vec<_>>();
    let session = vx_session_new_with(|session| session);
    let boolean = BoolArray::from_iter(expected.iter().copied());
    let array = vx_array_new_with(boolean.into_array());
    let mut error = ptr::null_mut();
    let mut memory = TestMemory::default();
    let memory_callbacks = test_memory_callbacks(&mut memory);
    // SAFETY: The session and array handles remain live until cursor creation finishes.
    let cursor = unsafe {
        vx_velox_export_cursor_new(session, array, &raw const memory_callbacks, &raw mut error)
    };
    vortex_ensure!(!cursor.is_null(), "Boolean cursor creation failed");
    vortex_ensure!(error.is_null(), "Boolean cursor returned an error");

    let mut capture = BoolCapture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: None,
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: Some(capture_bool),
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    // SAFETY: The cursor and callback state remain live through the call.
    let status =
        unsafe { vx_velox_export_cursor_visit(cursor, 65, 63, &raw const visitor, &raw mut error) };
    assert_eq!(status, 0);
    vortex_ensure!(error.is_null(), "Boolean export window returned an error");
    assert_eq!(capture.length, 63);
    assert_eq!(capture.values_bit_offset, 1);
    assert_eq!(capture.validity_bit_offset, 1);
    assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
    assert_eq!(capture.retained_bytes, 6 * size_of::<u64>());
    assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 48);

    // SAFETY: Each owned handle is freed once. The callback retained the Boolean owner.
    unsafe {
        vx_velox_export_cursor_free(cursor);
        vx_velox_array_free(array);
        vx_session_free(session);
    }
    assert_eq!(
        memory.retained_bytes.load(Ordering::Relaxed),
        capture.retained_bytes
    );
    for (relative_index, expected) in expected[65..128].iter().enumerate() {
        let value_bit = capture.values_bit_offset + relative_index;
        let validity_bit = capture.validity_bit_offset + relative_index;
        // SAFETY: The retained buffers cover every captured value and validity bit.
        let (actual, is_valid) = unsafe {
            (
                *capture.values.add(value_bit / 8) & (1 << (value_bit % 8)) != 0,
                *capture.validity.add(validity_bit / 8) & (1 << (validity_bit % 8)) != 0,
            )
        };
        assert_eq!(is_valid, expected.is_some());
        if let Some(expected) = expected {
            assert_eq!(actual, *expected);
        }
    }
    release_bool_capture(&capture)?;
    assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn export_cursor_reuses_one_prepared_array_across_windows() -> VortexResult<()> {
    let session = vx_session_new_with(|session| session);
    let array = vx_array_new_with(
        PrimitiveArray::from_option_iter([Some(10_i64), None, Some(30), Some(40), Some(50)])
            .into_array(),
    );
    let mut error = ptr::null_mut();
    let mut memory = TestMemory::default();
    let memory_callbacks = test_memory_callbacks(&mut memory);
    // SAFETY: The session and array handles remain live until cursor creation finishes.
    let cursor = unsafe {
        vx_velox_export_cursor_new(session, array, &raw const memory_callbacks, &raw mut error)
    };
    vortex_ensure!(!cursor.is_null(), "export cursor creation failed");
    vortex_ensure!(error.is_null(), "export cursor returned an error");
    assert!(memory.retained_bytes.load(Ordering::Relaxed) >= 48);

    let mut first = Capture::default();
    let first_visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut first).cast(),
        visit_primitive: Some(capture_primitive),
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    // SAFETY: The cursor and callback state remain live through the call.
    let status = unsafe {
        vx_velox_export_cursor_visit(cursor, 1, 2, &raw const first_visitor, &raw mut error)
    };
    assert_eq!(status, 0);
    vortex_ensure!(error.is_null(), "first export window returned an error");
    assert_eq!(first.length, 2);
    assert_eq!(first.validity_bit_offset, 1);
    // SAFETY: The callback retained two readable i64 values.
    let first_values = unsafe { slice::from_raw_parts(first.values.cast::<i64>(), 2) };
    assert_eq!(first_values, [0, 30]);
    assert_eq!(
        first.retained_bytes,
        memory.retained_bytes.load(Ordering::Relaxed)
    );
    let owner = first
        .owner
        .ok_or_else(|| vortex_err!("first export window returned no owner"))?
        .owner;
    release_capture(&first)?;

    let mut second = Capture::default();
    let second_visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut second).cast(),
        visit_primitive: Some(capture_primitive),
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    // SAFETY: The cursor and callback state remain live through the call.
    let status = unsafe {
        vx_velox_export_cursor_visit(cursor, 3, 2, &raw const second_visitor, &raw mut error)
    };
    assert_eq!(status, 0);
    vortex_ensure!(error.is_null(), "second export window returned an error");
    assert_eq!(second.length, 2);
    assert_eq!(second.validity_bit_offset, 3);
    assert_eq!(
        second
            .owner
            .ok_or_else(|| vortex_err!("second export window returned no owner"))?
            .owner,
        owner
    );

    // SAFETY: Each owned handle is freed exactly once. The second callback retained the owner.
    unsafe {
        vx_velox_export_cursor_free(cursor);
        vx_velox_array_free(array);
        vx_session_free(session);
    }
    // SAFETY: The retained cursor owner keeps these two i64 values live.
    let second_values = unsafe { slice::from_raw_parts(second.values.cast::<i64>(), 2) };
    assert_eq!(second_values, [40, 50]);
    release_capture(&second)?;
    assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn export_cursor_decodes_sliced_bitpacked_into_exact_owner() -> VortexResult<()> {
    let session = vx_session_new_with(|session| {
        vortex_fastlanes::initialize(&session);
        session
    });
    let session_ref = unsafe { vx_session_ref(session)? };
    let values = (0..2_050).map(|index| (index % 7 != 0).then_some(i64::from(index % 100)));
    let primitive = PrimitiveArray::from_option_iter(values).into_array();
    let mut execution = session_ref.create_execution_ctx();
    let bitpacked = BitPackedData::encode(&primitive, 7, &mut execution)?;
    vortex_ensure!(
        bitpacked.patches().is_none(),
        "test bit-packed array unexpectedly contains patches"
    );
    let slice_begin = 113;
    let slice_end = 1_941;
    let sliced = bitpacked.into_array().slice(slice_begin..slice_end)?;
    let array = vx_array_new_with(sliced);
    let mut error = ptr::null_mut();
    let mut memory = TestMemory::default();
    let memory_callbacks = test_memory_callbacks(&mut memory);
    // SAFETY: The session and array handles remain live until cursor creation finishes.
    let cursor = unsafe {
        vx_velox_export_cursor_new(session, array, &raw const memory_callbacks, &raw mut error)
    };
    vortex_ensure!(!cursor.is_null(), "export cursor creation failed");
    vortex_ensure!(error.is_null(), "export cursor returned an error");
    let sliced_length = slice_end - slice_begin;
    let expected_retained = sliced_length * size_of::<i64>()
        + sliced_length.div_ceil(u64::BITS as usize) * size_of::<u64>();
    assert_eq!(
        memory.retained_bytes.load(Ordering::Relaxed),
        expected_retained
    );

    let window_offset = 997;
    let window_length = 6;
    let mut capture = Capture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: Some(capture_primitive),
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    // SAFETY: The cursor and callback state remain live through the call.
    let status = unsafe {
        vx_velox_export_cursor_visit(
            cursor,
            window_offset,
            window_length,
            &raw const visitor,
            &raw mut error,
        )
    };
    assert_eq!(status, 0);
    vortex_ensure!(error.is_null(), "export window returned an error");
    assert_eq!(capture.primitive_type, Some(VX_VELOX_PRIMITIVE_I64));
    assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
    assert_eq!(
        capture.validity_bit_offset,
        window_offset % u64::BITS as usize
    );
    assert_eq!(capture.retained_bytes, expected_retained);
    // SAFETY: The callback retained `window_length` readable i64 values.
    let actual = unsafe { slice::from_raw_parts(capture.values.cast::<i64>(), window_length) };
    for (relative_index, value) in actual.iter().enumerate() {
        let sliced_index = window_offset + relative_index;
        let source_index = slice_begin + sliced_index;
        // SAFETY: The retained bitmap covers every row in the sliced array.
        let validity_index = capture.validity_bit_offset + relative_index;
        let validity_byte = unsafe { *capture.validity.add(validity_index / 8) };
        let is_valid = validity_byte & (1 << (validity_index % 8)) != 0;
        assert_eq!(is_valid, source_index % 7 != 0);
        if is_valid {
            assert_eq!(*value, i64::try_from(source_index % 100)?);
        }
    }

    // SAFETY: Each owned handle is freed exactly once. The callback retained the owner.
    unsafe {
        vx_velox_export_cursor_free(cursor);
        vx_velox_array_free(array);
        vx_session_free(session);
    }
    assert_eq!(
        memory.retained_bytes.load(Ordering::Relaxed),
        expected_retained
    );
    release_capture(&capture)?;
    assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn patched_bitpacked_uses_retained_canonical_fallback() -> VortexResult<()> {
    let session = vx_session_new_with(|session| {
        vortex_fastlanes::initialize(&session);
        session
    });
    let session_ref = unsafe { vx_session_ref(session)? };
    let expected = [1_u64, 2, 3, u64::MAX];
    let primitive = PrimitiveArray::from_iter(expected).into_array();
    let mut execution = session_ref.create_execution_ctx();
    let bitpacked = BitPackedData::encode(&primitive, 2, &mut execution)?;
    vortex_ensure!(
        bitpacked.patches().is_some(),
        "test bit-packed array unexpectedly omitted patches"
    );
    let mut memory = TestMemory::default();
    let export = PrimitiveExport::try_new(
        bitpacked.into_array(),
        session_ref,
        Some(test_memory_callbacks(&mut memory)),
    )?;
    assert!(matches!(export.owner.values, PrimitiveValues::Retained(_)));
    assert_eq!(
        memory.retained_bytes.load(Ordering::Relaxed),
        export.owner.retained_bytes()
    );
    // SAFETY: The export owner contains `expected.len()` initialized u64 values.
    let actual =
        unsafe { slice::from_raw_parts(export.owner.values().cast::<u64>(), expected.len()) };
    assert_eq!(actual, expected);
    drop(export);
    assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
    unsafe { vx_session_free(session) };
    Ok(())
}

#[test]
fn visits_sparse_nullable_values_with_retained_buffers() -> VortexResult<()> {
    let session = vx_session_new_with(|session| session);
    let array = vx_array_new_with(
        PrimitiveArray::from_option_iter([Some(10_i64), None, Some(30), Some(40)]).into_array(),
    );
    let rows = [1_u64, 3];
    let request = vx_velox_visit_request {
        struct_size: size_of::<vx_velox_visit_request>(),
        rows: rows.as_ptr(),
        row_count: rows.len(),
    };
    let mut capture = Capture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: Some(capture_primitive),
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    let mut error = ptr::null_mut();
    // SAFETY: Every handle and callback object stays live for this call.
    let status = unsafe {
        vx_velox_array_visit(
            session,
            array,
            &raw const request,
            &raw const visitor,
            &raw mut error,
        )
    };
    assert_eq!(status, 0);
    vortex_ensure!(error.is_null(), "visitor returned an error");
    assert_eq!(capture.primitive_type, Some(VX_VELOX_PRIMITIVE_I64));
    assert_eq!(capture.length, 2);
    assert_eq!(capture.values_length, 2 * size_of::<i64>());
    assert!(capture.values_alignment.is_power_of_two());
    assert_eq!(capture.values.addr() % capture.values_alignment, 0);
    assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
    assert_eq!(capture.validity_length, size_of::<u64>());
    assert_eq!(capture.validity_bit_offset, 0);
    assert!(capture.validity_alignment.is_power_of_two());
    assert_eq!(capture.validity.addr() % capture.validity_alignment, 0);
    assert_eq!(
        capture.retained_bytes,
        capture.values_length + size_of::<u64>()
    );
    // SAFETY: The callback retained the owner before storing these pointers.
    let values = unsafe { slice::from_raw_parts(capture.values.cast::<i64>(), 2) };
    assert_eq!(values, [0, 40]);
    // SAFETY: The retained validity pointer has one readable word.
    let validity = unsafe { *capture.validity };
    assert_eq!(validity & 0b11, 0b10);

    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
    // SAFETY: This release matches the retain in `capture_primitive`.
    unsafe { release(owner.owner) };
    // SAFETY: Each owned handle is freed exactly once.
    unsafe {
        vx_velox_array_free(array);
        vx_session_free(session);
    }
    Ok(())
}

#[test]
fn copies_sliced_values_into_exact_owned_storage() -> VortexResult<()> {
    let session = vx_session_new_with(|session| session);
    let source = PrimitiveArray::from_iter(0_i32..16);
    let source_values = source.buffer_handle().try_to_host_sync()?;
    // SAFETY: The source contains sixteen i32 values. The fifth value is in bounds.
    let source_slice = unsafe { source_values.as_ptr().add(5 * size_of::<i32>()) };
    drop(source_values);
    let array = vx_array_new_with(source.into_array().slice(5..8)?);
    let request = vx_velox_visit_request {
        struct_size: size_of::<vx_velox_visit_request>(),
        rows: ptr::null(),
        row_count: 0,
    };
    let mut capture = Capture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: Some(capture_primitive),
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    let mut error = ptr::null_mut();
    let status = unsafe {
        vx_velox_array_visit(
            session,
            array,
            &raw const request,
            &raw const visitor,
            &raw mut error,
        )
    };
    assert_eq!(status, 0);
    vortex_ensure!(error.is_null(), "visitor returned an error");
    assert_eq!(capture.values_length, 3 * size_of::<i32>());
    assert_eq!(capture.retained_bytes, 2 * size_of::<u64>());
    assert_ne!(capture.values, source_slice);
    // SAFETY: Each owned handle is freed exactly once. The callback retained the value owner.
    unsafe {
        vx_velox_array_free(array);
        vx_session_free(session);
    }
    // SAFETY: The retained compact buffer contains three i32 values.
    let values = unsafe { slice::from_raw_parts(capture.values.cast::<i32>(), 3) };
    assert_eq!(values, [5, 6, 7]);
    assert!(capture.values_alignment.is_power_of_two());
    assert_eq!(capture.values.addr() % capture.values_alignment, 0);
    assert_eq!(capture.validity_alignment, 0);

    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
    unsafe { release(owner.owner) };
    Ok(())
}

#[test]
fn copies_validity_into_word_padded_storage() -> VortexResult<()> {
    let session = vx_session_new_with(|session| session);
    let session_ref = unsafe { vx_session_ref(session)? };
    let primitive = PrimitiveArray::from_option_iter([Some(1_i32), None, Some(3)]);
    let mut execution = session_ref.create_execution_ctx();
    let Mask::Values(mask) = primitive
        .validity()?
        .execute_mask(primitive.len(), &mut execution)?
    else {
        vortex_bail!("Expected bitmap validity");
    };
    let expected_validity = mask.bit_buffer().inner().as_ptr();
    let array = vx_array_new_with(primitive.into_array());
    let request = vx_velox_visit_request {
        struct_size: size_of::<vx_velox_visit_request>(),
        rows: ptr::null(),
        row_count: 0,
    };
    let mut capture = Capture::default();
    let visitor = vx_velox_visitor {
        struct_size: size_of::<vx_velox_visitor>(),
        abi_version: crate::VX_VELOX_ABI_VERSION,
        context: (&raw mut capture).cast(),
        visit_primitive: Some(capture_primitive),
        last_error: None,
        visit_varbin: None,
        visit_dictionary: None,
        visit_constant: None,
        visit_bool: None,
        visit_struct: None,
        visit_list: None,
        visit_map: None,
    };
    let mut error = ptr::null_mut();
    let status = unsafe {
        vx_velox_array_visit(
            session,
            array,
            &raw const request,
            &raw const visitor,
            &raw mut error,
        )
    };
    assert_eq!(status, 0);
    vortex_ensure!(error.is_null(), "visitor returned an error");
    assert_ne!(capture.validity, expected_validity);
    assert_eq!(capture.validity_bit_offset, 0);
    assert_eq!(capture.validity_length, size_of::<u64>());
    assert!(capture.validity_alignment >= align_of::<u64>());
    assert_eq!(
        capture.retained_bytes,
        capture.values_length.div_ceil(size_of::<u64>()) * size_of::<u64>() + size_of::<u64>()
    );

    let owner = capture
        .owner
        .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
    let release = owner
        .release
        .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
    unsafe { release(owner.owner) };
    unsafe {
        vx_velox_array_free(array);
        vx_session_free(session);
    }
    Ok(())
}

#[test]
fn rejects_unsorted_rows() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter([1_i64, 2, 3]).into_array();
    let rows = [2_u64, 1];
    let request = vx_velox_visit_request {
        struct_size: size_of::<vx_velox_visit_request>(),
        rows: rows.as_ptr(),
        row_count: rows.len(),
    };
    match selected_array(&array, &request) {
        Ok(_) => vortex_bail!("unsorted rows unexpectedly succeeded"),
        Err(error) => assert!(error.to_string().contains("unique and increasing")),
    }
    Ok(())
}
