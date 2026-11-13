// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use futures::stream;
use tempfile::NamedTempFile;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::assert_arrays_eq;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::validity::Validity;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::DecimalDType;
use vortex::dtype::NativePType;
use vortex::error::VortexError;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::session::SessionExt;
use vortex_cuda::VarBinExportLayout;
use vortex_cuda::arrow::ARROW_DEVICE_CUDA;
use vortex_cuda::arrow::ArrowArray;
use vortex_cuda::arrow::ArrowDeviceArrayStream;
use vortex_cuda::arrow::release_device_array;
use vortex_cuda_driver::CudaContext;
use vortex_cuda_driver::event;
use vortex_cuda_driver::memory;
use vortex_cuda_driver::sys::CUevent;
use vortex_cuda_macros::test as cuda_test;
use vortex_ffi::vx_error_free;
use vortex_ffi::vx_error_message;
use vortex_ffi::vx_session_free;

use super::*;
use crate::scan_export_ctx;

struct Output {
    schema: FFI_ArrowSchema,
    array: ArrowDeviceArray,
}

impl Output {
    fn empty() -> Self {
        Self {
            schema: FFI_ArrowSchema::empty(),
            array: ArrowDeviceArray::empty(),
        }
    }

    fn is_eos(&self) -> bool {
        assert_eq!(
            self.schema.release().is_none(),
            self.array.array.release.is_none()
        );
        self.schema.release().is_none()
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        release_device_array(&mut self.array);
    }
}

fn take_error(error: *mut vx_error) -> Option<String> {
    if error.is_null() {
        return None;
    }
    // SAFETY: All callers transfer an owned error from these FFI calls. Copy before freeing it.
    let message = unsafe { vx_error_message(error).as_str() }
        .unwrap()
        .to_owned();
    unsafe { vx_error_free(error) };
    Some(message)
}

fn next(handle: &mut vx_cuda_device_batches) -> VortexResult<Output> {
    let mut output = Output::empty();
    next_into(handle, &mut output)?;
    Ok(output)
}

/// Advance into existing output storage through the public ABI.
///
/// Shared by success and error tests so both check status codes and error ownership.
fn next_into(handle: &mut vx_cuda_device_batches, output: &mut Output) -> VortexResult<()> {
    let mut error = ptr::null_mut();
    // SAFETY: The handle is exclusively borrowed and outputs are live, disjoint, and writable.
    let status = unsafe {
        vx_cuda_device_batches_next(
            handle,
            &raw mut output.schema,
            &raw mut output.array,
            &raw mut error,
        )
    };
    if let Some(error) = take_error(error) {
        assert_eq!(status, VX_CUDA_ERR);
        return Err(vortex_err!("{error}"));
    }
    assert_eq!(status, VX_CUDA_OK);
    Ok(())
}

fn session() -> VortexResult<VortexSession> {
    // Override Decode per handle, rather than relying on the session default being Preserve.
    Ok(VortexSession::default().with_some(
        CudaSession::try_default()?
            .with_dictionary_export(DictionaryExport::Decode)
            .with_varbin_export_layout(VarBinExportLayout::VarBin),
    ))
}

fn batches(session: &VortexSession, arrays: Vec<ArrayRef>) -> VortexResult<vx_cuda_device_batches> {
    let dtype = arrays[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, stream::iter(arrays.into_iter().map(Ok))).boxed();
    vx_cuda_device_batches::new(stream, session, ffi_runtime())
}

/// Build a primitive fixture for a mixed-encoding batch stream.
///
/// Keeps the element type explicit at call sites while returning an erased array.
fn primitive<T: NativePType>(values: impl IntoIterator<Item = T>) -> ArrayRef {
    PrimitiveArray::from_iter(values).into_array()
}

fn strings(values: &[Option<&str>]) -> ArrayRef {
    VarBinViewArray::from_iter_nullable_str(values.iter().copied()).into_array()
}

fn dictionary(codes: ArrayRef, values: &[Option<&str>]) -> VortexResult<ArrayRef> {
    Ok(DictArray::try_new(codes, strings(values))?.into_array())
}

fn table(column: ArrayRef) -> ArrayRef {
    let len = column.len();
    StructArray::new(["text"].into(), vec![column], len, Validity::NonNullable).into_array()
}

fn synchronize(array: &ArrowDeviceArray) -> VortexResult<Arc<CudaContext>> {
    assert_eq!(array.device_type, ARROW_DEVICE_CUDA);
    let context = CudaContext::new(usize::try_from(array.device_id)?)
        .map_err(|error| vortex_err!("opening device context: {error}"))?;
    context
        .bind_to_thread()
        .map_err(|error| vortex_err!("binding context: {error}"))?;
    if !array.sync_event.is_null() {
        // SAFETY: The live exported array owns the CUDA event handle.
        unsafe { event::synchronize(*array.sync_event.cast::<CUevent>()) }
            .map_err(|error| vortex_err!("waiting for device batch: {error}"))?;
    }
    Ok(context)
}

/// Read one buffer from a live, synchronized Arrow array with the specified element type/length.
unsafe fn buffer<T: NativePType>(
    array: &ArrowArray,
    slot: usize,
    len: usize,
) -> VortexResult<Vec<T>> {
    assert!(slot < usize::try_from(array.n_buffers)?);
    let mut values = vec![T::default(); len];
    if len != 0 {
        // SAFETY: The caller guarantees the buffer layout, lifetime, and synchronization.
        let source = unsafe { *array.buffers.add(slot) };
        unsafe { memory::memcpy_dtoh_sync(&mut values, source as u64) }
            .map_err(|error| vortex_err!("copying device buffer: {error}"))?;
    }
    Ok(values)
}

/// Read validity from a live, synchronized zero-offset Arrow array.
unsafe fn validity(array: &ArrowArray) -> VortexResult<Vec<bool>> {
    assert_eq!(array.offset, 0);
    let len = usize::try_from(array.length)?;
    // SAFETY: These exports always have a validity buffer slot.
    if unsafe { (*array.buffers).is_null() } {
        assert_eq!(array.null_count, 0);
        return Ok(vec![true; len]);
    }
    // SAFETY: The bitmap covers the live array's rows.
    let bytes = unsafe { buffer::<u8>(array, 0, len.div_ceil(8)) }?;
    let bits: Vec<_> = (0..len)
        .map(|i| bytes[i / 8] & (1 << (i % 8)) != 0)
        .collect();
    assert_eq!(
        usize::try_from(array.null_count)?,
        bits.iter().filter(|bit| !**bit).count()
    );
    Ok(bits)
}

/// Decode the fixture's synchronized Utf8 or dictionary<Utf8> array through the public ABI.
unsafe fn read_strings(array: &ArrowArray, dtype: &DataType) -> VortexResult<Vec<Option<String>>> {
    let len = usize::try_from(array.length)?;
    // SAFETY: The fixture supplies a live synchronized array and its matching physical schema.
    let valid = unsafe { validity(array) }?;
    if let DataType::Dictionary(codes_type, values_type) = dtype {
        // SAFETY: A dictionary schema requires a live dictionary child in the matching array.
        let dictionary = unsafe { array.dictionary.as_ref() }.expect("missing dictionary");
        let values = unsafe { read_strings(dictionary, values_type) }?;
        // SAFETY: The matching schema determines the signed codes buffer's element width.
        let codes: Vec<i64> = unsafe {
            match codes_type.as_ref() {
                DataType::Int16 => buffer::<i16>(array, 1, len)?
                    .into_iter()
                    .map(i64::from)
                    .collect(),
                DataType::Int32 => buffer::<i32>(array, 1, len)?
                    .into_iter()
                    .map(i64::from)
                    .collect(),
                DataType::Int64 => buffer::<i64>(array, 1, len)?,
                other => panic!("unexpected dictionary index type: {other}"),
            }
        };
        return codes
            .into_iter()
            .zip(valid)
            .map(|(code, valid)| {
                if valid {
                    Ok(values[usize::try_from(code)?].clone())
                } else {
                    Ok(None)
                }
            })
            .collect();
    }
    assert_eq!(dtype, &DataType::Utf8);
    assert!(
        array.dictionary.is_null(),
        "plain batches must not acquire identity dictionaries"
    );
    // SAFETY: A zero-offset Utf8 array has len + 1 i32 offsets and that many bytes of data.
    let offsets = unsafe { buffer::<i32>(array, 1, len + 1) }?;
    let bytes = unsafe { buffer::<u8>(array, 2, usize::try_from(offsets[len])?) }?;
    offsets
        .windows(2)
        .zip(valid)
        .map(|(offsets, valid)| {
            if valid {
                Ok(Some(
                    std::str::from_utf8(
                        &bytes[usize::try_from(offsets[0])?..usize::try_from(offsets[1])?],
                    )
                    .map_err(VortexError::external)?
                    .to_owned(),
                ))
            } else {
                Ok(None)
            }
        })
        .collect()
}

#[cuda_test]
fn test_per_batch_dictionaries_widths_plain_and_owned_lifetimes() -> VortexResult<()> {
    let session = session()?;
    let first = dictionary(primitive([1u8, 0, 1]), &[Some("red"), Some("blue")])?;
    let second = dictionary(
        primitive([0u8, 1, 0]),
        &[Some("green"), Some("a longer dictionary value")],
    )?;
    let third = strings(&[Some("plain"), None, Some("no identity dictionary")]);
    let fourth = dictionary(primitive([1u16, 0, 1]), &[Some("small"), Some("wide")])?;
    let fifth = dictionary(primitive([0u32, 1, 0]), &[Some("widest"), Some("last")])?;
    let inputs = vec![first, second, third, fourth, fifth];
    let expected = [
        strings(&[Some("blue"), Some("red"), Some("blue")]),
        strings(&[
            Some("green"),
            Some("a longer dictionary value"),
            Some("green"),
        ]),
        inputs[2].clone(),
        strings(&[Some("wide"), Some("small"), Some("wide")]),
        strings(&[Some("widest"), Some("last"), Some("widest")]),
    ];
    let mut handle = batches(&session, inputs.into_iter().map(table).collect())?;
    assert_eq!(
        session.get::<CudaSession>().dictionary_export(),
        DictionaryExport::Decode
    );
    let mut outputs = Vec::new();
    for _ in 0..expected.len() {
        outputs.push(next(&mut handle)?);
    }
    assert!(next(&mut handle)?.is_eos());
    // SAFETY: Transfer the exclusively owned handle to its matching destructor on this thread.
    unsafe { vx_cuda_device_batches_free(Box::into_raw(Box::new(handle))) };
    drop(session);

    let index_types = [
        Some(DataType::Int16),
        Some(DataType::Int16),
        None,
        Some(DataType::Int32),
        Some(DataType::Int64),
    ];
    let mut ctx = VortexSession::default().create_execution_ctx();
    for ((output, expected), index_type) in outputs.iter().zip(expected).zip(index_types) {
        let schema = Schema::try_from(&output.schema)?;
        assert_eq!(schema.fields().len(), 1);
        let dtype = schema.field(0).data_type();
        let expected_dtype = index_type.map_or(DataType::Utf8, |index| {
            DataType::Dictionary(Box::new(index), Box::new(DataType::Utf8))
        });
        assert_eq!(dtype, &expected_dtype);
        let _context = synchronize(&output.array)?;
        assert_eq!(output.array.array.n_children, 1);
        // SAFETY: This live table batch has one child, described by the checked schema.
        let actual = unsafe { read_strings(&**output.array.array.children, dtype) }?;
        assert_arrays_eq!(
            VarBinViewArray::from_iter_nullable_str(actual).into_array(),
            expected,
            &mut ctx
        );
    }
    // Schema and array ownership are independent, even after freeing the producer.
    let mut output = outputs.pop().unwrap();
    drop(mem::replace(&mut output.schema, FFI_ArrowSchema::empty()));
    let _context = synchronize(&output.array)?;
    Ok(())
}

/// Upload fixture buffers without decoding to exercise device-resident dictionary export.
fn upload(array: ArrayRef, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    let mut slots = Vec::new();
    for slot in array.slots().iter() {
        slots.push(match slot {
            Some(child) => Some(upload(child.clone(), ctx)?),
            None => None,
        });
    }
    let mut buffers = Vec::new();
    for buffer in array.buffer_handles() {
        buffers.push(ctx.ensure_on_device_sync(buffer)?);
    }
    // SAFETY: The fixture's slots and buffers are unchanged except for their memory placement.
    unsafe { array.with_slots(slots.into())?.with_buffers(buffers) }
}

#[cuda_test]
fn test_numeric_dictionary_preserved_alongside_strings() -> VortexResult<()> {
    let session = session()?;
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
    let codes = primitive([1u8, 0, 1]);
    let numbers = DictArray::try_new(codes.clone(), primitive([10i32, 30]))?.into_array();
    let text = dictionary(codes, &[Some("red"), Some("blue")])?;
    let input = StructArray::new(
        ["number", "text"].into(),
        vec![numbers, text],
        3,
        Validity::NonNullable,
    )
    .into_array();
    let input = upload(input, &mut ctx)?;
    assert!(!input.is_host());
    ctx.synchronize_stream()?;
    let mut handle = batches(&session, vec![input])?;
    let output = next(&mut handle)?;
    let schema = Schema::try_from(&output.schema)?;
    assert_eq!(schema.fields().len(), 2);
    for (field, value_type) in schema
        .fields()
        .iter()
        .zip([DataType::Int32, DataType::Utf8])
    {
        assert_eq!(
            field.data_type(),
            &DataType::Dictionary(Box::new(DataType::Int16), Box::new(value_type))
        );
    }
    let _context = synchronize(&output.array)?;
    assert_eq!(output.array.array.length, 3);
    assert_eq!(output.array.array.n_children, 2);
    // SAFETY: The checked schema describes the live, synchronized dictionary children.
    let numbers = unsafe { &**output.array.array.children };
    let text = unsafe { &**output.array.array.children.add(1) };
    let values = unsafe { numbers.dictionary.as_ref() }.expect("missing numeric dictionary");
    assert_eq!(numbers.length, 3);
    assert_eq!(values.length, 2);
    // SAFETY: The numeric dictionary has three i16 indices and two i32 dictionary values.
    unsafe {
        assert_eq!(validity(numbers)?, vec![true; 3]);
        assert_eq!(validity(values)?, vec![true; 2]);
        assert_eq!(buffer::<i16>(numbers, 1, 3)?, vec![1, 0, 1]);
        assert_eq!(buffer::<i32>(values, 1, 2)?, vec![10, 30]);
    }
    // SAFETY: The second field describes the live, synchronized string dictionary.
    let actual = unsafe { read_strings(text, schema.field(1).data_type()) }?;
    assert_arrays_eq!(
        VarBinViewArray::from_iter_nullable_str(actual).into_array(),
        strings(&[Some("blue"), Some("red"), Some("blue")]),
        ctx.execution_ctx()
    );
    assert!(next(&mut handle)?.is_eos());
    Ok(())
}

#[cuda_test]
fn test_null_dictionary_values_and_codes() -> VortexResult<()> {
    let session = session()?;
    let input = dictionary(
        PrimitiveArray::from_option_iter([Some(0u8), Some(1), None, Some(2), Some(1)]).into_array(),
        &[Some("valid"), None, Some("also valid")],
    )?;
    let mut handle = batches(&session, vec![input])?;
    let output = next(&mut handle)?;
    let field = Field::try_from(&output.schema)?;
    assert!(field.is_nullable());
    let _context = synchronize(&output.array)?;
    // SAFETY: The schema describes this live, synchronized dictionary array.
    let actual = unsafe { read_strings(&output.array.array, field.data_type()) }?;
    assert_arrays_eq!(
        VarBinViewArray::from_iter_nullable_str(actual).into_array(),
        strings(&[Some("valid"), None, None, Some("also valid"), None]),
        handle.ctx.execution_ctx()
    );
    // Both the dictionary child bitmap and the logical row bitmap must survive Preserve export.
    assert_eq!(output.array.array.null_count, 3);
    // SAFETY: Both bitmaps belong to the live, synchronized dictionary export.
    unsafe {
        assert_eq!(
            validity(&*output.array.array.dictionary)?,
            vec![true, false, true]
        );
        assert_eq!(
            validity(&output.array.array)?,
            vec![true, false, false, true, false]
        );
    }
    Ok(())
}

#[cuda_test]
fn test_zero_row_batch_and_fused_eos() -> VortexResult<()> {
    let session = session()?;
    let empty = primitive::<i32>([]);
    let mut handle = batches(&session, vec![empty, primitive([7i32])])?;
    let output = next(&mut handle)?;
    assert!(!output.is_eos());
    assert_eq!(output.array.array.length, 0);
    assert_eq!(next(&mut handle)?.array.array.length, 1);
    let device_id = handle.ctx.stream().context().ordinal() as i64;
    let first_eos = next(&mut handle)?;
    assert_eos(&first_eos, device_id);
    let repeated_eos = next(&mut handle)?;
    assert_eos(&repeated_eos, device_id);
    Ok(())
}

#[cuda_test]
fn test_empty_stream_and_fused_eos() -> VortexResult<()> {
    let session = session()?;
    let empty = primitive::<i32>([]);
    let stream = ArrayStreamAdapter::new(empty.dtype().clone(), stream::empty()).boxed();
    let mut handle = vx_cuda_device_batches::new(stream, &session, ffi_runtime())?;
    let device_id = handle.ctx.stream().context().ordinal() as i64;
    let first_eos = next(&mut handle)?;
    assert_eos(&first_eos, device_id);
    let repeated_eos = next(&mut handle)?;
    assert_eos(&repeated_eos, device_id);
    Ok(())
}

/// Assert released outputs with the expected CUDA device metadata.
///
/// Device fields remain populated even though end-of-stream carries no data.
fn assert_eos(output: &Output, device_id: i64) {
    assert!(output.is_eos());
    assert_eq!(output.array.device_type, ARROW_DEVICE_CUDA);
    assert_eq!(output.array.device_id, device_id);
}

fn assert_terminal(
    handle: &mut vx_cuda_device_batches,
    expected_error: &str,
    polls: &AtomicUsize,
) -> VortexResult<()> {
    let mut output = next(handle)?;
    let schema = Field::try_from(&output.schema)?;
    let private_data = output.array.array.private_data;
    for expected in [expected_error, "terminal", "terminal"] {
        let error = next_into(handle, &mut output).expect_err("failed batch must return an error");
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(Field::try_from(&output.schema)?, schema);
        assert_eq!(output.array.array.private_data, private_data);
        assert_eq!(
            polls.load(Ordering::Relaxed),
            2,
            "retry consumed more input"
        );
    }
    Ok(())
}

#[cuda_test]
fn test_scan_dtype_export_errors_and_panics_are_terminal() -> VortexResult<()> {
    let session = session()?;
    for failure in ["scan", "dtype", "export", "panic"] {
        let seed = if failure == "export" {
            DecimalArray::from_iter([1i64], DecimalDType::new(10, 2)).into_array()
        } else {
            primitive([1i32])
        };
        let mut handle = batches(&session, vec![seed.clone()])?;
        let bad = match failure {
            "scan" | "panic" => Err(vortex_err!("injected scan failure")),
            "dtype" => Ok(primitive([1i64])),
            // Same logical decimal dtype, but its physical storage requires unsupported narrowing.
            "export" => Ok(DecimalArray::from_iter([1i128], DecimalDType::new(10, 2)).into_array()),
            _ => unreachable!(),
        };
        let polls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&polls);
        // Deliberately bypass ArrayStreamAdapter's debug dtype assertion to reach the handle's
        // own guard in both debug and release builds. The trailing batch must never be consumed.
        handle.array_iter = Box::new(vec![Ok(seed.clone()), bad, Ok(seed)].into_iter().inspect(
            move |_| {
                let index = counter.fetch_add(1, Ordering::Relaxed);
                assert!(
                    !(failure == "panic" && index == 1),
                    "injected polling panic"
                );
            },
        ));
        let message = match failure {
            "scan" => "injected scan failure",
            "dtype" => "logical dtype changed",
            "export" => "narrowing would require",
            "panic" => "injected polling panic",
            _ => unreachable!(),
        };
        assert_terminal(&mut handle, message, &polls)?;
    }
    Ok(())
}

#[cuda_test]
fn test_device_check_rejects_non_cuda_export() -> VortexResult<()> {
    let session = session()?;
    let mut handle = batches(&session, vec![primitive([42i32])])?;
    let mut output = next(&mut handle)?;
    handle.check_device(&output.array)?;
    output.array.device_type = 0;
    let error = handle
        .check_device(&output.array)
        .expect_err("non-CUDA export must be rejected");
    assert!(error.to_string().contains("non-CUDA device type"));
    Ok(())
}

#[cuda_test]
fn test_device_check_rejects_different_cuda_device() -> VortexResult<()> {
    let session = session()?;
    let mut handle = batches(&session, vec![primitive([42i32])])?;
    let mut output = next(&mut handle)?;
    handle.check_device(&output.array)?;
    output.array.device_id += 1;
    let error = handle
        .check_device(&output.array)
        .expect_err("export from a different CUDA device must be rejected");
    assert!(error.to_string().contains("moved from CUDA device"));
    Ok(())
}

#[cuda_test]
fn test_invalid_output_pointers_do_not_consume_or_poison() -> VortexResult<()> {
    let session = session()?;
    let mut handle = batches(&session, vec![primitive([42i32])])?;
    let mut output = Output::empty();
    output.array.array.length = 123;
    let schema = &raw mut output.schema;
    let array = &raw mut output.array;
    for (schema, array) in [
        (ptr::null_mut(), array),
        (schema, ptr::null_mut()),
        (ptr::without_provenance_mut::<FFI_ArrowSchema>(1), array),
        (schema, ptr::without_provenance_mut::<ArrowDeviceArray>(1)),
    ] {
        let mut error = ptr::null_mut();
        // SAFETY: Deliberately rejected null/unaligned pointers are never dereferenced; the other
        // output and the handle are valid. These failures must not poll or poison the handle.
        let status =
            unsafe { vx_cuda_device_batches_next(&raw mut handle, schema, array, &raw mut error) };
        assert_eq!(status, VX_CUDA_ERR);
        assert!(take_error(error).is_some());
        assert!(output.schema.release().is_none());
        assert_eq!(output.array.array.length, 123);
    }
    assert_eq!(next(&mut handle)?.array.array.length, 1);
    assert!(next(&mut handle)?.is_eos());
    Ok(())
}

#[test]
fn test_null_and_unaligned_handles_without_cuda() {
    for handle in [
        ptr::null_mut(),
        ptr::without_provenance_mut::<vx_cuda_device_batches>(1),
    ] {
        let mut output = Output::empty();
        output.array.array.length = 123;
        let mut error = ptr::null_mut();
        // SAFETY: Deliberately rejected handle pointers are never dereferenced; outputs are valid.
        let status = unsafe {
            vx_cuda_device_batches_next(
                handle,
                &raw mut output.schema,
                &raw mut output.array,
                &raw mut error,
            )
        };
        assert_eq!(status, VX_CUDA_ERR);
        assert!(take_error(error).is_some());
        assert!(output.schema.release().is_none());
        assert_eq!(output.array.array.length, 123);
    }
    // SAFETY: Null is explicitly supported, including when error details are discarded.
    unsafe {
        vx_cuda_device_batches_free(ptr::null_mut());
        assert_eq!(
            vx_cuda_device_batches_next(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut()
            ),
            VX_CUDA_ERR
        );
    }
}

fn view(value: &str) -> vx_view {
    vx_view {
        ptr: value.as_ptr().cast(),
        len: value.len(),
    }
}

#[test]
fn test_open_rejects_invalid_inputs_before_cuda() {
    let invalid_options = vx_cuda_scan_options {
        flags: u32::MAX,
        batch_rows: 0,
    };
    let duplicate_names = [view("text"), view("text")];
    let cases = [
        (
            &raw const invalid_options,
            ptr::null(),
            0,
            "unsupported CUDA scan option flags",
        ),
        (ptr::null(), ptr::null(), 1, "null CUDA scan columns"),
        (
            ptr::null(),
            duplicate_names.as_ptr(),
            2,
            "duplicate CUDA scan column",
        ),
        (ptr::null(), ptr::null(), 0, "null vx_session"),
    ];
    for (options, columns, ncolumns, message) in cases {
        let mut error = ptr::null_mut();
        // SAFETY: All borrowed inputs are live; null inputs are rejected before opening a file
        // or initializing CUDA. Only the nonzero column count intentionally lacks its array.
        let handle = unsafe {
            vx_cuda_scan_path_device_batches(
                ptr::null(),
                view("unused.vortex"),
                options,
                columns,
                ncolumns,
                &raw mut error,
            )
        };
        assert!(handle.is_null());
        let error = take_error(error).expect("missing error");
        assert!(error.contains(message), "{error}");
    }
}

#[cuda_test]
fn test_path_projection_batch_rows_and_session_lifetime() -> VortexResult<()> {
    let session = session()?.with_handle(ffi_runtime().handle());
    let input = StructArray::new(
        ["unused", "ids"].into(),
        vec![
            primitive([90i32, 91, 92, 93, 94]),
            primitive([0i32, 1, 2, 3, 4]),
        ],
        5,
        Validity::NonNullable,
    );
    let mut bytes = ByteBufferMut::empty();
    ffi_runtime().block_on(
        session
            .write_options()
            .with_strategy(Arc::new(FlatLayoutStrategy::default()))
            .write(&mut bytes, input.into_array().to_array_stream()),
    )?;
    let mut file = NamedTempFile::new()?;
    file.write_all(&bytes)?;
    let path = file
        .path()
        .to_str()
        .ok_or_else(|| vortex_err!("non-UTF-8 test path"))?;
    let columns = [view("ids")];
    let options = vx_cuda_scan_options {
        batch_rows: 2,
        ..Default::default()
    };
    let session_handle = Box::into_raw(Box::new(session)).cast::<vx_session>();
    let mut error = ptr::null_mut();
    // SAFETY: The session handle and all borrowed views/options are valid for this call.
    let handle = unsafe {
        vx_cuda_scan_path_device_batches(
            session_handle,
            view(path),
            &raw const options,
            columns.as_ptr(),
            columns.len(),
            &raw mut error,
        )
    };
    // SAFETY: The scan must retain its own session resources, independent of this owned handle.
    unsafe { vx_session_free(session_handle) };
    assert!(take_error(error).is_none());
    assert!(!handle.is_null());
    // SAFETY: Take ownership of the returned handle on its creating thread.
    let mut handle = unsafe { Box::from_raw(handle) };
    let mut ids = Vec::new();
    for rows in [2, 2, 1] {
        let output = next(&mut handle)?;
        assert_eq!(output.array.array.length, rows);
        let schema = Schema::try_from(&output.schema)?;
        assert_eq!(
            schema,
            Schema::new(vec![Field::new("ids", DataType::Int32, false)])
        );
        let _context = synchronize(&output.array)?;
        // SAFETY: The checked schema describes the live synchronized table's sole i32 child.
        unsafe {
            ids.extend(buffer::<i32>(
                &**output.array.array.children,
                1,
                usize::try_from(rows)?,
            )?);
        }
    }
    assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    assert!(next(&mut handle)?.is_eos());
    Ok(())
}

#[cuda_test]
fn test_existing_device_stream_still_decodes_mixed_batches() -> VortexResult<()> {
    let session = VortexSession::default().with_some(
        CudaSession::try_default()?.with_varbin_export_layout(VarBinExportLayout::VarBin),
    );
    assert_eq!(
        session.get::<CudaSession>().dictionary_export(),
        DictionaryExport::Preserve
    );
    let arrays = vec![
        dictionary(primitive([0u8]), &[Some("dictionary")])?,
        strings(&[Some("plain")]),
        dictionary(primitive([0u16]), &[Some("wide")])?,
    ];
    let stream = ArrayStreamAdapter::new(
        arrays[0].dtype().clone(),
        stream::iter(arrays.into_iter().map(Ok)),
    )
    .boxed();
    let mut stream = ArrowDeviceArrayStream::new(stream, scan_export_ctx(&session)?, ffi_runtime());
    let mut schema = FFI_ArrowSchema::empty();
    // SAFETY: The stream owns its callbacks and all outputs are writable and released below.
    unsafe {
        assert_eq!(
            stream.get_schema.unwrap()(&raw mut stream, (&raw mut schema).cast()),
            0
        );
        assert_eq!(Field::try_from(&schema)?.data_type(), &DataType::Utf8);
        for expected in ["dictionary", "plain", "wide"] {
            let mut array = ArrowDeviceArray::empty();
            assert_eq!(stream.get_next.unwrap()(&raw mut stream, &raw mut array), 0);
            let _context = synchronize(&array)?;
            let values = read_strings(&array.array, &DataType::Utf8)?;
            release_device_array(&mut array);
            assert_eq!(values, vec![Some(expected.to_owned())]);
        }
        stream.release.unwrap()(&raw mut stream);
    }
    Ok(())
}
