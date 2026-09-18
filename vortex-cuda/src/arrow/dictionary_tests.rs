// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Poll;

use futures::stream;
use rstest::rstest;
use vortex::array::IntoArray;
use vortex::array::arrays::Constant;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::assert_arrays_eq;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::error::vortex_bail;

use super::tests::last_error;
use super::*;
use crate::CudaSession;

/// Preserve encodings while moving all buffers, including validity, to CUDA so unsupported
/// decoding errors instead of falling back to the CPU.
pub(super) fn upload(array: ArrayRef, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    // Constants store scalar metadata, not replaceable data buffers.
    if array.as_opt::<Constant>().is_some() {
        return Ok(array);
    }
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
    // SAFETY: Slots and buffers are byte-for-byte copies; only their placement changes.
    unsafe { array.with_slots(slots.into())?.with_buffers(buffers) }
}

/// Copy a device buffer from a live, unreleased array produced by this exporter to the host.
fn buffer(array: &ArrowArray, index: usize) -> VortexResult<ByteBuffer> {
    // SAFETY: Only called on live arrays produced by our exporter, before their release.
    let private = unsafe { &*array.private_data.cast::<PrivateData>() };
    let buffer = private.buffers[index]
        .as_ref()
        .ok_or_else(|| vortex_err!("missing exported buffer {index}"))?;
    buffer.cuda_device_ptr()?;
    buffer.try_to_host_sync()
}

/// Rebuild supported zero-offset, dictionary-free exports as host arrays for comparison.
/// Requires live arrays from this exporter and their matching logical dtype.
fn read_plain(array: &ArrowArray, dtype: &DType) -> VortexResult<ArrayRef> {
    assert!(array.dictionary.is_null());
    assert_eq!(array.offset, 0);
    let len = usize::try_from(array.length)?;
    let validity = if !dtype.is_nullable() {
        assert_eq!(array.null_count, 0);
        Validity::NonNullable
    } else if array.null_count == 0 {
        Validity::AllValid
    } else {
        Validity::from(BitBuffer::new(buffer(array, 0)?, len))
    };
    match dtype {
        DType::Primitive(ptype, _) => {
            assert_eq!(array.n_buffers, 2);
            assert_eq!(array.n_children, 0);
            Ok(PrimitiveArray::from_byte_buffer(buffer(array, 1)?, *ptype, validity).into_array())
        }
        DType::Utf8(_) => {
            assert_eq!(array.n_buffers, 3);
            assert_eq!(array.n_children, 0);
            let offsets = Buffer::<i32>::from_byte_buffer(buffer(array, 1)?).into_array();
            Ok(VarBinArray::try_new(
                offsets,
                buffer(array, 2)?.slice_unaligned(..),
                dtype.clone(),
                validity,
            )?
            .into_array())
        }
        DType::Struct(fields, _) => {
            assert_eq!(array.n_buffers, 1);
            assert_eq!(usize::try_from(array.n_children)?, fields.nfields());
            let mut children = Vec::new();
            for (index, dtype) in fields.fields().enumerate() {
                // SAFETY: The live struct owns exactly n_children child pointers.
                let child = unsafe { &**array.children.add(index) };
                children.push(read_plain(child, &dtype)?);
            }
            Ok(StructArray::try_new(fields.names().clone(), children, len, validity)?.into_array())
        }
        _ => vortex_bail!("unsupported test dtype {dtype}"),
    }
}

fn wrap_struct(array: ArrayRef) -> ArrayRef {
    let len = array.len();
    let inner = StructArray::new(["value"].into(), vec![array], len, Validity::NonNullable);
    StructArray::new(
        ["nested"].into(),
        vec![inner.into_array()],
        len,
        Validity::NonNullable,
    )
    .into_array()
}

fn values_and_expected(strings: bool) -> (ArrayRef, ArrayRef) {
    if strings {
        let long = "an out-of-line dictionary value";
        (
            VarBinViewArray::from_iter_nullable_str([Some("short"), None, Some(long)]).into_array(),
            VarBinViewArray::from_iter_nullable_str([Some(long), None, Some("short"), Some(long)])
                .into_array(),
        )
    } else {
        (
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            PrimitiveArray::from_iter([30i32, 20, 10, 30]).into_array(),
        )
    }
}

fn dictionary(values: ArrayRef, width: PType) -> VortexResult<ArrayRef> {
    let codes = match width {
        PType::U8 => PrimitiveArray::from_iter([2u8, 1, 0, 2]).into_array(),
        PType::U16 => PrimitiveArray::from_iter([2u16, 1, 0, 2]).into_array(),
        PType::U32 => PrimitiveArray::from_iter([2u32, 1, 0, 2]).into_array(),
        _ => vortex_bail!("unsupported test index width {width}"),
    };
    Ok(DictArray::try_new(codes, values)?.into_array())
}

fn get_schema(stream: &mut ArrowDeviceArrayStream) -> VortexResult<Field> {
    let callback = stream.get_schema.expect("missing get_schema");
    let mut schema = FFI_ArrowSchema::empty();
    // SAFETY: The stream and output schema are live and writable.
    let status = unsafe { callback(stream, (&raw mut schema).cast()) };
    assert_eq!(status, 0, "{}", last_error(stream)?);
    Ok(Field::try_from(&schema)?)
}

fn get_next(stream: &mut ArrowDeviceArrayStream) -> (i32, ArrowDeviceArray) {
    let callback = stream.get_next.expect("missing get_next");
    let mut array = ArrowDeviceArray::empty();
    // SAFETY: The stream and output array are live and writable.
    let status = unsafe { callback(stream, &raw mut array) };
    (status, array)
}

/// Upload chunks and synchronize before handing them to a separate export context.
fn upload_chunks(
    chunks: Vec<ArrayRef>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Vec<VortexResult<ArrayRef>>> {
    let mut device_chunks = Vec::new();
    for chunk in chunks {
        let chunk = upload(chunk, ctx)?;
        assert!(!chunk.is_host());
        device_chunks.push(Ok(chunk));
    }
    ctx.synchronize_stream()?;
    Ok(device_chunks)
}

#[rstest]
#[case::plain_first_primitives(false, false, true)]
#[case::dictionary_first_nested_strings(true, true, false)]
#[crate::test]
fn test_decode_mixed_dictionary_device_stream(
    #[case] strings: bool,
    #[case] nested: bool,
    #[case] plain_first: bool,
    #[values(false, true)] schema_first: bool,
) -> VortexResult<()> {
    let runtime = CurrentThreadRuntime::new();
    let session = vortex::array::array_session()
        .with_some(CudaSession::try_default()?.with_dictionary_export(DictionaryExport::Decode));
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
    let (values, expected) = values_and_expected(strings);
    let nested_values = DictArray::try_new(
        PrimitiveArray::from_iter([0u8, 1, 2]).into_array(),
        values.clone(),
    )?
    .into_array();
    let mut chunks = vec![
        dictionary(values.clone(), PType::U8)?,
        dictionary(values, PType::U16)?,
        dictionary(nested_values, PType::U32)?,
        expected.clone(),
    ];
    if plain_first {
        chunks.rotate_right(1);
    }
    let wrap = |array| if nested { wrap_struct(array) } else { array };
    let expected = wrap(expected);
    let chunks = chunks.into_iter().map(wrap).collect();
    let chunks = upload_chunks(chunks, &mut ctx)?;
    let mut stream = ArrayStreamAdapter::new(expected.dtype().clone(), stream::iter(chunks))
        .boxed()
        .export_device_array_stream(&session, &runtime)?;
    let plain_schema = Field::try_from(&arrow_schema_for_array(&expected, &mut ctx)?)?;
    if schema_first {
        assert_eq!(get_schema(&mut stream)?, plain_schema);
    }
    for _ in 0..4 {
        let (status, mut array) = get_next(&mut stream);
        assert_eq!(status, 0, "{}", last_error(&mut stream)?);
        assert_eq!(array.device_type, ARROW_DEVICE_CUDA);
        let actual = read_plain(&array.array, expected.dtype())?;
        assert_arrays_eq!(actual, expected, ctx.execution_ctx());
        release_device_array(&mut array);
    }
    assert_eq!(get_schema(&mut stream)?, plain_schema);
    let (status, eos) = get_next(&mut stream);
    assert_eq!(status, 0);
    assert!(eos.array.release.is_none());
    // SAFETY: This is the live stream's final use.
    unsafe { stream.release.expect("missing release")(&raw mut stream) };
    Ok(())
}

#[rstest]
#[case::values(true, false)]
#[case::error(true, true)]
#[case::empty(false, false)]
#[crate::test]
fn test_decode_stream_schema_does_not_poll(
    #[case] has_batch: bool,
    #[case] fails: bool,
    #[values(false, true)] consume: bool,
) -> VortexResult<()> {
    let runtime = CurrentThreadRuntime::new();
    let session = vortex::array::array_session()
        .with_some(CudaSession::try_default()?.with_dictionary_export(DictionaryExport::Decode));
    let array = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
    let dtype = array.dtype().clone();
    let mut batch = has_batch.then(|| {
        if fails {
            Err(vortex_err!("deferred scan error"))
        } else {
            Ok(array)
        }
    });
    let polls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&polls);
    let input = stream::poll_fn(move |_| {
        counter.fetch_add(1, Ordering::Relaxed);
        Poll::Ready(batch.take())
    });
    let mut stream = ArrayStreamAdapter::new(dtype, input)
        .boxed()
        .export_device_array_stream(&session, &runtime)?;
    for _ in 0..2 {
        assert_eq!(
            get_schema(&mut stream)?,
            Field::new("", DataType::Int32, false)
        );
    }
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    if consume {
        let (status, mut array) = get_next(&mut stream);
        assert_eq!(polls.load(Ordering::Relaxed), 1);
        if fails {
            assert_eq!(status, LIBC_EIO);
            assert!(last_error(&mut stream)?.contains("deferred scan error"));
            assert!(array.array.release.is_none());
        } else {
            assert_eq!(status, 0, "{}", last_error(&mut stream)?);
            assert_eq!(array.array.release.is_some(), has_batch);
            if has_batch {
                assert_eq!(array.array.length, 3);
            }
            release_device_array(&mut array);
        }
    }
    // SAFETY: This is the live stream's final use, including schema-only consumers.
    unsafe { stream.release.expect("missing release")(&raw mut stream) };
    assert_eq!(polls.load(Ordering::Relaxed), usize::from(consume));
    Ok(())
}

#[crate::test]
fn test_decode_stream_validates_dtype_and_device() -> VortexResult<()> {
    let runtime = CurrentThreadRuntime::new();
    let session = vortex::array::array_session()
        .with_some(CudaSession::try_default()?.with_dictionary_export(DictionaryExport::Decode));
    let array = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
    let mut stream = array
        .to_array_stream()
        .boxed()
        .export_device_array_stream(&session, &runtime)?;
    // SAFETY: The stream is live and exclusively borrowed until the state is no longer used.
    let state = unsafe { device_stream_private_data(&raw mut stream) }.expect("missing state");
    let mut first = state.export_stream_array(array.clone())?;
    release_device_array(&mut first);
    assert!(state.schema.is_some());

    let error = state
        .export_stream_array(PrimitiveArray::from_iter([10u32, 20, 30]).into_array())
        .expect_err("accepted a different dtype");
    assert!(error.to_string().contains("stream array dtype changed"));
    let error = state.check_device(&ArrowDeviceArray::empty()).unwrap_err();
    assert!(error.to_string().contains("non-CUDA device type"));
    state.device_id = -1;
    let error = state
        .export_stream_array(array)
        .expect_err("accepted a different device");
    assert!(
        error
            .to_string()
            .contains("stream array moved from CUDA device")
    );
    // SAFETY: This is the live stream's final use; no state borrow remains.
    unsafe { stream.release.expect("missing release")(&raw mut stream) };
    Ok(())
}

#[crate::test]
async fn test_decode_non_contiguous_dictionary_list_view() -> VortexResult<()> {
    let session = vortex::array::array_session()
        .with_some(CudaSession::try_default()?.with_dictionary_export(DictionaryExport::Decode));
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
    let (values, expected) = values_and_expected(true);
    let array = ListViewArray::new(
        dictionary(values, PType::U8)?,
        PrimitiveArray::from_iter([2i32, 0]).into_array(),
        PrimitiveArray::from_iter([2i32, 2]).into_array(),
        Validity::NonNullable,
    )
    .into_array();
    let expected = expected.take(PrimitiveArray::from_iter([2u32, 3, 0, 1]).into_array())?;
    let array = upload(array, &mut ctx)?;
    let mut exported = array.export_device_array_with_schema(&mut ctx).await?;
    assert_eq!(
        Field::try_from(&exported.schema)?,
        Field::new_list(
            "",
            Field::new(Field::LIST_FIELD_DEFAULT_NAME, DataType::Utf8, true),
            false,
        )
    );
    assert_eq!(exported.array.array.length, 2);
    assert_eq!(exported.array.array.n_children, 1);
    assert_eq!(
        Buffer::<i32>::from_byte_buffer(buffer(&exported.array.array, 1)?).as_ref(),
        &[0, 2, 4]
    );
    // SAFETY: This live list array owns the single child checked above.
    let values = unsafe { &**exported.array.array.children };
    let actual = read_plain(values, expected.dtype())?;
    assert_arrays_eq!(actual, expected, ctx.execution_ctx());
    release_device_array(&mut exported.array);
    Ok(())
}

#[crate::test]
async fn test_decode_unsupported_device_dictionary_does_not_fall_back_to_cpu() -> VortexResult<()> {
    let session = vortex::array::array_session().with_some(CudaSession::try_default()?);
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
    // A dictionary of structs can be preserved, but has no CUDA gather kernel today.
    let (values, _) = values_and_expected(false);
    let array = upload(dictionary(wrap_struct(values), PType::U8)?, &mut ctx)?;
    assert!(!array.is_host());
    let mut preserved = array.clone().export_device_array(&mut ctx).await?;
    assert!(!preserved.array.dictionary.is_null());
    release_device_array(&mut preserved);

    let mut ctx = ctx.with_dictionary_export(DictionaryExport::Decode);
    let error = match array.export_device_array_with_schema(&mut ctx).await {
        Ok(mut exported) => {
            release_device_array(&mut exported.array);
            vortex_bail!("unsupported device dictionary unexpectedly decoded");
        }
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("CPU fallback with device-resident buffers is not supported")
    );
    Ok(())
}

#[rstest]
#[case::different_dictionary_width(Some(PType::U16))]
#[case::plain_chunk(None)]
#[crate::test]
fn test_default_dictionary_device_stream(#[case] second_width: Option<PType>) -> VortexResult<()> {
    let runtime = CurrentThreadRuntime::new();
    let session = crate::cuda_session();
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
    assert_eq!(
        ctx.cuda_session().dictionary_export(),
        DictionaryExport::Preserve
    );
    let (values, expected) = values_and_expected(false);
    let first = dictionary(values.clone(), PType::U8)?;
    let second = match second_width {
        Some(width) => dictionary(values, width)?,
        None => expected.clone(),
    };
    let chunks = upload_chunks(vec![first, second], &mut ctx)?;
    let mut stream = ArrayStreamAdapter::new(expected.dtype().clone(), stream::iter(chunks))
        .boxed()
        .export_device_array_stream(&session, &runtime)?;
    assert_eq!(
        get_schema(&mut stream)?.data_type(),
        &DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Int32),)
    );
    let (status, mut array) = get_next(&mut stream);
    assert_eq!(status, 0);
    assert!(!array.array.dictionary.is_null());
    release_device_array(&mut array);
    let (status, rejected) = get_next(&mut stream);
    assert_eq!(status, LIBC_EIO);
    assert!(last_error(&mut stream)?.contains("Arrow schema changed"));
    assert!(rejected.array.release.is_none());
    // SAFETY: This is the live stream's final use.
    unsafe { stream.release.expect("missing release")(&raw mut stream) };
    Ok(())
}
