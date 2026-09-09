// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::ffi::FFI_ArrowSchema;
use futures::future::BoxFuture;
use futures::stream;
use rstest::rstest;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Constant;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::ListArray;
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
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;

use super::ARROW_DEVICE_CUDA;
use super::ArrowArray;
use super::ArrowDeviceArray;
use super::ArrowDeviceArrayStream;
use super::DeviceArrayExt;
use super::DeviceArrayStreamExt;
use super::LIBC_EIO;
use super::PrivateData;
use super::release_device_array;
use super::tests::last_error;
use crate::CudaBufferExt;
use crate::CudaExecutionCtx;
use crate::CudaSession;
use crate::DictionaryExport;

// Keep the compressed encoding tree intact while moving every buffer, including validity, to
// CUDA. Any unsupported decode must now error, rather than silently using the CPU executor.
pub(super) fn upload(
    array: ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> BoxFuture<'_, VortexResult<ArrayRef>> {
    Box::pin(async move {
        // Constants store scalar metadata, not replaceable data buffers.
        if array.as_opt::<Constant>().is_some() {
            return Ok(array);
        }
        let mut slots = Vec::new();
        for slot in array.slots().iter() {
            slots.push(match slot {
                Some(child) => Some(upload(child.clone(), ctx).await?),
                None => None,
            });
        }
        let mut buffers = Vec::new();
        for buffer in array.buffer_handles() {
            buffers.push(ctx.ensure_on_device(buffer).await?);
        }
        // SAFETY: Slots and buffers are byte-for-byte copies; only their placement changes.
        unsafe { array.with_slots(slots.into())?.with_buffers(buffers) }
    })
}

fn buffer(array: &ArrowArray, index: usize) -> VortexResult<ByteBuffer> {
    // SAFETY: Only called on live arrays produced by our exporter, before their release.
    let private = unsafe { &*array.private_data.cast::<PrivateData>() };
    let buffer = private.buffers[index]
        .as_ref()
        .ok_or_else(|| vortex_err!("missing exported buffer {index}"))?;
    buffer.cuda_device_ptr()?;
    buffer.try_to_host_sync()
}

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
            let offsets = PrimitiveArray::from_byte_buffer(
                buffer(array, 1)?,
                PType::I32,
                Validity::NonNullable,
            )
            .into_array();
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

fn get_schema(stream: &mut ArrowDeviceArrayStream) -> VortexResult<FFI_ArrowSchema> {
    let callback = stream
        .get_schema
        .ok_or_else(|| vortex_err!("missing get_schema"))?;
    let mut schema = FFI_ArrowSchema::empty();
    // SAFETY: The stream and output schema are live and writable.
    let status = unsafe { callback(stream, (&raw mut schema).cast()) };
    assert_eq!(status, 0, "{}", last_error(stream)?);
    Ok(schema)
}

fn get_next(stream: &mut ArrowDeviceArrayStream) -> VortexResult<(i32, ArrowDeviceArray)> {
    let callback = stream
        .get_next
        .ok_or_else(|| vortex_err!("missing get_next"))?;
    let mut array = ArrowDeviceArray::empty();
    // SAFETY: The stream and output array are live and writable.
    let status = unsafe { callback(stream, &raw mut array) };
    Ok((status, array))
}

fn release_stream(stream: &mut ArrowDeviceArrayStream) -> VortexResult<()> {
    let release = stream
        .release
        .ok_or_else(|| vortex_err!("missing release"))?;
    // SAFETY: This callback belongs to this live stream, and this is its final use.
    unsafe { release(stream) };
    Ok(())
}

#[rstest]
#[crate::test]
fn test_decode_mixed_dictionary_device_stream(
    #[values(false, true)] strings: bool,
    #[values(false, true)] nested: bool,
    #[values(false, true)] plain_first: bool,
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
    let expected = if nested {
        wrap_struct(expected)
    } else {
        expected
    };
    let chunks = runtime.block_on(async {
        let mut device_chunks = Vec::new();
        for chunk in chunks {
            let chunk = if nested { wrap_struct(chunk) } else { chunk };
            let chunk = upload(chunk, &mut ctx).await?;
            assert!(!chunk.is_host());
            device_chunks.push(Ok(chunk));
        }
        ctx.synchronize_stream()?;
        Ok::<_, vortex::error::VortexError>(device_chunks)
    })?;
    let mut stream = ArrayStreamAdapter::new(expected.dtype().clone(), stream::iter(chunks))
        .boxed()
        .export_device_array_stream(&session, &runtime)?;
    let schema = get_schema(&mut stream)?;
    let mut plain = runtime.block_on(expected.clone().export_device_array_with_schema(&mut ctx))?;
    assert_eq!(Field::try_from(&schema)?, Field::try_from(&plain.schema)?);
    release_device_array(&mut plain.array);
    for _ in 0..4 {
        let (status, mut array) = get_next(&mut stream)?;
        assert_eq!(status, 0, "{}", last_error(&mut stream)?);
        assert_eq!(array.device_type, ARROW_DEVICE_CUDA);
        let actual = read_plain(&array.array, expected.dtype())?;
        assert_arrays_eq!(actual, expected, ctx.execution_ctx());
        release_device_array(&mut array);
    }
    let (status, eos) = get_next(&mut stream)?;
    assert_eq!(status, 0);
    assert!(eos.array.release.is_none());
    release_stream(&mut stream)
}

#[rstest]
#[crate::test]
async fn test_decode_dictionary_without_schema(
    #[values(false, true)] strings: bool,
) -> VortexResult<()> {
    let session = vortex::array::array_session()
        .with_some(CudaSession::try_default()?.with_dictionary_export(DictionaryExport::Decode));
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
    let (values, expected) = values_and_expected(strings);
    let array = upload(dictionary(values, PType::U8)?, &mut ctx).await?;
    let mut exported = array.export_device_array(&mut ctx).await?;
    let actual = read_plain(&exported.array, expected.dtype())?;
    assert_arrays_eq!(actual, expected, ctx.execution_ctx());
    release_device_array(&mut exported);
    Ok(())
}

#[rstest]
#[case::list(0)]
#[case::fixed_size_list(1)]
#[case::non_contiguous_list_view(2)]
#[crate::test]
async fn test_decode_dictionary_list_child(
    #[case] layout: usize,
    #[values(false, true)] strings: bool,
) -> VortexResult<()> {
    let session = vortex::array::array_session()
        .with_some(CudaSession::try_default()?.with_dictionary_export(DictionaryExport::Decode));
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
    let (values, mut expected) = values_and_expected(strings);
    let elements = dictionary(values, PType::U8)?;
    let array = match layout {
        0 => ListArray::try_new(
            elements,
            PrimitiveArray::from_iter([0i32, 2, 4]).into_array(),
            Validity::NonNullable,
        )?
        .into_array(),
        1 => FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?.into_array(),
        _ => {
            expected = expected.take(PrimitiveArray::from_iter([2u32, 3, 0, 1]).into_array())?;
            ListViewArray::new(
                elements,
                PrimitiveArray::from_iter([2i32, 0]).into_array(),
                PrimitiveArray::from_iter([2i32, 2]).into_array(),
                Validity::NonNullable,
            )
            .into_array()
        }
    };
    let array = upload(array, &mut ctx).await?;
    let mut exported = array.export_device_array_with_schema(&mut ctx).await?;
    assert_eq!(
        Field::try_from(&exported.schema)?,
        Field::new_list(
            "",
            Field::new(
                Field::LIST_FIELD_DEFAULT_NAME,
                if strings {
                    DataType::Utf8
                } else {
                    DataType::Int32
                },
                expected.dtype().is_nullable()
            ),
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
    let child = unsafe { &**exported.array.array.children };
    let actual = read_plain(child, expected.dtype())?;
    assert_arrays_eq!(actual, expected, ctx.execution_ctx());
    release_device_array(&mut exported.array);
    Ok(())
}

#[crate::test]
async fn test_decode_unsupported_device_dictionary_does_not_fall_back_to_cpu() -> VortexResult<()> {
    let cuda = CudaSession::try_default()?;
    let session = vortex::array::array_session().with_some(cuda.clone());
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
    // A dictionary of structs can be preserved, but has no CUDA gather kernel today.
    let (values, _) = values_and_expected(false);
    let array = upload(dictionary(wrap_struct(values), PType::U8)?, &mut ctx).await?;
    assert!(!array.is_host());
    let mut preserved = array.clone().export_device_array(&mut ctx).await?;
    assert!(!preserved.array.dictionary.is_null());
    release_device_array(&mut preserved);

    let session = vortex::array::array_session()
        .with_some(cuda.with_dictionary_export(DictionaryExport::Decode));
    let mut ctx = CudaSession::create_execution_ctx(&session)?;
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
#[case::same_dictionary_width(Some(PType::U8), false)]
#[case::different_dictionary_width(Some(PType::U16), true)]
#[case::plain_chunk(None, true)]
#[crate::test]
fn test_default_dictionary_device_stream(
    #[case] second_width: Option<PType>,
    #[case] reject: bool,
) -> VortexResult<()> {
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
    let chunks = runtime.block_on(async {
        let chunks = vec![
            Ok(upload(first, &mut ctx).await?),
            Ok(upload(second, &mut ctx).await?),
        ];
        ctx.synchronize_stream()?;
        Ok::<_, vortex::error::VortexError>(chunks)
    })?;
    let mut stream = ArrayStreamAdapter::new(expected.dtype().clone(), stream::iter(chunks))
        .boxed()
        .export_device_array_stream(&session, &runtime)?;
    let schema = get_schema(&mut stream)?;
    assert_eq!(
        Field::try_from(&schema)?.data_type(),
        &DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Int32),)
    );
    let (status, mut first) = get_next(&mut stream)?;
    assert_eq!(status, 0);
    assert!(!first.array.dictionary.is_null());
    release_device_array(&mut first);
    let (status, mut second) = get_next(&mut stream)?;
    if reject {
        assert_eq!(status, LIBC_EIO);
        assert!(last_error(&mut stream)?.contains("Arrow schema changed"));
        assert!(second.array.release.is_none());
    } else {
        assert_eq!(status, 0);
        assert!(!second.array.dictionary.is_null());
        release_device_array(&mut second);
    }
    release_stream(&mut stream)
}
