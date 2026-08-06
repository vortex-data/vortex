// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![expect(clippy::cast_possible_truncation)]

use rstest::rstest;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Zstd;
use crate::ZstdArray;
use crate::ZstdData;
use crate::ZstdFrameMetadata;
use crate::ZstdMetadata;

#[test]
fn test_zstd_compress_decompress() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::from_iter(data.clone());

    let compressed = Zstd::from_primitive(&array, 3, 0, &mut ctx).unwrap();
    // this data should be compressible
    assert!(compressed.frames.len() < array.into_array().nbytes() as usize);
    assert!(compressed.dictionary.is_none());

    // check full decompression works
    let decompressed = Zstd::decompress(&compressed, &mut ctx).unwrap();
    assert_arrays_eq!(decompressed, PrimitiveArray::from_iter(data), &mut ctx);

    // check slicing works
    let slice = compressed.slice(100..105).unwrap();
    for i in 0_i32..5 {
        assert_nth_scalar!(slice, i as usize, 100 + i, &mut ctx);
    }
    assert_arrays_eq!(
        slice,
        PrimitiveArray::from_iter([100, 101, 102, 103, 104]),
        &mut ctx
    );

    let slice = compressed.slice(200..200).unwrap();
    assert_arrays_eq!(
        slice,
        PrimitiveArray::from_iter(Vec::<i32>::new()),
        &mut ctx
    );
}

#[test]
fn test_zstd_empty() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<i32> = vec![];
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let compressed = Zstd::from_primitive(&array, 3, 100, &mut ctx).unwrap();

    assert_arrays_eq!(compressed, PrimitiveArray::from_iter(data), &mut ctx);
}

#[test]
fn test_zstd_with_validity_and_multi_frame() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<i32> = (0..200).collect();
    let mut validity: Vec<bool> = vec![false; 200];
    validity[3] = true;
    validity[177] = true;
    let array = PrimitiveArray::new(
        Buffer::from(data),
        Validity::Array(BoolArray::from_iter(validity).into_array()),
    );

    let compressed = Zstd::from_primitive(&array, 0, 30, &mut ctx).unwrap();
    assert!(compressed.dictionary.is_none());
    assert_nth_scalar!(compressed, 0, None::<i32>, &mut ctx);
    assert_nth_scalar!(compressed, 3, 3, &mut ctx);
    assert_nth_scalar!(compressed, 10, None::<i32>, &mut ctx);
    assert_nth_scalar!(compressed, 177, 177, &mut ctx);

    let decompressed = Zstd::decompress(&compressed, &mut ctx)
        .unwrap()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let decompressed_values = decompressed.as_slice::<i32>();
    assert_eq!(decompressed_values[3], 3);
    assert_eq!(decompressed_values[177], 177);
    assert!(
        decompressed
            .validity()
            .unwrap()
            .mask_eq(&array.validity().unwrap(), decompressed.len(), &mut ctx)
            .unwrap()
    );

    // check slicing works
    let slice = compressed.slice(176..179).unwrap();
    let primitive = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();
    assert_eq!(
        i32::try_from(&primitive.execute_scalar(1, &mut ctx).unwrap()).unwrap(),
        177
    );
    assert!(
        primitive
            .validity()
            .unwrap()
            .mask_eq(
                &Validity::Array(BoolArray::from_iter(vec![false, true, false]).into_array()),
                primitive.len(),
                &mut ctx
            )
            .unwrap()
    );
}

#[test]
fn test_zstd_with_dict() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let compressed = Zstd::from_primitive(&array, 0, 16, &mut ctx).unwrap();
    assert!(compressed.dictionary.is_some());
    assert_nth_scalar!(compressed, 0, 0, &mut ctx);
    assert_nth_scalar!(compressed, 199, 199, &mut ctx);

    let decompressed = Zstd::decompress(&compressed, &mut ctx)
        .unwrap()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    assert_arrays_eq!(decompressed, PrimitiveArray::from_iter(data), &mut ctx);

    // check slicing works
    let slice = compressed.slice(176..179).unwrap();
    let primitive = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();
    assert_arrays_eq!(
        primitive,
        PrimitiveArray::from_iter([176, 177, 178]),
        &mut ctx
    );
}

#[test]
fn test_validity_vtable() {
    let mut ctx = array_session().create_execution_ctx();
    let mask_bools = vec![false, true, true, false, true];
    let array = PrimitiveArray::new(
        (0..5).collect::<Buffer<_>>(),
        Validity::Array(BoolArray::from_iter(mask_bools.clone()).into_array()),
    );
    let compressed = Zstd::from_primitive(&array, 3, 0, &mut ctx).unwrap();
    let arr = compressed.as_array();
    assert_eq!(
        arr.validity()
            .unwrap()
            .execute_mask(arr.len(), &mut array_session().create_execution_ctx())
            .unwrap(),
        Mask::from_iter(mask_bools)
    );
    let sliced = compressed.slice(1..4).unwrap();
    assert_eq!(
        sliced
            .validity()
            .unwrap()
            .execute_mask(sliced.len(), &mut array_session().create_execution_ctx())
            .unwrap(),
        Mask::from_iter(vec![true, true, false])
    );
}

#[test]
fn test_zstd_var_bin_view() {
    let mut ctx = array_session().create_execution_ctx();
    let data: [Option<&'static [u8]>; 5] = [
        Some(b"foo"),
        Some(b"bar"),
        None,
        Some(b"Lorem ipsum dolor sit amet"),
        Some(b"baz"),
    ];
    let array = VarBinViewArray::from_iter(data, DType::Utf8(Nullability::Nullable));

    let compressed = Zstd::from_var_bin_view(&array, 0, 3, &mut ctx).unwrap();
    assert!(compressed.dictionary.is_none());
    assert_nth_scalar!(compressed, 0, "foo", &mut ctx);
    assert_nth_scalar!(compressed, 1, "bar", &mut ctx);
    assert_nth_scalar!(compressed, 2, None::<String>, &mut ctx);
    assert_nth_scalar!(compressed, 3, "Lorem ipsum dolor sit amet", &mut ctx);
    assert_nth_scalar!(compressed, 4, "baz", &mut ctx);

    let sliced = compressed.slice(1..4).unwrap();
    assert_nth_scalar!(sliced, 0, "bar", &mut ctx);
    assert_nth_scalar!(sliced, 1, None::<String>, &mut ctx);
    assert_nth_scalar!(sliced, 2, "Lorem ipsum dolor sit amet", &mut ctx);
}

#[test]
fn test_zstd_append_to_offset_builder() {
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::from_iter(
        [
            Some(b"foo".as_slice()),
            Some(b"bar".as_slice()),
            None,
            Some(b"Lorem ipsum dolor sit amet".as_slice()),
            Some(b"baz".as_slice()),
        ],
        DType::Utf8(Nullability::Nullable),
    );
    let compressed = Zstd::from_var_bin_view(&array, 0, 3, &mut ctx)
        .unwrap()
        .slice(1..4)
        .unwrap();
    let mut builder =
        VarBinBuilder::<i32>::with_capacity(compressed.dtype().clone(), compressed.len());
    compressed
        .append_to_builder(&mut builder, &mut ctx)
        .unwrap();
    assert_arrays_eq!(
        builder.finish_into_varbin(),
        array.into_array().slice(1..4).unwrap(),
        &mut ctx
    );
}

/// A slice decompresses whole frames, so the frames hold values on either side of the ones it
/// requests. `append_views_built_at` publishes the buffers it is handed as they are, so only the
/// requested region may reach it — otherwise the finished array retains the whole frames.
#[test]
fn test_zstd_append_to_view_builder_keeps_only_the_sliced_bytes() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    // Long enough that the values live in the data buffers rather than inline in the views.
    let values = (0..12)
        .map(|i| format!("value number {i} padded well past the inline limit"))
        .collect::<Vec<_>>();
    let array = VarBinViewArray::from_iter_str(&values);
    // Three values per frame, so the slice below starts and ends inside a frame holding more.
    let compressed = Zstd::from_var_bin_view(&array, 0, 3, &mut ctx)?.slice(4..8)?;

    // Seeded with a value of its own so the pushed buffers land after an in-progress buffer, and
    // appended to twice so the second push has to rebase past the first.
    let mut builder = VarBinViewBuilder::with_capacity(compressed.dtype().clone(), 9);
    builder.append_value(&values[0]);
    compressed.append_to_builder(&mut builder, &mut ctx)?;
    compressed.append_to_builder(&mut builder, &mut ctx)?;
    let appended = builder.finish_into_varbinview();

    let expected = VarBinViewArray::from_iter_str(
        std::iter::once(&values[0])
            .chain(&values[4..8])
            .chain(&values[4..8]),
    );
    assert_arrays_eq!(appended, expected, &mut ctx);

    // Each stored value costs its bytes plus the u32 length prefix zstd interleaves.
    let sliced_bytes: usize = values[4..8]
        .iter()
        .map(|value| value.len() + size_of::<u32>())
        .sum();
    let buffered: usize = appended
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host().len())
        .sum();
    assert_eq!(buffered, values[0].len() + 2 * sliced_bytes);
    Ok(())
}

#[test]
fn test_zstd_decompress_var_bin_view() {
    let mut ctx = array_session().create_execution_ctx();
    let data: [Option<&'static [u8]>; 5] = [
        Some(b"foo"),
        Some(b"bar"),
        None,
        Some(b"Lorem ipsum dolor sit amet"),
        Some(b"baz"),
    ];
    let array = VarBinViewArray::from_iter(data, DType::Utf8(Nullability::Nullable));

    let compressed = Zstd::from_var_bin_view(&array, 0, 3, &mut ctx).unwrap();
    assert!(compressed.dictionary.is_none());
    assert_nth_scalar!(compressed, 0, "foo", &mut ctx);
    assert_nth_scalar!(compressed, 1, "bar", &mut ctx);
    assert_nth_scalar!(compressed, 2, None::<String>, &mut ctx);
    assert_nth_scalar!(compressed, 3, "Lorem ipsum dolor sit amet", &mut ctx);
    assert_nth_scalar!(compressed, 4, "baz", &mut ctx);

    let decompressed = Zstd::decompress(&compressed, &mut ctx)
        .unwrap()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap();
    assert_nth_scalar!(decompressed, 0, "foo", &mut ctx);
    assert_nth_scalar!(decompressed, 1, "bar", &mut ctx);
    assert_nth_scalar!(decompressed, 2, None::<String>, &mut ctx);
    assert_nth_scalar!(decompressed, 3, "Lorem ipsum dolor sit amet", &mut ctx);
    assert_nth_scalar!(decompressed, 4, "baz", &mut ctx);
}

#[test]
fn test_sliced_array_children() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<Option<i32>> = (0..10).map(|v| (v != 5).then_some(v)).collect();
    let compressed =
        Zstd::from_primitive(&PrimitiveArray::from_option_iter(data), 0, 100, &mut ctx).unwrap();
    let sliced = compressed.slice(0..4).unwrap();
    sliced.children();
}

/// Six rows, five of them stored, compressed into `values_per_frame`-sized frames with the frame
/// metadata a reader would have deserialized rewritten by `corrupt`.
fn corrupt_var_bin_view_metadata(
    values_per_frame: usize,
    corrupt: impl FnOnce(&mut ZstdMetadata),
    ctx: &mut ExecutionCtx,
) -> VortexResult<(VarBinViewArray, ZstdArray)> {
    let array = VarBinViewArray::from_iter(
        [
            Some(b"foo".as_slice()),
            Some(b"bar".as_slice()),
            None,
            Some(b"Lorem ipsum dolor sit amet".as_slice()),
            Some(b"baz".as_slice()),
            Some(b"quux".as_slice()),
        ],
        DType::Utf8(Nullability::Nullable),
    );
    let mut data = ZstdData::from_var_bin_view(&array, 0, values_per_frame, ctx)?;
    corrupt(&mut data.metadata);
    let compressed = Zstd::try_new(array.dtype().clone(), data, array.validity()?)?;
    Ok((array, compressed))
}

/// Frame metadata comes straight off disk, so an inconsistent value count has to surface as an
/// error from both read paths rather than a panic or a wrapped length.
#[rstest]
#[case::frame_holds_fewer_values_than_claimed(|metadata: &mut ZstdMetadata| {
    metadata.frames[0].n_values = 1000;
})]
#[case::frame_value_counts_overflow(|metadata: &mut ZstdMetadata| {
    metadata.frames[1].n_values = u64::MAX;
})]
#[case::missing_value_count_across_frames(|metadata: &mut ZstdMetadata| {
    metadata.frames[0].n_values = 0;
})]
fn test_zstd_rejects_corrupt_frame_metadata(
    #[case] corrupt: fn(&mut ZstdMetadata),
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let (_, compressed) = corrupt_var_bin_view_metadata(3, corrupt, &mut ctx)?;

    assert!(Zstd::decompress(&compressed, &mut ctx).is_err());

    let mut builder =
        VarBinBuilder::<i32>::with_capacity(compressed.dtype().clone(), compressed.len());
    assert!(
        compressed
            .append_to_builder(&mut builder, &mut ctx)
            .is_err()
    );
    Ok(())
}

/// Frame bytes are as untrusted as the metadata describing them: a frame whose last value is
/// nothing but a length prefix has to error from every read path, not read back as an empty value.
#[test]
fn test_zstd_rejects_a_frame_ending_in_a_dangling_length_prefix() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let mut values = Vec::new();
    values.extend_from_slice(&3u32.to_le_bytes());
    values.extend_from_slice(b"cat");
    // A prefix with no value after it. Treating the missing value as empty still totals the three
    // bytes the metadata implies for two values.
    values.extend_from_slice(&1u32.to_le_bytes());

    let dtype = DType::Utf8(Nullability::NonNullable);
    let compressed = Zstd::try_new(
        dtype.clone(),
        ZstdData::new(
            None,
            vec![ByteBuffer::from(zstd::bulk::compress(&values, 3)?)],
            ZstdMetadata {
                dictionary_size: 0,
                frames: vec![ZstdFrameMetadata {
                    uncompressed_size: values.len() as u64,
                    n_values: 2,
                }],
            },
            2,
        ),
        Validity::NonNullable,
    )?;

    assert!(Zstd::decompress(&compressed, &mut ctx).is_err());
    let mut varbin = VarBinBuilder::<i32>::with_capacity(dtype.clone(), 2);
    assert!(compressed.append_to_builder(&mut varbin, &mut ctx).is_err());
    let mut views = VarBinViewBuilder::with_capacity(dtype, 2);
    assert!(compressed.append_to_builder(&mut views, &mut ctx).is_err());
    Ok(())
}

/// Metadata written before frames recorded their value count leaves it at zero. A single frame
/// holds every stored value, so those arrays still read back.
#[test]
fn test_zstd_reads_legacy_single_frame_var_bin_metadata() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let (array, compressed) =
        corrupt_var_bin_view_metadata(0, |metadata| metadata.frames[0].n_values = 0, &mut ctx)?;

    assert_arrays_eq!(compressed, array.clone().into_array(), &mut ctx);
    assert_arrays_eq!(
        compressed.slice(2..5)?,
        array.into_array().slice(2..5)?,
        &mut ctx
    );
    Ok(())
}

/// The same legacy metadata for fixed-width values recovers the count from the frame size, which
/// stays exact across frames.
#[test]
fn test_zstd_reads_legacy_primitive_frame_metadata() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0..200_i32);
    let mut data = ZstdData::from_primitive(&array, 3, 30, &mut ctx)?;
    for frame in &mut data.metadata.frames {
        frame.n_values = 0;
    }
    let compressed = Zstd::try_new(array.dtype().clone(), data, array.validity()?)?;

    assert_arrays_eq!(compressed, PrimitiveArray::from_iter(0..200_i32), &mut ctx);
    assert_arrays_eq!(
        compressed.slice(100..105)?,
        PrimitiveArray::from_iter(100..105_i32),
        &mut ctx
    );
    Ok(())
}

/// Tests that each beginning of a frame in ZSTD matches
/// the buffer alignment when compressing primitive arrays.
#[test]
fn test_zstd_frame_start_buffer_alignment() {
    let mut ctx = array_session().create_execution_ctx();
    let data = vec![0u8; 2];
    let aligned_buffer = Buffer::copy_from_aligned(&data, Alignment::new(8));
    // u8 array now has a 8-byte alignment.
    let array = PrimitiveArray::new(aligned_buffer, Validity::NonNullable);
    let compressed = Zstd::from_primitive(&array, 0, 1, &mut ctx);

    assert!(compressed.is_ok());
}

#[test]
fn test_zstd_rejects_mismatched_frame_content_size() {
    let mut ctx = array_session().create_execution_ctx();
    let compressed =
        Zstd::from_primitive(&PrimitiveArray::from_iter([1_i32, 2, 3]), 0, 0, &mut ctx).unwrap();
    let mut data = compressed.data().clone();
    data.metadata.frames[0].uncompressed_size = 16 * 1024 * 1024 * 1024;

    let error = Zstd::try_new(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        data,
        Validity::NonNullable,
    )
    .unwrap_err();
    assert!(error.to_string().contains("metadata declares"));
}
