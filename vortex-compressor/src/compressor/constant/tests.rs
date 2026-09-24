// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::Masked;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use super::is_constant_for_compression;
use crate::CascadingCompressor;
use crate::builtins::FloatDictScheme;
use crate::builtins::IntDictScheme;
use crate::builtins::StringDictScheme;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

/// Constant detection is built into the compressor, so it must work with no schemes at all.
fn empty_compressor() -> CascadingCompressor {
    CascadingCompressor::new(Vec::new())
}

#[test]
fn constant_int_compresses_without_schemes() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![7i64; 100], Validity::NonNullable).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn constant_int_with_nulls_compresses_to_masked_constant() -> VortexResult<()> {
    let validity =
        Validity::Array(BoolArray::from_iter((0..100).map(|i| i % 10 != 0)).into_array());
    let array = PrimitiveArray::new(buffer![7i64; 100], validity).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(compressed.is::<Masked>());
    Ok(())
}

#[test]
fn constant_string_compresses_without_schemes() -> VortexResult<()> {
    let array = VarBinViewArray::from_iter_str(std::iter::repeat_n("hello", 100)).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn constant_bool_compresses_without_schemes() -> VortexResult<()> {
    let array = BoolArray::from_iter(std::iter::repeat_n(true, 100)).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn constant_decimal_compresses_without_schemes() -> VortexResult<()> {
    let array = DecimalArray::new(
        buffer![123_456i128; 100],
        DecimalDType::new(20, 2),
        Validity::NonNullable,
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn constant_timestamp_compresses_without_schemes() -> VortexResult<()> {
    let ts = PrimitiveArray::from_iter(std::iter::repeat_n(1_704_067_200_000i64, 100));
    let array =
        TemporalArray::new_timestamp(ts.into_array(), TimeUnit::Milliseconds, None).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[rstest]
#[case::u8(PrimitiveArray::new(buffer![3u8; 1000], Validity::NonNullable).into_array())]
#[case::u32(PrimitiveArray::new(buffer![42u32; 1000], Validity::NonNullable).into_array())]
#[case::f64(PrimitiveArray::new(buffer![1.5f64; 1000], Validity::NonNullable).into_array())]
#[case::bool(BoolArray::from_iter(std::iter::repeat_n(false, 1000)).into_array())]
#[case::utf8(VarBinViewArray::from_iter_str(std::iter::repeat_n("abc", 1000)).into_array())]
#[case::decimal(
    DecimalArray::new(buffer![5i64; 1000], DecimalDType::new(10, 2), Validity::NonNullable)
        .into_array()
)]
fn constant_compresses_with_dict_schemes(#[case] array: ArrayRef) -> VortexResult<()> {
    let compressor =
        CascadingCompressor::new(vec![&IntDictScheme, &FloatDictScheme, &StringDictScheme]);
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = compressor.compress(&array, &mut ctx)?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn signed_zero_floats_are_not_constant() -> VortexResult<()> {
    let array =
        PrimitiveArray::new(buffer![0.0f64, -0.0, 0.0, -0.0], Validity::NonNullable).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(!compressed.is::<Constant>());
    Ok(())
}

/// Every tenth value is null.
fn nulls_every_tenth() -> Validity {
    Validity::Array(BoolArray::from_iter((0..100).map(|i| i % 10 != 0)).into_array())
}

#[rstest]
#[case::u32(PrimitiveArray::new(buffer![7u32; 100], nulls_every_tenth()).into_array())]
#[case::f64(PrimitiveArray::new(buffer![1.5f64; 100], nulls_every_tenth()).into_array())]
#[case::bool(
    BoolArray::new(BitBuffer::new_set(100), nulls_every_tenth()).into_array()
)]
#[case::utf8(
    VarBinViewArray::from_iter_nullable_str((0..100).map(|i| (i % 10 != 0).then_some("abc")))
        .into_array()
)]
#[case::decimal(
    DecimalArray::new(buffer![5i64; 100], DecimalDType::new(10, 2), nulls_every_tenth())
        .into_array()
)]
fn masked_constant_compresses_to_masked_constant(
    #[case] array: ArrayRef,
    #[values(false, true)] dictionary: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = if dictionary {
        CascadingCompressor::new(vec![&IntDictScheme, &FloatDictScheme, &StringDictScheme])
    } else {
        empty_compressor()
    };
    let compressed = compressor.compress(&array, &mut ctx)?;
    assert!(compressed.is::<Masked>());
    assert!(compressed.children()[0].is::<Constant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[rstest]
#[case::values_differ(PrimitiveArray::new(
    (0..100u32).collect::<Buffer<u32>>(),
    nulls_every_tenth()
))]
#[case::signed_zeros(PrimitiveArray::new(
    (0..100).map(|i| if i % 2 == 0 { 0.0f64 } else { -0.0 }).collect::<Buffer<f64>>(),
    nulls_every_tenth()
))]
#[case::nan_among_equal_values(PrimitiveArray::new(
    (0..100).map(|i| if i == 5 { f64::NAN } else { 1.5 }).collect::<Buffer<f64>>(),
    nulls_every_tenth()
))]
fn non_constant_valid_values_are_not_masked_constant(
    #[case] array: PrimitiveArray,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = array.into_array();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(!compressed.is::<Masked>());
    assert!(!compressed.is::<Constant>());
    Ok(())
}

#[test]
fn non_constant_int_is_left_canonical_without_schemes() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter(0..100i64).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(!compressed.is::<Constant>());
    assert_eq!(compressed.dtype(), array.dtype());
    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct FloatExtension;

impl ExtVTable for FloatExtension {
    type Metadata = u8;
    type NativeValue<'a> = f64;

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> ExtId {
        ExtId::new("test.float")
    }

    fn serialize_metadata(&self, metadata: &u8) -> VortexResult<Vec<u8>> {
        Ok(vec![*metadata])
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<u8> {
        vortex_ensure!(metadata.len() == 1, "expected one metadata byte");
        Ok(metadata[0])
    }

    fn validate_dtype(dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(dtype.storage_dtype().is_float(), "expected float storage");
        Ok(())
    }

    fn unpack_native<'a>(_dtype: &'a ExtDType<Self>, value: &'a ScalarValue) -> VortexResult<f64> {
        value.as_primitive().cast::<f64>()
    }
}

#[rstest]
#[case::equal(1.5, 1.5, true)]
#[case::nan_among_equal_values(1.5, f64::NAN, false)]
#[case::equal_nans(f64::NAN, f64::NAN, true)]
#[case::different_nan_payloads(f64::NAN, f64::from_bits(f64::NAN.to_bits() + 1), false)]
#[case::signed_zeros(0.0, -0.0, false)]
fn nullable_float_bit_patterns(
    #[case] first: f64,
    #[case] second: f64,
    #[case] constant: bool,
    #[values(false, true)] extension: bool,
    #[values(false, true)] dictionary: bool,
) -> VortexResult<()> {
    let storage = PrimitiveArray::new(
        (0..100)
            .map(|i| if i == 5 { second } else { first })
            .collect::<Buffer<_>>(),
        nulls_every_tenth(),
    )
    .into_array();
    let array = if extension {
        let dtype = ExtDType::<FloatExtension>::try_new(0, storage.dtype().clone())?;
        ExtensionArray::new(dtype.erased(), storage).into_array()
    } else {
        storage
    };
    let compressor = if dictionary {
        CascadingCompressor::new(vec![&FloatDictScheme])
    } else {
        empty_compressor()
    };
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array, &mut ctx)?;
    assert_arrays_eq!(compressed, array, &mut ctx);
    assert_eq!(compressed.is::<Masked>(), constant);
    if constant {
        assert!(compressed.children()[0].is::<Constant>());
    }
    Ok(())
}

#[rstest]
#[case::empty("", "", true)]
#[case::inline("abc", "abc", true)]
#[case::twelve_bytes("abcdefghijkl", "abcdefghijkl", true)]
#[case::thirteen_bytes("abcdefghijklm", "abcdefghijklm", true)]
#[case::different_inline_suffix("abcdefghijkA", "abcdefghijkB", false)]
#[case::different_suffix("abcdefghijklA", "abcdefghijklB", false)]
#[case::different_prefix("Abcdefghijklm", "Bbcdefghijklm", false)]
#[case::different_lengths("abcdefghijkl", "abcdefghijklm", false)]
#[case::embedded_zero("abc\0defghijkl", "abc\0defghijkl", true)]
fn string_and_binary_constants(
    #[case] first: &str,
    #[case] second: &str,
    #[case] constant: bool,
    #[values(false, true)] binary: bool,
    #[values(false, true)] nullable: bool,
) -> VortexResult<()> {
    let nullability = Nullability::from(nullable);
    let dtype = if binary {
        DType::Binary(nullability)
    } else {
        DType::Utf8(nullability)
    };
    let array = VarBinViewArray::from_iter(
        (0..100).map(|i| {
            if nullable && i % 10 == 0 {
                None
            } else {
                Some(if i == 99 { second } else { first })
            }
        }),
        dtype,
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(is_constant_for_compression(&array, &mut ctx)?, constant);
    let compressed =
        CascadingCompressor::new(vec![&StringDictScheme]).compress(&array, &mut ctx)?;
    assert_arrays_eq!(compressed, array, &mut ctx);
    if constant {
        if nullable {
            assert!(compressed.is::<Masked>());
            assert!(compressed.children()[0].is::<Constant>());
        } else {
            assert!(compressed.is::<Constant>());
        }
    }
    Ok(())
}

#[rstest]
#[case::shared_reference(false, false)]
#[case::different_offsets(true, false)]
#[case::different_buffers(false, true)]
fn binary_references(
    #[case] different_offsets: bool,
    #[case] different_buffers: bool,
    #[values(false, true)] nullable: bool,
) -> VortexResult<()> {
    let value = b"\xff\0abcdefghijkl";
    let data = [value.as_slice(), value.as_slice()].concat();
    let views = (0..100)
        .map(|i| {
            BinaryView::make_view(
                value,
                u32::from(different_buffers && i % 2 == 0),
                if different_offsets && i % 2 == 0 {
                    14
                } else {
                    0
                },
            )
        })
        .collect::<Buffer<_>>();
    let validity = if nullable {
        nulls_every_tenth()
    } else {
        Validity::NonNullable
    };
    let mut ctx = SESSION.create_execution_ctx();
    let array = VarBinViewArray::try_new(
        views,
        Arc::from([ByteBuffer::from(data.clone()), ByteBuffer::from(data)]),
        DType::Binary(Nullability::from(nullable)),
        validity,
        &mut ctx,
    )?
    .into_array()
    .slice(1..99)?;
    assert!(is_constant_for_compression(&array, &mut ctx)?);
    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[rstest]
#[case::integer(PrimitiveArray::new(
    (0..100).map(|i| if i % 10 == 0 { 99i32 } else { 7 }).collect::<Buffer<_>>(),
    nulls_every_tenth(),
).into_array())]
#[case::boolean(BoolArray::new(
    (0..100).map(|i| i % 10 != 0).collect::<BitBuffer>(),
    nulls_every_tenth(),
).into_array())]
#[case::decimal(DecimalArray::new(
    (0..100).map(|i| if i % 10 == 0 { 99i64 } else { 7 }).collect::<Buffer<_>>(),
    DecimalDType::new(10, 2),
    nulls_every_tenth(),
).into_array())]
fn values_under_nulls_are_ignored(
    #[case] array: ArrayRef,
    #[values(false, true)] sliced: bool,
) -> VortexResult<()> {
    let array = if sliced { array.slice(1..99)? } else { array };
    let mut ctx = SESSION.create_execution_ctx();
    assert!(is_constant_for_compression(&array, &mut ctx)?);
    let compressed = empty_compressor().compress(&array, &mut ctx)?;
    assert!(compressed.is::<Masked>());
    assert!(compressed.children()[0].is::<Constant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}
