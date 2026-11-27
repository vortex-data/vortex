// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression schemes.

mod bitpacking;
mod delta;
mod elias_fano;
mod for_;
mod rle;
mod runend;
mod sequence;
mod sparse;
mod zigzag;

#[cfg(feature = "pco")]
mod pco;

pub(crate) use bitpacking::BITPACKING_V1;
pub use bitpacking::BitPackingScheme;
pub use delta::DeltaScheme;
pub use elias_fano::EliasFanoScheme;
pub(crate) use for_::FOR_V1;
pub use for_::FoRScheme;
#[cfg(feature = "pco")]
pub use pco::PcoScheme;
pub use rle::IntRLEScheme;
pub(crate) use rle::rle_encode;
pub use runend::RunEndScheme;
pub use sequence::SequenceScheme;
pub use sparse::SparseScheme;
// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::IntDictScheme;
pub use vortex_compressor::stats::IntegerStats;
pub use zigzag::ZigZagScheme;

/// Threshold for the average run length in an array before we consider run-length encoding.
pub(crate) const RUN_LENGTH_THRESHOLD: u32 = 4;

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_compressor::CascadingCompressor;
    use vortex_compressor::scheme::Scheme;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::BitPackedArrayExt;
    use vortex_fastlanes::BitWidthsView;
    use vortex_session::VortexSession;

    use crate::schemes::binary::BinaryDictScheme;
    use crate::schemes::decimal::DecimalDictScheme;
    use crate::schemes::integer::BitPackingScheme;
    use crate::schemes::integer::IntDictScheme;
    use crate::schemes::string::StringDictScheme;

    static BITPACKING: BitPackingScheme = BitPackingScheme::v1();

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_fastlanes::initialize(&session);
        session
    });

    fn assert_1bit_patched_dict(scheme: &'static dyn Scheme, array: ArrayRef) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = CascadingCompressor::new(vec![scheme, &BITPACKING]);
        let compressed = compressor.compress(&array, &mut ctx)?;
        let dict = compressed.as_opt::<Dict>().expect("expected Dict");
        let packed = dict
            .codes()
            .as_opt::<BitPacked>()
            .expect("expected BitPacked codes");
        assert!(matches!(packed.bit_widths(), BitWidthsView::Global(1)));
        assert!(packed.patches().is_some());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn skewed_int_dict_compresses_to_1bit_with_patches() -> VortexResult<()> {
        let rare = [100i64, 200, 300, 400, 500, 600, 700, 800];
        let array = PrimitiveArray::from_iter((0..4096).map(|i| {
            if i % 512 == 0 {
                rare[(i / 512) % rare.len()]
            } else if i % 3 == 0 {
                10
            } else {
                25
            }
        }))
        .into_array();
        assert_1bit_patched_dict(&IntDictScheme, array)
    }

    #[test]
    fn skewed_string_and_binary_dicts_compress_to_1bit_with_patches() -> VortexResult<()> {
        let rare = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"];
        let strings: Vec<&str> = (0..4096)
            .map(|i| {
                if i % 512 == 0 {
                    rare[(i / 512) % rare.len()]
                } else if i % 3 == 0 {
                    "hot_a"
                } else {
                    "hot_b"
                }
            })
            .collect();
        let utf8 = VarBinViewArray::from_iter_str(strings.iter().copied()).into_array();
        assert_1bit_patched_dict(&StringDictScheme, utf8)?;

        let binary = VarBinViewArray::from_iter(
            strings.iter().map(|s| Some(s.as_bytes())),
            DType::Binary(Nullability::NonNullable),
        )
        .into_array();
        assert_1bit_patched_dict(&BinaryDictScheme, binary)
    }

    #[test]
    fn skewed_decimal_dict_compresses_to_1bit_with_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let rare = [1000i128, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let values: Buffer<i128> = (0..4096)
            .map(|i| {
                if i % 512 == 0 {
                    rare[(i / 512) % rare.len()]
                } else if i % 3 == 0 {
                    0
                } else {
                    250
                }
            })
            .collect();
        let array = DecimalArray::new(
            values,
            DecimalDType::new(38, 2),
            Validity::NonNullable,
            &mut ctx,
        )
        .into_array();
        assert_1bit_patched_dict(&DecimalDictScheme, array)
    }
}
