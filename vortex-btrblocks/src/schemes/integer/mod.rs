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
pub(crate) const RUN_LENGTH_THRESHOLD: u32 = vortex_compressor::builtins::RLE_MIN_RUN_LENGTH;

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
    use crate::schemes::integer::FOR_V1;
    use crate::schemes::integer::IntDictScheme;
    use crate::schemes::integer::IntRLEScheme;
    use crate::schemes::integer::RunEndScheme;
    use crate::schemes::integer::SparseScheme;
    use crate::schemes::integer::ZigZagScheme;
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

    fn int_compressor() -> CascadingCompressor {
        CascadingCompressor::new(vec![
            &FOR_V1,
            &ZigZagScheme,
            &BITPACKING,
            &SparseScheme,
            &IntDictScheme,
            &RunEndScheme,
            &IntRLEScheme,
        ])
    }

    #[test]
    fn two_value_dict_beats_two_bit_bitpacking() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Values {0, 3} need 2 bits in BitPacking (4,096 B) but 1 bit in IntDict (~2,064 B).
        let array =
            PrimitiveArray::from_iter((0..16_384i64).map(|i| if i % 2 == 0 { 0 } else { 3 }))
                .into_array();

        let compressed = int_compressor().compress(&array, &mut ctx)?;
        let bp_only = CascadingCompressor::new(vec![&BITPACKING]).compress(&array, &mut ctx)?;

        assert!(compressed.is::<Dict>(), "{}", compressed.display_tree());
        assert!(
            compressed.nbytes() < bp_only.nbytes(),
            "expected Dict ({} B) < BitPacking ({} B)",
            compressed.nbytes(),
            bp_only.nbytes()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn skewed_medium_cardinality_ints_select_patched_dict_over_bitpacking() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 100K rows, 2,000 distinct values in 0..2000 where 98% hit large IDs 1984..2000.
        // Raw BitPacking needs 11 bits (~137.5 KB); frequency-sorted IntDict packs at 4 bits +
        // 2% patches (~65 KB).
        let array = PrimitiveArray::from_iter((0..100_000i64).map(|i| {
            if i % 50 == 0 {
                (i / 50) % 1984
            } else {
                1984 + (i % 16)
            }
        }))
        .into_array();

        let compressed = int_compressor().compress(&array, &mut ctx)?;
        let bp_only = CascadingCompressor::new(vec![&BITPACKING]).compress(&array, &mut ctx)?;

        let dict = compressed
            .as_opt::<Dict>()
            .unwrap_or_else(|| panic!("expected Dict, got:\n{}", compressed.display_tree()));
        let packed = dict
            .codes()
            .as_opt::<BitPacked>()
            .expect("expected BitPacked codes");
        assert!(matches!(packed.bit_widths(), BitWidthsView::Global(4)));
        assert!(packed.patches().is_some());
        assert!(
            compressed.nbytes() < bp_only.nbytes(),
            "expected Dict ({} B) < BitPacking ({} B)",
            compressed.nbytes(),
            bp_only.nbytes()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn uniform_dense_ints_do_not_select_dict_over_bitpacking() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 128 uniform values in 0..128: raw BitPacking packs at 7 bits (14,336 B), whereas
        // IntDict's u8 codes cannot be bit-packed (8 - 7 < 2) and would bloat to 17,408 B.
        let array = PrimitiveArray::from_iter((0..16_384i64).map(|i| i % 128)).into_array();

        let compressed = int_compressor().compress(&array, &mut ctx)?;
        assert!(
            !compressed.is::<Dict>(),
            "uniform dense ints should not pick Dict:\n{}",
            compressed.display_tree()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn high_distinct_runs_prefer_runend_over_dict_and_low_distinct_runs_prefer_dict()
    -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 1,024 runs of length 16, each run a distinct wide value: direct RunEnd beats Dict(RunEnd).
        let unique_runs =
            PrimitiveArray::from_iter((0..16_384i64).map(|i| (i / 16) * 1_000_000)).into_array();
        let compressed_unique = int_compressor().compress(&unique_runs, &mut ctx)?;
        assert!(
            !compressed_unique.is::<Dict>(),
            "runs with distinct values per run should not pick Dict:\n{}",
            compressed_unique.display_tree()
        );
        assert_arrays_eq!(compressed_unique, unique_runs, &mut ctx);

        // 1,024 runs of length 16 cycling over only 4 wide values: Dict(RunEnd) beats direct RunEnd.
        let low_card_runs =
            PrimitiveArray::from_iter((0..16_384i64).map(|i| ((i / 16) % 4) * 1_000_000_000))
                .into_array();
        let compressed_low = int_compressor().compress(&low_card_runs, &mut ctx)?;
        let runend_only = CascadingCompressor::new(vec![&RunEndScheme, &FOR_V1, &BITPACKING])
            .compress(&low_card_runs, &mut ctx)?;
        assert!(
            compressed_low.is::<Dict>(),
            "low-cardinality runs should pick Dict:\n{}",
            compressed_low.display_tree()
        );
        assert!(
            compressed_low.nbytes() < runend_only.nbytes(),
            "expected Dict ({} B) < RunEnd ({} B)",
            compressed_low.nbytes(),
            runend_only.nbytes()
        );
        assert_arrays_eq!(compressed_low, low_card_runs, &mut ctx);
        Ok(())
    }
}
