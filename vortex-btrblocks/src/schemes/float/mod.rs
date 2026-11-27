// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression schemes.

mod alp;
mod alprd;
mod rle;
mod sparse;

#[cfg(feature = "pco")]
mod pco;

pub use alp::ALPScheme;
pub use alprd::ALPRDScheme;
#[cfg(feature = "pco")]
pub use pco::PcoScheme;
pub use rle::FloatRLEScheme;
pub use sparse::NullDominatedSparseScheme;
// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::FloatDictScheme;
pub use vortex_compressor::stats::FloatStats;

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_alp::ALP;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_compressor::CascadingCompressor;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::BitPackedArrayExt;
    use vortex_fastlanes::BitWidthsView;
    use vortex_session::VortexSession;

    use crate::schemes::float::ALPRDScheme;
    use crate::schemes::float::ALPScheme;
    use crate::schemes::float::FloatDictScheme;
    use crate::schemes::float::FloatRLEScheme;
    use crate::schemes::float::NullDominatedSparseScheme;
    use crate::schemes::integer::BitPackingScheme;
    use crate::schemes::integer::FOR_V1;
    use crate::schemes::integer::IntDictScheme;
    use crate::schemes::integer::IntRLEScheme;
    use crate::schemes::integer::RunEndScheme;
    use crate::schemes::integer::SparseScheme;
    use crate::schemes::integer::ZigZagScheme;

    static BITPACKING: BitPackingScheme = BitPackingScheme::v1();

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_alp::initialize(&session);
        vortex_fastlanes::initialize(&session);
        vortex_sparse::initialize(&session);
        vortex_zigzag::initialize(&session);
        session
    });

    /// On a skewed float column with 2 hot values and several rare outliers, frequency-ordered
    /// dictionary codes let BitPacking pick a 1-bit width with patches for the rare codes.
    #[test]
    fn test_skewed_float_dict_compresses_to_1bit_with_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let rare = [10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let values: Vec<f64> = (0..4096)
            .map(|i| {
                if i % 512 == 0 {
                    rare[(i / 512) % rare.len()]
                } else if i % 3 == 0 {
                    0.0
                } else {
                    2.5
                }
            })
            .collect();
        let array = PrimitiveArray::from_iter(values).into_array();

        let compressor = CascadingCompressor::new(vec![&FloatDictScheme, &BITPACKING]);
        let compressed = compressor.compress(&array, &mut ctx)?;
        let dict = compressed
            .as_opt::<Dict>()
            .expect("expected FloatDictScheme");
        let packed = dict
            .codes()
            .as_opt::<BitPacked>()
            .expect("expected BitPacked codes");
        assert!(matches!(packed.bit_widths(), BitWidthsView::Global(1)));
        assert!(packed.patches().is_some());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    fn float_compressor() -> CascadingCompressor {
        CascadingCompressor::new(vec![
            &ALPScheme,
            &ALPRDScheme,
            &FloatDictScheme,
            &NullDominatedSparseScheme,
            &FloatRLEScheme,
            &FOR_V1,
            &ZigZagScheme,
            &BITPACKING,
            &SparseScheme,
            &IntDictScheme,
            &RunEndScheme,
            &IntRLEScheme,
        ])
    }

    /// On medium-cardinality skewed decimal floats (Bimbo.Venta_hoy-style: 100K rows drawing
    /// Zipfian prices from 5,000 distinct 0.01-stepped f64s), FloatDict must beat ALP and produce
    /// a strictly smaller compressed array than ALP without FloatDict.
    #[test]
    fn skewed_medium_cardinality_decimal_floats_select_dict_over_alp() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 100K rows, 5,000 distinct 0.01-stepped prices across $0.00..$50,000.00:
        // 90% hit the top 16 prices, 8% hit the next 48 prices, 2% hit the remaining 4,936.
        let distinct_prices: Vec<f64> = (0..5_000usize)
            .map(|k| ((k * 1_000 + 37) % 5_000_000) as f64 / 100.0)
            .collect();
        let values: Vec<f64> = (0..100_000usize)
            .map(|i| {
                let rank = match i % 100 {
                    0..90 => i % 16,
                    90..98 => 16 + (i % 48),
                    _ => 64 + ((i / 100) % (5_000 - 64)),
                };
                distinct_prices[rank]
            })
            .collect();
        let array = PrimitiveArray::from_iter(values).into_array();

        let compressed = float_compressor().compress(&array, &mut ctx)?;
        let alp_without_dict = CascadingCompressor::new(vec![
            &ALPScheme,
            &ALPRDScheme,
            &FOR_V1,
            &ZigZagScheme,
            &BITPACKING,
            &SparseScheme,
        ])
        .compress(&array, &mut ctx)?;

        assert!(
            compressed.is::<Dict>(),
            "expected FloatDict on skewed medium-cardinality floats, got:\n{}",
            compressed.display_tree()
        );
        assert!(
            compressed.nbytes() < alp_without_dict.nbytes(),
            "expected FloatDict ({} B) < ALP ({} B)",
            compressed.nbytes(),
            alp_without_dict.nbytes()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    /// On high-cardinality uniform ALP-friendly floats (100K rows with 45,000 distinct 0.01-stepped
    /// prices), ALP without Dict is strictly smaller and must still beat FloatDict.
    #[test]
    fn high_cardinality_uniform_alp_floats_still_select_alp_over_dict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<f64> = (0..100_000usize)
            .map(|i| (i % 45_000) as f64 / 100.0)
            .collect();
        let array = PrimitiveArray::from_iter(values).into_array();

        let compressed = float_compressor().compress(&array, &mut ctx)?;
        let dict_only = CascadingCompressor::new(vec![&FloatDictScheme, &FOR_V1, &BITPACKING])
            .compress(&array, &mut ctx)?;

        assert!(
            compressed.is::<ALP>(),
            "expected ALP on high-cardinality uniform decimal floats, got:\n{}",
            compressed.display_tree()
        );
        assert!(dict_only.is::<Dict>(), "{}", dict_only.display_tree());
        assert!(
            compressed.nbytes() < dict_only.nbytes(),
            "expected ALP ({} B) < FloatDict ({} B)",
            compressed.nbytes(),
            dict_only.nbytes()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }
}
