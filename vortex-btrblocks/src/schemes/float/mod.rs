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

    use crate::schemes::float::FloatDictScheme;
    use crate::schemes::integer::BitPackingScheme;

    static BITPACKING: BitPackingScheme = BitPackingScheme::v1();

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_fastlanes::initialize(&session);
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
}
