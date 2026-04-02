// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveData;
use vortex_array::dtype::NativePType;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_integer_ptype;
use vortex_array::stats::ArrayStats;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::FoRArray;
use crate::FoRData;
impl FoRData {
    pub fn encode(array: PrimitiveArray) -> VortexResult<FoRArray> {
        let array_ref = array.clone().into_array();
        let stats = ArrayStats::from(array_ref.statistics().to_owned());
        let min = array_ref
            .statistics()
            .compute_stat(Stat::Min)?
            .ok_or_else(|| vortex_err!("Min stat not found"))?;

        let encoded = match_each_integer_ptype!(array.ptype(), |T| {
            compress_primitive::<T>(array, T::try_from(&min)?)?.into_array()
        });
        let for_data = FoRData::try_new(encoded, min)?;
        let for_array = FoRArray::try_from_data(for_data)?;
        let for_ref = for_array.clone().into_array();
        for_array
            .stats_set()
            .to_ref(&for_ref)
            .inherit_from(stats.to_ref(&for_ref));
        Ok(for_array)
    }
}

fn compress_primitive<T: NativePType + WrappingSub + PrimInt>(
    parray: PrimitiveArray,
    min: T,
) -> VortexResult<PrimitiveData> {
    // Set null values to the min value, ensuring that decompress into a value in the primitive
    // range (and stop them wrapping around).
    parray
        .into_data()
        .map_each_with_validity::<T, _, _>(|(v, bool)| {
            if bool {
                v.wrapping_sub(&min)
            } else {
                T::zero()
            }
        })
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use itertools::Itertools;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::expr::stats::StatsProvider;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedData;
    use crate::FoRArray;
    use crate::r#for::array::for_decompress::decompress;
    use crate::r#for::array::for_decompress::fused_decompress;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_compress_round_trip_small() {
        let array = PrimitiveArray::new(
            (1i32..10).collect::<vortex_buffer::Buffer<_>>(),
            Validity::NonNullable,
        );
        let compressed = FoRData::encode(array.clone()).unwrap();
        assert_eq!(i32::try_from(compressed.reference_scalar()).unwrap(), 1);

        assert_arrays_eq!(compressed, array);
    }

    #[test]
    fn test_compress() {
        // Create a range offset by a million.
        let array = PrimitiveArray::new(
            (0u32..10_000)
                .map(|v| v + 1_000_000)
                .collect::<vortex_buffer::Buffer<_>>(),
            Validity::NonNullable,
        );
        let compressed = FoRData::encode(array).unwrap();
        assert_eq!(
            u32::try_from(compressed.reference_scalar()).unwrap(),
            1_000_000u32
        );
    }

    #[test]
    fn test_zeros() {
        let array = PrimitiveArray::new(buffer![0i32; 100], Validity::NonNullable);
        assert_eq!(array.statistics().len(), 0);

        let dtype = array.dtype().clone();
        let compressed = FoRData::encode(array).unwrap();
        assert_eq!(compressed.reference_scalar().dtype(), &dtype);
        assert!(compressed.reference_scalar().dtype().is_signed_int());
        assert!(compressed.encoded().dtype().is_signed_int());

        let encoded = compressed.encoded().scalar_at(0).unwrap();
        assert_eq!(encoded, Scalar::from(0i32));
    }

    #[test]
    fn test_decompress() {
        // Create a range offset by a million.
        let array = PrimitiveArray::from_iter((0u32..100_000).step_by(1024).map(|v| v + 1_000_000));
        let compressed = FoRData::encode(array.clone()).unwrap();
        assert_arrays_eq!(compressed, array);
    }

    #[test]
    fn test_decompress_fused() {
        // Create a range offset by a million.
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedData::encode(&array.into_array(), 3).unwrap();
        let compressed = FoRData::try_new(bp.into_array(), 10u32.into()).unwrap();
        assert_arrays_eq!(compressed, expect);
    }

    #[test]
    fn test_decompress_fused_patches() -> VortexResult<()> {
        // Create a range offset by a million.
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedData::encode(&array.into_array(), 2).unwrap();
        let compressed = FoRArray::try_from_data(
            FoRData::try_new(bp.clone().into_array(), 10u32.into()).unwrap(),
        )?;
        let decompressed = fused_decompress::<u32>(
            &compressed,
            bp.as_view(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert_arrays_eq!(decompressed, expect);
        Ok(())
    }

    #[test]
    fn test_overflow() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(i8::MIN..=i8::MAX);
        let compressed = FoRData::encode(array.clone()).unwrap();
        assert_eq!(
            i8::MIN,
            compressed
                .reference_scalar()
                .as_primitive()
                .typed_value::<i8>()
                .unwrap()
        );

        let encoded = compressed
            .encoded()
            .to_primitive()
            .reinterpret_cast(PType::U8);
        let unsigned: Vec<u8> = (0..=u8::MAX).collect_vec();
        let expected_unsigned = PrimitiveArray::from_iter(unsigned);
        assert_arrays_eq!(encoded, expected_unsigned);

        let decompressed = decompress(&compressed, &mut SESSION.create_execution_ctx())?;
        array
            .as_slice::<i8>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                assert_eq!(*v, i8::try_from(&compressed.scalar_at(i).unwrap()).unwrap());
            });
        assert_arrays_eq!(decompressed, array);
        Ok(())
    }
}
