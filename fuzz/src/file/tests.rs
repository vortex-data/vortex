// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Unstructured;
use itertools::Itertools;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::col;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::extension::datetime::random_temporal_ext_dtype;
use vortex_array::scalar::arbitrary::random_scalar;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;

use crate::RUNTIME;
use crate::SESSION;
use crate::array::assert_array_eq;

/// Deterministic pseudo-random bytes for driving [`Unstructured`].
fn pseudo_random_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        })
        .collect()
}

/// Temporal columns must round-trip through file write/scan, including a filter pushed down
/// on the temporal column.
#[test]
fn temporal_columns_roundtrip_through_file_io() -> VortexResult<()> {
    let bytes = pseudo_random_bytes(256 * 1024, 3);
    let mut u = Unstructured::new(&bytes);
    let mut ctx = SESSION.create_execution_ctx();

    for _ in 0..4 {
        let Ok(ext_dtype) = random_temporal_ext_dtype(&mut u, Nullability::NonNullable) else {
            break;
        };
        let field_dtype = DType::Extension(ext_dtype);
        let dtype = DType::Struct(
            StructFields::new(FieldNames::from(["ts"]), vec![field_dtype.clone()]),
            Nullability::NonNullable,
        );
        let Ok(array) = ArbitraryArray::arbitrary_with_config(
            &mut u,
            &ArbitraryArrayConfig {
                dtype: Some(dtype.clone()),
                len: 1..=256,
            },
        ) else {
            break;
        };
        let array = array.0;

        let Ok(needle) = random_scalar(&mut u, &field_dtype) else {
            break;
        };
        let filter_expr = Binary.new_expr(Operator::Gte, [col("ts"), lit(needle)]);

        // Expected result: filter the in-memory array.
        let bool_mask = array
            .clone()
            .apply(&filter_expr)?
            .execute::<BoolArray>(&mut ctx)?;
        let mask = bool_mask.to_mask_fill_null_false(&mut ctx);
        let expected = array.clone().filter(mask)?;

        // Write the array to a file and scan it back with the filter pushed down.
        let mut buff = ByteBufferMut::empty();
        SESSION
            .write_options()
            .blocking(&*RUNTIME)
            .write(&mut buff, array.to_array_iterator())?;

        let output = SESSION
            .open_options()
            .open_buffer(buff)?
            .scan()?
            .with_projection(root())
            .with_filter(filter_expr)
            .into_array_iter(&*RUNTIME)?
            .try_collect::<_, Vec<_>, _>()?;

        let actual = match output.len() {
            0 => Canonical::empty(expected.dtype()).into_array(),
            1 => output.into_iter().next().vortex_expect("one chunk"),
            _ => ChunkedArray::from_iter(output).into_array(),
        };

        assert_array_eq(&expected, &actual, 0).map_err(|e| vortex_err!("file roundtrip: {e}"))?;
    }
    Ok(())
}
