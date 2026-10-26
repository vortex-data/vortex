// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public selection oracle for a compressed logical validity child.

#![expect(clippy::tests_outside_test_module)]

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_file::register_default_encodings;
use vortex_pco::Pco;
use vortex_runend::RunEnd;
use vortex_session::VortexSession;

#[test]
fn compressed_validity_selection_oracle() -> VortexResult<()> {
    let session = VortexSession::empty();
    register_default_encodings(&session);
    let mut ctx = session.create_execution_ctx();
    let validity = Validity::Array(
        RunEnd::try_new(
            PrimitiveArray::from_iter([32768u64, 65536, 98304, 131072]).into_array(),
            BoolArray::from_iter([false, true, false, true]).into_array(),
            &mut ctx,
        )?
        .into_array(),
    );
    let input = PrimitiveArray::new((0..131072i64).collect::<Vec<_>>(), validity);
    let expected =
        PrimitiveArray::from_option_iter((0..131072i64).map(|i| (i / 32768 % 2 == 1).then_some(i)));
    let encoded = Pco::from_primitive(input.as_view(), 3, 127, &mut ctx)?.into_array();
    let indices =
        PrimitiveArray::from_iter([0u32, 32767, 32768, 65535, 65536, 98304, 131071, 32768])
            .into_array();
    assert_arrays_eq!(
        encoded.take(indices.clone())?,
        expected.take(indices)?,
        &mut ctx
    );
    assert_arrays_eq!(
        encoded.slice(30001..130001)?,
        expected.slice(30001..130001)?,
        &mut ctx
    );
    let mut probe = encoded.repeated_probe();
    for index in [0, 32767, 32768, 65535, 65536, 98304, 131071, 32768] {
        assert_eq!(
            probe.execute_scalar(index, &mut ctx)?,
            expected.execute_scalar(index, &mut ctx)?
        );
    }
    Ok(())
}
