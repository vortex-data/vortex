// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated JSON values for Arrow transport checks.
//!
//! Valid JSON numbers and objects exercise byte preservation, including insignificant whitespace.

mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::builders::builder_with_capacity_in;
    use vortex_array::dtype::DType;
    use vortex_array::memory::BufferAllocatorRef;
    use vortex_array::scalar::Scalar;
    use vortex_arrow::test_harness::DefaultExport;
    use vortex_arrow::test_harness::check_generated;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Json;

    fn check_json_round_trip(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullable = tc.draw(gs::booleans());
        let mut builder = builder_with_capacity_in(
            &DType::Utf8(nullable.into()),
            len,
            BufferAllocatorRef::static_ref(),
        );

        for _ in 0..len {
            if nullable && tc.draw(gs::booleans()) {
                builder.append_nulls(1);
            } else {
                let value = tc.draw(gs::integers::<i64>());
                let text = if tc.draw(gs::booleans()) {
                    format!(" {{ \"n\": {value} }} ")
                } else {
                    value.to_string()
                };
                builder.append_scalar(&Scalar::utf8(text, nullable.into()))?;
            }
        }

        let array = ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, builder.finish())?
            .into_array();

        check_generated(tc, array, DefaultExport::Supported, &mut ctx)
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn arrow_transport(tc: TestCase) {
        check_json_round_trip(&tc)
            .vortex_expect("Arrow transport must preserve dtype and stored values");
    }
}
