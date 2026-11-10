// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated vector and tensor metadata for Arrow transport checks.
//!
//! Storage widths follow the generated shapes so transport failures can shrink to valid inputs.

mod tests {
    use std::sync::Arc;

    use hegel::TestCase;
    use hegel::generators as gs;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_arrow::test_harness::DefaultExport;
    use vortex_arrow::test_harness::check_generated;
    use vortex_arrow::test_harness::draw_storage;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::fixed_shape_tensor::FixedShapeTensor;
    use crate::fixed_shape_tensor::FixedShapeTensorMetadata;
    use crate::vector::Vector;

    fn check_tensor_round_trips(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullable = tc.draw(gs::booleans());
        let size = tc.draw(gs::integers::<u32>().max_value(4));
        let ptype = tc.draw(gs::sampled_from(&[PType::F16, PType::F32, PType::F64]));
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(ptype, Nullability::NonNullable)),
            size,
            nullable.into(),
        );
        let storage = draw_storage(tc, &storage_dtype, len, &mut ctx)?;
        let vector =
            ExtensionArray::try_new_from_vtable(Vector, EmptyMetadata, storage)?.into_array();

        check_generated(tc, vector, DefaultExport::Supported, &mut ctx)?;

        let shape = tc.draw(gs::vecs(gs::integers::<usize>().max_value(3)).max_size(3));
        let size = u32::try_from(shape.iter().product::<usize>())?;
        let ptype = tc.draw(gs::sampled_from(&[
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
            PType::F16,
            PType::F32,
            PType::F64,
        ]));
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(ptype, Nullability::NonNullable)),
            size,
            nullable.into(),
        );
        let storage = draw_storage(tc, &storage_dtype, len, &mut ctx)?;

        let rank = shape.len();
        let mut metadata = FixedShapeTensorMetadata::new(shape);
        if tc.draw(gs::booleans()) {
            metadata = metadata.with_dim_names((0..rank).map(|i| format!("axis{i}")).collect())?;
        }
        if tc.draw(gs::booleans()) {
            let mut permutation: Vec<_> = (0..rank).collect();
            if rank > 0 {
                permutation.rotate_left(tc.draw(gs::integers::<usize>().max_value(rank - 1)));
            }
            if tc.draw(gs::booleans()) {
                permutation.reverse();
            }
            metadata = metadata.with_permutation(permutation)?;
        }

        let tensor =
            ExtensionArray::try_new_from_vtable(FixedShapeTensor, metadata, storage)?.into_array();

        check_generated(tc, tensor, DefaultExport::Unsupported, &mut ctx)
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn arrow_transport(tc: TestCase) {
        check_tensor_round_trips(&tc)
            .vortex_expect("Arrow transport must preserve dtype and stored values");
    }
}
