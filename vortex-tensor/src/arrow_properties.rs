// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated vector and tensor metadata for Arrow transport checks.
//!
//! Storage widths follow the generated shapes so transport failures can shrink to valid inputs.

mod tests {
    use std::sync::Arc;

    use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
    use arrow_array::Float32Array;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use hegel::TestCase;
    use hegel::generators as gs;
    use prost::Message;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_arrow::test_harness::check_arrow;
    use vortex_arrow::test_harness::check_containers;
    use vortex_arrow::test_harness::check_generated;
    use vortex_arrow::test_harness::draw_storage_case;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::fixed_shape_tensor::FixedShapeTensor;
    use crate::fixed_shape_tensor::FixedShapeTensorMetadata;
    use crate::vector::Vector;
    use crate::vector::VectorMetadata;

    /// Test-owned wire mapping, independent of the dtype serializer and Arrow exporter.
    #[derive(prost::Message)]
    struct TensorWire {
        #[prost(uint32, repeated, tag = "1")]
        shape: Vec<u32>,
        #[prost(string, repeated, tag = "2")]
        names: Vec<String>,
        #[prost(uint32, repeated, tag = "3")]
        permutation: Vec<u32>,
    }

    #[track_caller]
    fn check_vector_transport(tc: &TestCase) -> VortexResult<()> {
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
        let storage = draw_storage_case(tc, &storage_dtype, len, &mut ctx)?;
        let vector = ExtensionArray::try_new_from_vtable(
            Vector,
            VectorMetadata::try_new(ptype, size)?,
            storage.array,
        )?
        .into_array();

        let mut field = Field::new("value", storage.arrow.data_type().clone(), nullable);
        field
            .metadata_mut()
            .insert("ARROW:extension:name", "vortex.tensor.vector");
        check_generated(tc, vector, &field, storage.arrow, &mut ctx)
    }

    #[track_caller]
    fn check_tensor_transport(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullable = tc.draw(gs::booleans());
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
        let storage = draw_storage_case(tc, &storage_dtype, len, &mut ctx)?;

        let rank = shape.len();
        let mut metadata = FixedShapeTensorMetadata::new(ptype, shape.clone());
        let names = tc
            .draw(gs::booleans())
            .then(|| (0..rank).map(|i| format!("axis{i}")).collect::<Vec<_>>());
        if let Some(names) = &names {
            metadata = metadata.with_dim_names(names.clone())?;
        }
        let permutation = if tc.draw(gs::booleans()) {
            let mut permutation: Vec<_> = (0..rank).collect();
            if rank > 0 {
                permutation.rotate_left(tc.draw(gs::integers::<usize>().max_value(rank - 1)));
            }
            if tc.draw(gs::booleans()) {
                permutation.reverse();
            }
            metadata = metadata.with_permutation(permutation.clone())?;
            Some(permutation)
        } else {
            None
        };

        let wire = TensorWire {
            shape: shape
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?,
            names: names.clone().unwrap_or_default(),
            permutation: permutation
                .iter()
                .flatten()
                .copied()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?,
        }
        .encode_to_vec();

        // Arrow metadata lists dimensions in physical order: physical[permutation[i]] = logical[i].
        let physical = |logical: &[String]| {
            let mut out = logical.to_vec();
            for (i, &p) in permutation.iter().flatten().enumerate() {
                out[p] = logical[i].clone();
            }
            out
        };
        let shape_json: Vec<String> = shape.iter().map(ToString::to_string).collect();
        let mut arrow_metadata = format!(r#"{{"shape":[{}]"#, physical(&shape_json).join(","));
        // Vortex metadata treats empty name and permutation lists as unset.
        if let Some(names) = names.as_ref().filter(|names| !names.is_empty()) {
            let quoted: Vec<String> = names.iter().map(|n| format!(r#""{n}""#)).collect();
            arrow_metadata.push_str(&format!(
                r#","dim_names":[{}]"#,
                physical(&quoted).join(",")
            ));
        }
        if let Some(permutation) = permutation.as_ref().filter(|p| !p.is_empty()) {
            let items: Vec<String> = permutation.iter().map(ToString::to_string).collect();
            arrow_metadata.push_str(&format!(r#","permutation":[{}]"#, items.join(",")));
        }
        arrow_metadata.push('}');

        let mut field = Field::new("value", storage.arrow.data_type().clone(), nullable);
        field.metadata_mut().extend([
            (
                "ARROW:extension:name".to_owned(),
                "arrow.fixed_shape_tensor".to_owned(),
            ),
            ("ARROW:extension:metadata".to_owned(), arrow_metadata),
        ]);
        let expected = storage.arrow;
        let tensor =
            ExtensionArray::try_new_from_vtable(FixedShapeTensor, metadata, storage.array)?
                .into_array();

        assert_eq!(tensor.dtype().as_extension().serialize_metadata()?, wire);
        check_generated(tc, tensor, &field, expected, &mut ctx)
    }

    #[track_caller]
    fn check_zero_width_vectors(rows: usize) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let valid: Vec<_> = (0..rows).map(|i| i % 2 == 0).collect();
        let storage = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(Vec::<f32>::new()).into_array(),
            0,
            Validity::from_iter(valid.iter().copied()),
            rows,
        )?
        .into_array();
        let array = ExtensionArray::try_new_from_vtable(
            Vector,
            VectorMetadata::try_new(PType::F32, 0)?,
            storage,
        )?
        .into_array();
        let child = Arc::new(Field::new("item", DataType::Float32, false));
        let expected = Arc::new(ArrowFixedSizeListArray::try_new_with_length(
            Arc::clone(&child),
            0,
            Arc::new(Float32Array::from(Vec::<f32>::new())),
            Some(NullBuffer::from(valid)),
            rows,
        )?);
        let mut field = Field::new("value", DataType::FixedSizeList(child, 0), true);
        field
            .metadata_mut()
            .insert("ARROW:extension:name", "vortex.tensor.vector");
        check_arrow(array.clone(), &field, expected, &mut ctx)?;
        check_containers(array, &mut ctx)
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    #[hegel::explicit_test_case(rows = 3usize)]
    fn zero_width_rows(tc: TestCase) {
        let rows = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
        check_zero_width_vectors(rows)
            .vortex_expect("zero-width lists must preserve nonempty row counts and nulls");
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn vector_transport(tc: TestCase) {
        check_vector_transport(&tc)
            .vortex_expect("Arrow transport must preserve dtype and stored values");
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn tensor_transport(tc: TestCase) {
        check_tensor_transport(&tc)
            .vortex_expect("tensor Arrow transport must preserve metadata and stored values");
    }
}
