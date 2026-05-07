// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use parquet_variant::VariantPath as PqVariantPath;
use parquet_variant::VariantPathElement as PqVariantPathElement;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use parquet_variant_compute::variant_get as arrow_variant_get;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantData;

pub(crate) static PARENT_KERNELS: ParentKernelSet<ParquetVariant> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ParquetVariant)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ParquetVariant)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ParquetVariant)),
    ParentKernelSet::lift(&VariantGetExecute),
]);

#[derive(Default, Debug)]
struct VariantGetExecute;

impl ExecuteParentKernel<ParquetVariant> for VariantGetExecute {
    type Parent = ExactScalarFn<VariantGet>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, ParquetVariant>,
        parent: ScalarFnArrayView<'_, VariantGet>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }

        let arrow_variant = array.to_arrow(ctx)?;
        let arrow_input: ArrowArrayRef = Arc::new(arrow_variant.into_inner());
        let get_options =
            GetOptions::new_with_path(to_parquet_variant_path(parent.options.path())?)
                .with_as_type(to_arrow_as_type(parent.options.dtype())?);

        let arrow_output = arrow_variant_get(&arrow_input, get_options)?;
        let output = if parent
            .options
            .dtype()
            .is_some_and(|dtype| !dtype.is_variant())
        {
            ArrayRef::from_arrow(arrow_output.as_ref(), true)?
        } else {
            let arrow_variant_output = ArrowVariantArray::try_new(arrow_output.as_ref())?;
            ParquetVariantData::from_arrow_variant_with_nullability(
                &arrow_variant_output,
                Nullability::Nullable,
            )?
        };

        vortex_ensure_eq!(
            output.dtype(),
            parent.dtype(),
            "VariantGet output dtype must match parent dtype"
        );
        Ok(Some(output))
    }
}

fn to_parquet_variant_path(path: &VariantPath) -> VortexResult<PqVariantPath<'static>> {
    path.elements()
        .iter()
        .map(|element| match element {
            VariantPathElement::Field(name) => Ok(PqVariantPathElement::field(Cow::Owned(
                name.as_ref().to_string(),
            ))),
            VariantPathElement::Index(index) => {
                let index = usize::try_from(*index)
                    .map_err(|_| vortex_err!("VariantGet path index {index} is too large"))?;
                Ok(PqVariantPathElement::index(index))
            }
        })
        .collect::<VortexResult<Vec<_>>>()
        .map(PqVariantPath::new)
}

fn to_arrow_as_type(dtype: Option<&DType>) -> VortexResult<Option<FieldRef>> {
    match dtype {
        Some(dtype) if !dtype.is_variant() => Ok(Some(Arc::new(Field::new(
            "variant_get",
            dtype.to_arrow_dtype()?,
            true,
        )))),
        Some(_) | None => Ok(None),
    }
}

impl SliceKernel for ParquetVariant {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.slice(range.clone())?;
        let metadata = array.metadata_array().slice(range.clone())?;
        let value = array
            .value_array()
            .map(|v| v.slice(range.clone()))
            .transpose()?;
        let typed_value = array
            .typed_value_array()
            .map(|tv| tv.slice(range))
            .transpose()?;
        Ok(Some(
            ParquetVariant::try_new(validity, metadata, value, typed_value)?.into_array(),
        ))
    }
}

impl FilterKernel for ParquetVariant {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.filter(mask)?;
        let metadata = array.metadata_array().filter(mask.clone())?;
        let value = array
            .value_array()
            .map(|v| v.filter(mask.clone()))
            .transpose()?;
        let typed_value = array
            .typed_value_array()
            .map(|tv| tv.filter(mask.clone()))
            .transpose()?;
        Ok(Some(
            ParquetVariant::try_new(validity, metadata, value, typed_value)?.into_array(),
        ))
    }
}

impl TakeExecute for ParquetVariant {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.take(indices)?;
        let metadata = array.metadata_array().take(indices.clone())?;
        let value = array
            .value_array()
            .map(|v| v.take(indices.clone()))
            .transpose()?;
        let typed_value = array
            .typed_value_array()
            .map(|tv| tv.take(indices.clone()))
            .transpose()?;
        Ok(Some(
            ParquetVariant::try_new(validity, metadata, value, typed_value)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::StringArray;
    use arrow_array::StructArray;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilder;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use parquet_variant_compute::VariantArrayBuilder;
    use parquet_variant_compute::json_to_variant;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray as VortexStructArray;
    use vortex_array::arrays::Variant;
    use vortex_array::arrays::VariantArray;
    use vortex_array::arrays::variant::VariantArrayExt;
    use vortex_array::dtype::DType as VortexDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::expr::variant_get;
    use vortex_array::scalar_fn::fns::variant_get::VariantPath;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use crate::ParquetVariantData;

    fn make_unshredded_array() -> VortexResult<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(4);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from(99i64));
        ParquetVariantData::from_arrow_variant(&builder.build())
    }

    fn make_nullable_array() -> VortexResult<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(4);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from(99i64));
        let inner = builder.build().into_inner();

        let null_struct = StructArray::try_new(
            inner.fields().clone(),
            inner.columns().to_vec(),
            Some(NullBuffer::from(vec![true, false, true, false])),
        )?;
        let arrow_variant = ArrowVariantArray::try_new(&null_struct)?;
        ParquetVariantData::from_arrow_variant(&arrow_variant)
    }

    fn make_unshredded_json_array(values: Vec<Option<&str>>) -> VortexResult<ArrayRef> {
        let json: ArrowArrayRef = Arc::new(StringArray::from(values));
        let arrow_variant = json_to_variant(&json)?;
        let canonical = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
        Ok(canonical.as_::<Variant>().core_storage().clone())
    }

    fn execute_variant_get(
        array: ArrayRef,
        path: &str,
        dtype: Option<VortexDType>,
    ) -> VortexResult<ArrayRef> {
        let expr = variant_get(root(), VariantPath::parse(path)?, dtype);
        array
            .apply(&expr)?
            .execute::<ArrayRef>(&mut LEGACY_SESSION.create_execution_ctx())
    }

    #[test]
    fn test_slice_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let sliced = arr.slice(1..3)?;

        assert_eq!(sliced.len(), 2);
        assert_eq!(
            sliced.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            sliced.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_slice_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        let sliced = arr.slice(0..3)?;

        assert_eq!(sliced.len(), 3);
        assert!(
            !sliced
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            sliced
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            !sliced
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );

        Ok(())
    }

    #[test]
    fn test_filter_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let mask = Mask::from_iter([true, false, true, false]);
        let filtered = arr.filter(mask)?;

        assert_eq!(filtered.len(), 2);
        assert_eq!(
            filtered.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            filtered.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_filter_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        // Keep rows 0 (valid), 1 (null), 3 (null)
        let mask = Mask::from_iter([true, true, false, true]);
        let filtered = arr.filter(mask)?;

        assert_eq!(filtered.len(), 3);
        assert!(
            !filtered
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            filtered
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            filtered
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );

        Ok(())
    }

    #[test]
    fn test_take_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let indices = PrimitiveArray::from_iter([2u64, 0, 3]);
        let taken = arr.take(indices.into_array())?;

        assert_eq!(taken.len(), 3);
        assert_eq!(
            taken.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            taken.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            taken.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_take_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        // Take: valid (0), null (1), null (3), valid (2)
        let indices = PrimitiveArray::from_iter([0u64, 1, 3, 2]);
        let taken = arr.take(indices.into_array())?;

        assert_eq!(taken.len(), 4);
        assert!(
            !taken
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            taken
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            taken
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            !taken
                .execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );

        Ok(())
    }

    #[test]
    fn test_variant_get_unshredded_field_as_i32() -> VortexResult<()> {
        let arr = make_unshredded_json_array(vec![
            Some(r#"{"a": 1}"#),
            None,
            Some(r#"{"a": null}"#),
            Some(r#"{"a": "wrong"}"#),
            Some(r#"{"b": 2}"#),
        ])?;

        let result = execute_variant_get(
            arr,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_eq!(
            result.dtype(),
            &VortexDType::Primitive(PType::I32, Nullability::Nullable)
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(
            result
                .execute_scalar(0, &mut ctx)?
                .as_primitive()
                .typed_value::<i32>(),
            Some(1)
        );
        for idx in 1..result.len() {
            assert!(result.execute_scalar(idx, &mut ctx)?.is_null());
        }

        Ok(())
    }

    #[test]
    fn test_variant_get_unshredded_list_index_as_i32() -> VortexResult<()> {
        let arr = make_unshredded_json_array(vec![
            Some(r#"{"items": [10, 20]}"#),
            Some(r#"{"items": []}"#),
            Some(r#"{"items": ["x", 7]}"#),
        ])?;

        let result = execute_variant_get(
            arr,
            "$.items[1]",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(
            result
                .execute_scalar(0, &mut ctx)?
                .as_primitive()
                .typed_value::<i32>(),
            Some(20)
        );
        assert!(result.execute_scalar(1, &mut ctx)?.is_null());
        assert_eq!(
            result
                .execute_scalar(2, &mut ctx)?
                .as_primitive()
                .typed_value::<i32>(),
            Some(7)
        );

        Ok(())
    }

    #[test]
    fn test_variant_get_unshredded_field_as_variant() -> VortexResult<()> {
        let arr = make_unshredded_json_array(vec![
            Some(r#"{"a": "ok"}"#),
            None,
            Some(r#"{"a": null}"#),
            Some(r#"{"b": 2}"#),
        ])?;

        let result = execute_variant_get(arr, "$.a", None)?;

        assert_eq!(result.dtype(), &VortexDType::Variant(Nullability::Nullable));
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let row0 = result.execute_scalar(0, &mut ctx)?;
        assert_eq!(
            row0.as_variant()
                .value()
                .and_then(|value| value.as_utf8().value())
                .map(|value| value.as_str()),
            Some("ok")
        );
        assert!(result.execute_scalar(1, &mut ctx)?.as_variant().is_null());
        assert_eq!(
            result
                .execute_scalar(2, &mut ctx)?
                .as_variant()
                .is_variant_null(),
            Some(true)
        );
        assert!(result.execute_scalar(3, &mut ctx)?.as_variant().is_null());

        Ok(())
    }

    fn binary_view_array(values: &[&[u8]]) -> ArrowArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for value in values {
            builder.append_value(*value);
        }
        Arc::new(builder.finish())
    }

    fn nullable_binary_view_array(values: &[Option<&[u8]>]) -> ArrowArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for value in values {
            match value {
                Some(value) => builder.append_value(*value),
                None => builder.append_null(),
            }
        }
        Arc::new(builder.finish())
    }

    fn i32_variant_value(value: i32) -> Vec<u8> {
        let mut builder = VariantBuilder::new();
        builder.append_value(value);
        builder.finish().1
    }

    fn object_with_b_value(value: &str) -> (Vec<u8>, Vec<u8>) {
        let mut builder = VariantBuilder::new();
        builder.new_object().with_field("b", value).finish();
        builder.finish()
    }

    fn object_with_a_and_b_value(a: i32, b: &str) -> (Vec<u8>, Vec<u8>) {
        let mut builder = VariantBuilder::new();
        builder
            .new_object()
            .with_field("a", a)
            .with_field("b", b)
            .finish();
        builder.finish()
    }

    fn make_partially_shredded_object_array() -> VortexResult<ArrayRef> {
        let (metadata0, root_value0) = object_with_a_and_b_value(99, "left");
        let (metadata1, root_value1) = object_with_b_value("right");
        let (metadata2, root_value2) = object_with_b_value("missing_a");

        let metadata = nullable_binary_view_array(&[
            Some(metadata0.as_slice()),
            Some(metadata1.as_slice()),
            Some(metadata2.as_slice()),
        ]);
        let value = nullable_binary_view_array(&[
            Some(root_value0.as_slice()),
            Some(root_value1.as_slice()),
            Some(root_value2.as_slice()),
        ]);

        let a_value1 = i32_variant_value(30);
        let a_value = nullable_binary_view_array(&[None, Some(a_value1.as_slice()), None]);
        let a_typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![Some(10), None, None]));
        let a_shredded: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new("typed_value", DataType::Int32, true)),
            ]
            .into(),
            vec![a_value, a_typed_value],
            None,
        )?);

        let typed_value: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![Arc::new(Field::new(
                "a",
                a_shredded.data_type().clone(),
                true,
            ))]
            .into(),
            vec![a_shredded],
            None,
        )?);

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new(
                    "typed_value",
                    typed_value.data_type().clone(),
                    true,
                )),
            ]
            .into(),
            vec![metadata, value, typed_value],
            None,
        )?;
        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        ParquetVariantData::from_arrow_variant(&arrow_variant)
    }

    fn make_canonical_raw_core_with_shredded_child() -> VortexResult<ArrayRef> {
        let raw_core = make_unshredded_json_array(vec![
            Some(r#"{"a": 99, "b": "left"}"#),
            Some(r#"{"a": 30, "b": "right"}"#),
            Some(r#"{"b": "missing_a"}"#),
        ])?;
        let shredded = VortexStructArray::try_from_iter([(
            "a",
            PrimitiveArray::from_option_iter([Some(10i32), None, None]).into_array(),
        )])?;

        Ok(VariantArray::try_new(raw_core, Some(shredded.into_array()))?.into_array())
    }

    fn make_canonical_raw_core_with_nested_nullable_shredded_child() -> VortexResult<ArrayRef> {
        let raw_core = make_unshredded_json_array(vec![
            Some(r#"{"a": {"b": 100}}"#),
            Some(r#"{"a": {"b": 200}}"#),
            Some(r#"{"a": {"b": 300}}"#),
        ])?;
        let nested = VortexStructArray::try_from_iter_with_validity(
            [("b", PrimitiveArray::from_iter([10i32, 20, 30]).into_array())],
            Validity::from_iter([false, true, true]),
        )?;
        let shredded = VortexStructArray::try_from_iter([("a", nested.into_array())])?.into_array();

        Ok(VariantArray::try_new(raw_core, Some(shredded))?.into_array())
    }

    fn assert_i32_scalars(array: &ArrayRef, expected: &[Option<i32>]) -> VortexResult<()> {
        assert_eq!(array.len(), expected.len());
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for (idx, expected) in expected.iter().enumerate() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            match expected {
                Some(expected) => {
                    assert_eq!(scalar.as_primitive().typed_value::<i32>(), Some(*expected));
                }
                None => assert!(scalar.is_null()),
            }
        }
        Ok(())
    }

    fn assert_variant_i32_scalars(array: &ArrayRef, expected: &[Option<i32>]) -> VortexResult<()> {
        assert_eq!(array.len(), expected.len());
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for (idx, expected) in expected.iter().enumerate() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            let variant = scalar.as_variant();
            match expected {
                Some(expected) => {
                    let value = variant
                        .value()
                        .ok_or_else(|| vortex_err!("expected non-null variant scalar"))?;
                    let value =
                        value.cast(&VortexDType::Primitive(PType::I32, Nullability::Nullable))?;
                    assert_eq!(value.as_primitive().typed_value::<i32>(), Some(*expected));
                }
                None => assert!(variant.is_null()),
            }
        }
        Ok(())
    }

    fn assert_utf8_scalars(array: &ArrayRef, expected: &[Option<&str>]) -> VortexResult<()> {
        assert_eq!(array.len(), expected.len());
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for (idx, expected) in expected.iter().enumerate() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            match expected {
                Some(expected) => {
                    assert_eq!(
                        scalar.as_utf8().value().map(|value| value.as_str()),
                        Some(*expected)
                    );
                }
                None => assert!(scalar.is_null()),
            }
        }
        Ok(())
    }

    fn make_shredded_typed_array() -> VortexResult<ArrayRef> {
        let metadata = binary_view_array(&[b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, typed_value],
            None,
        )?;
        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        ParquetVariantData::from_arrow_variant(&arrow_variant)
    }

    #[test]
    fn test_slice_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;

        let sliced = arr.slice(1..3)?;
        assert_eq!(sliced.len(), 2);
        assert_eq!(
            sliced.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            sliced.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_filter_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;
        let filtered = arr.filter(Mask::from_iter([true, false, true]))?;

        assert_eq!(filtered.len(), 2);
        assert_eq!(
            filtered.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            filtered.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_take_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;
        let taken = arr.take(PrimitiveArray::from_iter([2u64, 0, 2]).into_array())?;

        assert_eq!(taken.len(), 3);
        assert_eq!(
            taken.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            taken.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            taken.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_variant_get_direct_parquet_uses_shredded_and_raw_fallback() -> VortexResult<()> {
        let canonical = make_partially_shredded_object_array()?;
        let core_storage = canonical.as_::<Variant>().core_storage().clone();

        let result = execute_variant_get(
            core_storage,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_i32_scalars(&result, &[Some(10), Some(30), None])
    }

    #[test]
    fn test_variant_get_direct_parquet_uses_raw_storage_for_unshredded_path() -> VortexResult<()> {
        let canonical = make_partially_shredded_object_array()?;
        let core_storage = canonical.as_::<Variant>().core_storage().clone();

        let result = execute_variant_get(
            core_storage,
            "$.b",
            Some(VortexDType::Utf8(Nullability::NonNullable)),
        )?;

        assert_utf8_scalars(&result, &[Some("left"), Some("right"), Some("missing_a")])
    }

    #[test]
    fn test_variant_get_canonical_shredded_rewrites_to_core_storage() -> VortexResult<()> {
        let canonical = make_partially_shredded_object_array()?;
        assert!(canonical.as_::<Variant>().shredded().is_some());

        let result = execute_variant_get(
            canonical,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_i32_scalars(&result, &[Some(10), Some(30), None])
    }

    #[test]
    fn test_variant_get_canonical_shredded_child_overrides_raw_core() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_shredded_child()?;

        let result = execute_variant_get(
            canonical,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_i32_scalars(&result, &[Some(10), Some(30), None])
    }

    #[test]
    fn test_variant_get_canonical_shredded_respects_nested_validity() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_nested_nullable_shredded_child()?;

        let result = execute_variant_get(
            canonical,
            "$.a.b",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_i32_scalars(&result, &[Some(100), Some(20), Some(30)])
    }

    #[test]
    fn test_variant_get_canonical_shredded_untyped_uses_typed_rows() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_nested_nullable_shredded_child()?;

        let result = execute_variant_get(canonical, "$.a.b", None)?;

        assert_variant_i32_scalars(&result, &[Some(100), Some(20), Some(30)])
    }

    #[test]
    fn test_variant_get_canonical_core_storage_can_expose_shredded_data() -> VortexResult<()> {
        let canonical = make_partially_shredded_object_array()?;
        let variant = canonical.as_::<Variant>();
        let without_canonical_shredded =
            VariantArray::try_new(variant.core_storage().clone(), None)?.into_array();

        let result = execute_variant_get(
            without_canonical_shredded,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_i32_scalars(&result, &[Some(10), Some(30), None])
    }
}
