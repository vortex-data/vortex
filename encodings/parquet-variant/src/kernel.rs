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

pub(crate) static PARENT_KERNELS: ParentKernelSet<ParquetVariant> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ParquetVariant)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ParquetVariant)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ParquetVariant)),
    ParentKernelSet::lift(&VariantGetKernel),
]);

#[derive(Default, Debug)]
struct VariantGetKernel;

impl ExecuteParentKernel<ParquetVariant> for VariantGetKernel {
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
        let output = if parent.options.dtype().is_none_or(DType::is_variant) {
            let arrow_variant_output = ArrowVariantArray::try_new(arrow_output.as_ref())?;
            ParquetVariant::from_arrow_variant_nullable(&arrow_variant_output)?
        } else {
            ArrayRef::from_arrow(arrow_output.as_ref(), true)?
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
                name.as_ref().to_owned(),
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

    use arrow_array::Array as ArrowArray;
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
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray as VortexStructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::Variant;
    use vortex_array::arrays::VariantArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::arrays::variant::VariantArrayExt;
    use vortex_array::arrow::FromArrowArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar_is_null;
    use vortex_array::dtype::DType as VortexDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::expr::variant_get;
    use vortex_array::scalar_fn::fns::variant_get::VariantPath;
    use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_ensure;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use crate::ParquetVariant;
    use crate::ParquetVariantArrayExt;

    fn make_unshredded_array() -> VortexResult<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(4);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from(99i64));
        ParquetVariant::from_arrow_variant(&builder.build())
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
        ParquetVariant::from_arrow_variant(&arrow_variant)
    }

    fn make_unshredded_json_array(values: Vec<Option<&str>>) -> VortexResult<ArrayRef> {
        let json: ArrowArrayRef = Arc::new(StringArray::from(values));
        let arrow_variant = json_to_variant(&json)?;
        ParquetVariant::from_arrow_variant(&arrow_variant)
    }

    fn parse_path(path: &str) -> VortexResult<VariantPath> {
        if path.is_empty() || path == "$" {
            return Ok(VariantPath::root());
        }

        let mut elements = Vec::new();
        let mut pos = usize::from(path.as_bytes().first() == Some(&b'$'));
        if pos == 1
            && path
                .as_bytes()
                .get(pos)
                .is_some_and(|byte| !matches!(byte, b'.' | b'['))
        {
            vortex_bail!("Invalid Variant path {path:?}: expected '.' or '[' after '$'");
        }

        while pos < path.len() {
            match path.as_bytes()[pos] {
                b'.' => {
                    pos += 1;
                    let (field, next_pos) = parse_field(path, pos)?;
                    elements.push(VariantPathElement::field(field));
                    pos = next_pos;
                }
                b'[' => {
                    let (index, next_pos) = parse_index(path, pos + 1)?;
                    elements.push(VariantPathElement::index(index));
                    pos = next_pos;
                }
                _ if pos == 0 => {
                    let (field, next_pos) = parse_field(path, pos)?;
                    elements.push(VariantPathElement::field(field));
                    pos = next_pos;
                }
                _ => {
                    vortex_bail!("Invalid Variant path {path:?}: expected '.', '[', or end of path")
                }
            }
        }

        Ok(VariantPath::new(elements))
    }

    fn parse_field(path: &str, start: usize) -> VortexResult<(&str, usize)> {
        let mut pos = start;
        while path
            .as_bytes()
            .get(pos)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            pos += 1;
        }
        vortex_ensure!(
            pos > start,
            "Invalid Variant path {path:?}: expected field name"
        );
        Ok((&path[start..pos], pos))
    }

    fn parse_index(path: &str, start: usize) -> VortexResult<(u64, usize)> {
        let mut pos = start;
        while path
            .as_bytes()
            .get(pos)
            .is_some_and(|byte| byte.is_ascii_digit())
        {
            pos += 1;
        }
        vortex_ensure!(
            pos > start,
            "Invalid Variant path {path:?}: expected list index"
        );
        vortex_ensure!(
            path.as_bytes().get(pos) == Some(&b']'),
            "Invalid Variant path {path:?}: expected closing ']'"
        );
        let index = path[start..pos]
            .parse()
            .map_err(|_| vortex_err!("Invalid Variant path {path:?}: list index is too large"))?;
        Ok((index, pos + 1))
    }

    fn execute_variant_get(
        array: ArrayRef,
        path: &str,
        dtype: Option<VortexDType>,
    ) -> VortexResult<ArrayRef> {
        let expr = variant_get(root(), parse_path(path)?, dtype);
        array
            .apply(&expr)?
            .execute::<ArrayRef>(&mut LEGACY_SESSION.create_execution_ctx())
    }

    macro_rules! assert_rows_eq {
        ($actual:expr, $expected:expr, [$($expected_idx:expr),* $(,)?]) => {{
            let actual = $actual;
            let expected = $expected;
            let expected_rows = [$($expected_idx),*];
            assert_eq!(actual.len(), expected_rows.len());

            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            for (actual_idx, expected_idx) in expected_rows.into_iter().enumerate() {
                assert_eq!(
                    actual.execute_scalar(actual_idx, &mut ctx)?,
                    expected.execute_scalar(expected_idx, &mut ctx)?,
                    "row {actual_idx} should match source row {expected_idx}",
                );
            }
        }};
    }

    macro_rules! assert_nulls {
        ($array:expr, [$($is_null:expr),* $(,)?]) => {{
            let array = $array;
            let expected = [$($is_null),*];
            assert_eq!(array.len(), expected.len());

            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            for (idx, is_null) in expected.into_iter().enumerate() {
                assert_eq!(
                    array.execute_scalar(idx, &mut ctx)?.is_null(),
                    is_null,
                    "row {idx} nullness mismatch",
                );
            }
        }};
    }

    #[test]
    fn test_slice_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let sliced = arr.slice(1..3)?;

        assert_rows_eq!(&sliced, &arr, [1, 2]);
        Ok(())
    }

    #[test]
    fn test_slice_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        let sliced = arr.slice(0..3)?;

        assert_nulls!(&sliced, [false, true, false]);
        Ok(())
    }

    #[test]
    fn test_filter_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let mask = Mask::from_iter([true, false, true, false]);
        let filtered = arr.filter(mask)?;

        assert_rows_eq!(&filtered, &arr, [0, 2]);
        Ok(())
    }

    #[test]
    fn test_filter_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        // Keep rows 0 (valid), 1 (null), 3 (null)
        let mask = Mask::from_iter([true, true, false, true]);
        let filtered = arr.filter(mask)?;

        assert_nulls!(&filtered, [false, true, true]);
        Ok(())
    }

    #[test]
    fn test_take_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let indices = PrimitiveArray::from_iter([2u64, 0, 3]);
        let taken = arr.take(indices.into_array())?;

        assert_rows_eq!(&taken, &arr, [2, 0, 3]);
        Ok(())
    }

    #[test]
    fn test_take_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        // Take: valid (0), null (1), null (3), valid (2)
        let indices = PrimitiveArray::from_iter([0u64, 1, 3, 2]);
        let taken = arr.take(indices.into_array())?;

        assert_nulls!(&taken, [false, true, true, false]);
        Ok(())
    }

    #[rstest]
    #[case::field(
        "$.a",
        vec![
            Some(r#"{"a": 1}"#),
            None,
            Some(r#"{"a": null}"#),
            Some(r#"{"a": "wrong"}"#),
            Some(r#"{"b": 2}"#),
        ],
        vec![Some(1), None, None, None, None],
    )]
    #[case::list_index(
        "$.items[1]",
        vec![
            Some(r#"{"items": [10, 20]}"#),
            Some(r#"{"items": []}"#),
            Some(r#"{"items": ["x", 7]}"#),
        ],
        vec![Some(20), None, Some(7)],
    )]
    fn test_variant_get_unshredded_as_i32(
        #[case] path: &str,
        #[case] rows: Vec<Option<&str>>,
        #[case] expected: Vec<Option<i32>>,
    ) -> VortexResult<()> {
        let arr = make_unshredded_json_array(rows)?;
        let result = execute_variant_get(
            arr,
            path,
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_eq!(
            result.dtype(),
            &VortexDType::Primitive(PType::I32, Nullability::Nullable)
        );
        assert_arrays_eq!(result, PrimitiveArray::from_option_iter(expected));
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
        assert_nth_scalar_is_null!(result, 1);
        assert_eq!(
            result
                .execute_scalar(2, &mut ctx)?
                .as_variant()
                .is_variant_null(),
            Some(true)
        );
        assert_nth_scalar_is_null!(result, 3);

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

    fn string_variant_value(value: &str) -> Vec<u8> {
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

    fn make_partially_shredded_arrow_variant() -> VortexResult<ArrowVariantArray> {
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

        let b_value0 = string_variant_value("left");
        let b_value1 = string_variant_value("right");
        let b_value2 = string_variant_value("missing_a");
        let b_value = nullable_binary_view_array(&[
            Some(b_value0.as_slice()),
            Some(b_value1.as_slice()),
            Some(b_value2.as_slice()),
        ]);
        let b_shredded: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![Arc::new(Field::new("value", DataType::BinaryView, true))].into(),
            vec![b_value],
            None,
        )?);

        let typed_value: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![
                Arc::new(Field::new("a", a_shredded.data_type().clone(), true)),
                Arc::new(Field::new("b", b_shredded.data_type().clone(), true)),
            ]
            .into(),
            vec![a_shredded, b_shredded],
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
        Ok(ArrowVariantArray::try_new(&struct_array)?)
    }

    fn make_partially_shredded_parquet_array() -> VortexResult<ArrayRef> {
        let arrow_variant = make_partially_shredded_arrow_variant()?;
        let storage = arrow_variant.inner();
        let value_nullable = storage
            .fields()
            .iter()
            .find(|field| field.name() == "value")
            .map(|field| field.is_nullable())
            .unwrap_or(false);
        let typed_value_nullable = storage
            .fields()
            .iter()
            .find(|field| field.name() == "typed_value")
            .map(|field| field.is_nullable())
            .unwrap_or(false);

        let metadata =
            ArrayRef::from_arrow(arrow_variant.metadata_field() as &dyn ArrowArray, false)?;
        let value = arrow_variant
            .value_field()
            .map(|value| ArrayRef::from_arrow(value as &dyn ArrowArray, value_nullable))
            .transpose()?;
        let typed_value = arrow_variant
            .typed_value_field()
            .map(|typed_value| ArrayRef::from_arrow(typed_value.as_ref(), typed_value_nullable))
            .transpose()?;

        Ok(
            ParquetVariant::try_new(Validity::NonNullable, metadata, value, typed_value)?
                .into_array(),
        )
    }

    fn make_partially_shredded_object_array() -> VortexResult<ArrayRef> {
        let arrow_variant = make_partially_shredded_arrow_variant()?;
        let parquet_array = ParquetVariant::from_arrow_variant(&arrow_variant)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let Canonical::Variant(canonical) = parquet_array.execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant array"));
        };
        Ok(canonical.into_array())
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

    fn assert_variant_object_a_b(
        array: &ArrayRef,
        expected_a: &[Option<i32>],
        expected_b: &[&str],
    ) -> VortexResult<()> {
        assert_eq!(array.len(), expected_a.len());
        assert_eq!(array.len(), expected_b.len());
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for idx in 0..array.len() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            let object = scalar
                .as_variant()
                .value()
                .ok_or_else(|| vortex_err!("expected non-null variant object"))?
                .as_struct();

            match expected_a[idx] {
                Some(expected) => {
                    let value = object
                        .field("a")
                        .ok_or_else(|| vortex_err!("expected field a"))?
                        .as_variant()
                        .value()
                        .ok_or_else(|| vortex_err!("expected non-null field a"))?
                        .cast(&VortexDType::Primitive(PType::I32, Nullability::Nullable))?;
                    assert_eq!(value.as_primitive().typed_value::<i32>(), Some(expected));
                }
                None => assert!(object.field("a").is_none()),
            }

            let field_b = object
                .field("b")
                .ok_or_else(|| vortex_err!("expected field b"))?;
            let value = field_b
                .as_variant()
                .value()
                .ok_or_else(|| vortex_err!("expected non-null field b"))?;
            assert_eq!(
                value.as_utf8().value().map(|value| value.as_str()),
                Some(expected_b[idx])
            );
        }
        Ok(())
    }

    fn make_shredded_typed_array() -> VortexResult<ArrayRef> {
        let metadata = binary_view_array(&[b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![Some(10), None, Some(30)]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, true)),
            ]
            .into(),
            vec![metadata, typed_value],
            None,
        )?;
        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        ParquetVariant::from_arrow_variant(&arrow_variant)
    }

    fn assert_typed_value_i32(
        array: &ArrayRef,
        expected: impl IntoIterator<Item = Option<i32>>,
    ) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let executed = array.clone().execute::<ArrayRef>(&mut ctx)?;
        let typed_value = executed
            .as_::<ParquetVariant>()
            .typed_value_array()
            .ok_or_else(|| vortex_err!("expected typed_value child"))?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(typed_value, PrimitiveArray::from_option_iter(expected));
        Ok(())
    }

    #[test]
    fn test_slice_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;

        let sliced = arr.slice(1..3)?;
        assert_typed_value_i32(&sliced, [None, Some(30)])
    }

    #[test]
    fn test_filter_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;
        let filtered = arr.filter(Mask::from_iter([true, true, false]))?;

        assert_typed_value_i32(&filtered, [Some(10), None])
    }

    #[test]
    fn test_take_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;
        let taken = arr.take(PrimitiveArray::from_iter([2u64, 1, 0]).into_array())?;

        assert_typed_value_i32(&taken, [Some(30), None, Some(10)])
    }

    #[test]
    fn test_variant_get_direct_parquet_uses_shredded_and_raw_fallback() -> VortexResult<()> {
        let parquet = make_partially_shredded_parquet_array()?;

        let result = execute_variant_get(
            parquet,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10), Some(30), None])
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_direct_parquet_uses_raw_storage_for_unshredded_path() -> VortexResult<()> {
        let parquet = make_partially_shredded_parquet_array()?;

        let result = execute_variant_get(
            parquet,
            "$.b",
            Some(VortexDType::Utf8(Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            VarBinArray::from(vec![Some("left"), Some("right"), Some("missing_a")])
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_canonicalized_parquet_uses_top_level_shredded() -> VortexResult<()> {
        let canonical = make_partially_shredded_object_array()?;
        let canonical_variant = canonical.as_::<Variant>();

        let core_storage = canonical_variant
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant core storage"))?;
        assert!(core_storage.typed_value_array().is_none());
        assert!(core_storage.value_array().is_some());

        let shredded = canonical_variant
            .shredded()
            .ok_or_else(|| vortex_err!("expected canonical shredded child"))?
            .clone()
            .execute::<VortexStructArray>(&mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(
            shredded.unmasked_field_by_name("a")?.dtype(),
            &VortexDType::Variant(Nullability::Nullable)
        );
        assert!(shredded.unmasked_field_by_name_opt("b").is_none());

        let result = execute_variant_get(
            canonical,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10), Some(30), None])
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_canonical_shredded_child_overrides_raw_core() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_shredded_child()?;

        let result = execute_variant_get(
            canonical,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10), Some(30), None])
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_canonical_shredded_respects_nested_validity() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_nested_nullable_shredded_child()?;

        let result = execute_variant_get(
            canonical,
            "$.a.b",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(100), Some(20), Some(30)])
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_canonical_shredded_untyped_uses_typed_rows() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_nested_nullable_shredded_child()?;

        let result = execute_variant_get(canonical, "$.a.b", None)?;

        assert_variant_i32_scalars(&result, &[Some(100), Some(20), Some(30)])
    }

    #[test]
    fn test_variant_get_untyped_partial_object_preserves_raw_fields() -> VortexResult<()> {
        let canonical = make_partially_shredded_object_array()?;

        let direct_result =
            execute_variant_get(make_partially_shredded_parquet_array()?, "$", None)?;
        assert_variant_object_a_b(
            &direct_result,
            &[Some(10), Some(30), None],
            &["left", "right", "missing_a"],
        )?;

        let canonical_result = execute_variant_get(canonical, "$", None)?;
        assert_variant_object_a_b(
            &canonical_result,
            &[Some(10), Some(30), None],
            &["left", "right", "missing_a"],
        )
    }

    #[test]
    fn test_canonicalization_moves_shredded_child_out_of_core_storage() -> VortexResult<()> {
        let parquet_array = make_partially_shredded_parquet_array()?;
        assert!(
            parquet_array
                .as_::<ParquetVariant>()
                .typed_value_array()
                .is_some()
        );

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let Canonical::Variant(canonical) = parquet_array.execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant array"));
        };

        assert!(canonical.shredded().is_some());
        let core_storage = canonical
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant core storage"))?;
        assert!(core_storage.typed_value_array().is_none());
        assert!(core_storage.value_array().is_some());

        Ok(())
    }
}
