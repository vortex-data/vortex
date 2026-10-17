// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow plugin impls for the [`FixedShapeTensor`] extension type.
//!
//! [`FixedShapeTensor`] maps to the canonical `arrow.fixed_shape_tensor` extension type. Both
//! store `FixedSizeList<T, N>`, so the data conversion is structural. Only the metadata differs:
//!
//! ```text
//!   Vortex (logical order)              Arrow (physical order, JSON)
//!   logical_shape [500, 100, 200]  <->  "shape":       [100, 200, 500]
//!   dim_names     [W, C, H]        <->  "dim_names":   [C, H, W]
//!   permutation   [2, 0, 1]        <->  "permutations": [2, 0, 1]
//! ```
//!
//! Metadata validation and serialization match arrow-rs, which spells the permutation key
//! `permutations`. The spec key `permutation` is also accepted on import.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::ExtensionType;
use arrow_schema::extension::FixedShapeTensor as ArrowFixedShapeTensor;
use serde::Deserialize;
use serde::Serialize;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::types::arrow::drop_masked_nulls;
use crate::types::arrow::to_logical;
use crate::types::arrow::to_physical;
use crate::types::fixed_shape_tensor::FixedShapeTensor;
use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

/// Canonical Arrow extension name of fixed-shape tensors.
const ARROW_EXTENSION_NAME: &str = "arrow.fixed_shape_tensor";

static ARROW_FIXED_SHAPE_TENSOR: CachedId = CachedId::new(ARROW_EXTENSION_NAME);

/// The `shape` key of the Arrow metadata. arrow-rs validates the metadata but has no public
/// accessor for the shape.
#[derive(Deserialize)]
struct ArrowShape {
    shape: Vec<usize>,
}

/// Renames the spec key `permutation` to the arrow-rs key `permutations`.
///
/// Example: `{"shape": [2, 2], "permutation": [1, 0]}` becomes
/// `{"shape": [2, 2], "permutations": [1, 0]}`. Both keys at once is an error.
fn rename_permutation(json: &str) -> VortexResult<String> {
    let mut map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(json).map_err(|e| vortex_err!("{e}"))?;

    let Some(permutation) = map.remove("permutation") else {
        return Ok(json.to_string());
    };
    if map.contains_key("permutations") {
        vortex_bail!("{ARROW_EXTENSION_NAME} metadata has both permutation and permutations");
    }
    map.insert("permutations".to_string(), permutation);

    serde_json::to_string(&map).map_err(|e| vortex_err!("{e}"))
}

/// Arrow metadata as written on export: arrow-rs keys, unset fields omitted.
///
/// arrow-rs writes unset fields as `null` but rejects `null` when reading, so its own
/// serialization does not round-trip.
#[derive(Serialize)]
struct ArrowMetadataOut<'a> {
    shape: Vec<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    dim_names: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    permutations: Option<&'a [usize]>,
}

/// Converts validated Arrow metadata (physical order) to Vortex metadata (logical order).
fn to_vortex(
    tensor: &ArrowFixedShapeTensor,
    shape: &[usize],
) -> VortexResult<FixedShapeTensorMetadata> {
    let permutation = tensor.permutations();
    let mut metadata = FixedShapeTensorMetadata::new(to_logical(shape, permutation)?);

    if let Some(names) = tensor.dimension_names() {
        metadata = metadata.with_dim_names(to_logical(names, permutation)?)?;
    }
    if let Some(permutation) = permutation {
        metadata = metadata.with_permutation(permutation.to_vec())?;
    }

    Ok(metadata)
}

impl ArrowExportVTable for FixedShapeTensor {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_FIXED_SHAPE_TENSOR
    }

    fn vortex_id(&self) -> Id {
        FixedShapeTensor.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let DType::Extension(dtype) = dtype else {
            return Ok(None);
        };
        let Some(metadata) = dtype.metadata_opt::<FixedShapeTensor>() else {
            return Ok(None);
        };

        // The storage FixedSizeList is already the Arrow storage type.
        let mut field = session.to_arrow_field(name, dtype.storage_dtype())?;
        let DataType::FixedSizeList(elem, _) = field.data_type() else {
            vortex_bail!("FixedShapeTensor storage must export as FixedSizeList");
        };

        let permutation = metadata.permutation();
        let out = ArrowMetadataOut {
            shape: metadata.physical_shape().collect(),
            dim_names: metadata
                .dim_names()
                .map(|names| to_physical(names, permutation)),
            permutations: permutation,
        };

        // Validate with arrow-rs, then replace its metadata JSON with one it can read back.
        let tensor = ArrowFixedShapeTensor::try_new(
            elem.data_type().clone(),
            out.shape.clone(),
            out.dim_names.clone(),
            permutation.map(<[usize]>::to_vec),
        )?;
        field.try_with_extension_type(tensor)?;

        let json = serde_json::to_string(&out).map_err(|e| vortex_err!("{e}"))?;
        let mut field_metadata = field.metadata().clone();
        field_metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_string(), json);
        field.set_metadata(field_metadata);
        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        if !array
            .dtype()
            .as_extension_opt()
            .is_some_and(|ext| ext.is::<FixedShapeTensor>())
        {
            return Ok(ArrowExport::Unsupported(array));
        }

        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();

        let session = ctx.session().clone();
        let arrow_storage = session.arrow().execute_arrow(storage, Some(target), ctx)?;

        Ok(ArrowExport::Exported(arrow_storage))
    }
}

impl ArrowImportVTable for FixedShapeTensor {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_FIXED_SHAPE_TENSOR
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        if field.extension_type_name() != Some(ARROW_EXTENSION_NAME) {
            return Ok(None);
        }
        // Validates the data type and metadata exactly as arrow-rs does, once the spec key
        // `permutation` is renamed to the arrow-rs key `permutations`.
        let mut field_metadata = field.metadata().clone();
        if let Some(json) = field_metadata.get_mut(EXTENSION_TYPE_METADATA_KEY) {
            *json = rename_permutation(json)?;
        }
        let tensor =
            ArrowFixedShapeTensor::try_new_from_field_metadata(field.data_type(), &field_metadata)?;

        let elem_dtype =
            session.from_arrow_datatype(tensor.value_type(), Nullability::NonNullable)?;
        if !elem_dtype.is_primitive() {
            return Ok(None);
        }

        let json = field_metadata
            .get(EXTENSION_TYPE_METADATA_KEY)
            .vortex_expect("arrow-rs requires fixed shape tensor metadata");
        let ArrowShape { shape } = serde_json::from_str(json).map_err(|e| vortex_err!("{e}"))?;

        let storage_dtype = DType::FixedSizeList(
            Arc::new(elem_dtype),
            u32::try_from(tensor.list_size())?,
            field.is_nullable().into(),
        );
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(
                FixedShapeTensor,
                to_vortex(&tensor, &shape)?,
                storage_dtype,
            )?
            .erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        _field: &Field,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let DType::Extension(dtype) = dtype else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !dtype.is::<FixedShapeTensor>() {
            return Ok(ArrowImport::Unsupported(array));
        }
        let Some(fsl) = array.as_fixed_size_list_opt() else {
            return Ok(ArrowImport::Unsupported(array));
        };

        // Arrow allows element nulls under null rows; Vortex elements are non-nullable.
        let values = drop_masked_nulls(
            Arc::clone(fsl.values()),
            fsl.nulls(),
            usize::try_from(fsl.value_length())?,
        )?;
        let elem = Field::new(
            Field::LIST_FIELD_DEFAULT_NAME,
            values.data_type().clone(),
            false,
        );
        let fsl = ArrowFixedSizeListArray::try_new(
            Arc::new(elem),
            fsl.value_length(),
            values,
            fsl.nulls().cloned(),
        )?;

        let storage = session.from_arrow_array(Arc::new(fsl), dtype.is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(dtype.clone(), storage)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
    use arrow_array::Float32Array;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use serde_json::Value;
    use serde_json::json;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowSessionExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::ARROW_EXTENSION_NAME;
    use crate::tests::SESSION;
    use crate::types::fixed_shape_tensor::FixedShapeTensor;
    use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

    /// Logical `[3, 2]` (names `[W, H]`) stored physically as `[2, 3]` (names `[H, W]`).
    fn permuted_metadata() -> VortexResult<FixedShapeTensorMetadata> {
        FixedShapeTensorMetadata::new(vec![3, 2])
            .with_dim_names(vec!["W".into(), "H".into()])?
            .with_permutation(vec![1, 0])
    }

    fn tensor_dtype(metadata: FixedShapeTensorMetadata, nullable: bool) -> VortexResult<DType> {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            6,
            nullable.into(),
        );
        Ok(DType::Extension(
            ExtDType::try_with_vtable(FixedShapeTensor, metadata, storage)?.erased(),
        ))
    }

    /// An Arrow tensor column with element nulls under its null row.
    fn arrow_tensors(metadata: &str) -> (Field, ArrowArrayRef) {
        arrow_tensors_with_item(metadata, false)
    }

    fn arrow_tensors_with_item(metadata: &str, nullable_item: bool) -> (Field, ArrowArrayRef) {
        let item = Arc::new(Field::new("item", DataType::Float32, nullable_item));
        let mut field = Field::new("t", DataType::FixedSizeList(Arc::clone(&item), 4), true);
        field.set_metadata(
            [
                (
                    "ARROW:extension:name".to_string(),
                    ARROW_EXTENSION_NAME.to_string(),
                ),
                ("ARROW:extension:metadata".to_string(), metadata.to_string()),
            ]
            .into(),
        );

        let values = Float32Array::from(vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            None,
            None,
            None,
            None,
        ]);
        let rows = NullBuffer::from(vec![true, false]);
        let array = ArrowFixedSizeListArray::new(item, 4, Arc::new(values), Some(rows));
        (field, Arc::new(array))
    }

    #[test]
    fn to_arrow_field_writes_physical_metadata() -> VortexResult<()> {
        let session = SESSION.arrow();
        let field = session.to_arrow_field("t", &tensor_dtype(permuted_metadata()?, false)?)?;

        assert_eq!(field.extension_type_name(), Some(ARROW_EXTENSION_NAME));
        let metadata: Value = serde_json::from_str(field.extension_type_metadata().unwrap())
            .map_err(|e| vortex_err!("{e}"))?;
        assert_eq!(
            metadata,
            json!({"shape": [2, 3], "dim_names": ["H", "W"], "permutations": [1, 0]})
        );
        Ok(())
    }

    #[rstest]
    #[case::permuted(permuted_metadata())]
    #[case::shape_only(Ok(FixedShapeTensorMetadata::new(vec![2, 3])))]
    fn schema_roundtrip(
        #[case] metadata: VortexResult<FixedShapeTensorMetadata>,
    ) -> VortexResult<()> {
        let session = SESSION.arrow();
        let dtype = tensor_dtype(metadata?, true)?;
        let field = session.to_arrow_field("t", &dtype)?;
        assert_eq!(session.from_arrow_field(&field)?, dtype);
        Ok(())
    }

    #[test]
    fn array_roundtrip() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let session = SESSION.arrow();

        let values = buffer![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ];
        let storage = FixedSizeListArray::try_new(
            values.into_array(),
            6,
            Validity::from_iter([true, false]),
            2,
        )?;
        let original = ExtensionArray::try_new_from_vtable(
            FixedShapeTensor,
            permuted_metadata()?,
            storage.into_array(),
        )?
        .into_array();

        let field = session.to_arrow_field("t", original.dtype())?;
        let arrow = session.execute_arrow(original.clone(), Some(&field), &mut ctx)?;
        let imported = session.from_arrow_array(arrow, &field)?;

        assert_eq!(imported.dtype(), original.dtype());
        vortex_array::assert_arrays_eq!(imported, original, &mut ctx);
        Ok(())
    }

    #[test]
    fn imports_null_rows() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let session = SESSION.arrow();
        let (field, arrow) = arrow_tensors(r#"{"shape": [2, 2], "permutations": [1, 0]}"#);

        let imported = session.from_arrow_array(arrow, &field)?;

        let metadata = FixedShapeTensorMetadata::new(vec![2, 2]).with_permutation(vec![1, 0])?;
        let expected = DType::Extension(
            ExtDType::try_with_vtable(
                FixedShapeTensor,
                metadata,
                DType::FixedSizeList(
                    Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
                    4,
                    Nullability::Nullable,
                ),
            )?
            .erased(),
        );
        assert_eq!(imported.dtype(), &expected);
        assert!(imported.is_valid(0, &mut ctx)?);
        assert!(!imported.is_valid(1, &mut ctx)?);
        Ok(())
    }

    #[rstest]
    #[case::permutations(r#"{"shape": [2, 2], "permutations": [1, 0]}"#, Some(vec![1, 0]))]
    #[case::spec_key(r#"{"shape": [2, 2], "permutation": [1, 0]}"#, Some(vec![1, 0]))]
    #[case::shape_only(r#"{"shape": [2, 2]}"#, None)]
    fn accepts_metadata(
        #[case] json: &str,
        #[case] permutation: Option<Vec<usize>>,
    ) -> VortexResult<()> {
        let session = SESSION.arrow();
        let (field, _) = arrow_tensors(json);

        let dtype = session.from_arrow_field(&field)?;
        let metadata = dtype
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<FixedShapeTensor>().cloned())
            .unwrap();
        assert_eq!(metadata.logical_shape(), &[2, 2]);
        assert_eq!(metadata.permutation(), permutation.as_deref());
        Ok(())
    }

    #[rstest]
    #[case::both_keys(r#"{"shape": [2, 2], "permutation": [1, 0], "permutations": [1, 0]}"#)]
    #[case::unknown_key(r#"{"shape": [2, 2], "extra": 1}"#)]
    #[case::empty_dim_names(r#"{"shape": [2, 2], "dim_names": []}"#)]
    #[case::bad_permutation(r#"{"shape": [2, 2], "permutations": [1, 1]}"#)]
    #[case::list_size_mismatch(r#"{"shape": [2, 3]}"#)]
    #[case::missing_shape(r#"{}"#)]
    #[case::null_dim_names(r#"{"shape": [2, 2], "dim_names": null}"#)]
    fn rejects_metadata(#[case] json: &str) {
        let session = SESSION.arrow();
        let (field, _) = arrow_tensors(json);
        assert!(session.from_arrow_field(&field).is_err());
    }

    #[test]
    fn rejects_nullable_item() {
        let session = SESSION.arrow();
        let (field, _) = arrow_tensors_with_item(r#"{"shape": [2, 2]}"#, true);
        assert!(session.from_arrow_field(&field).is_err());
    }

    #[test]
    fn rejects_null_element_in_valid_row() {
        let session = SESSION.arrow();
        let (field, _) = arrow_tensors(r#"{"shape": [4]}"#);

        let values = Float32Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0)]);
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let arrow = ArrowFixedSizeListArray::new(item, 4, Arc::new(values), None);

        assert!(session.from_arrow_array(Arc::new(arrow), &field).is_err());
    }
}
