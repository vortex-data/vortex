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
//!   permutation   [2, 0, 1]        <->  "permutation": [2, 0, 1]
//! ```
//!
//! Export writes the canonical Arrow spec key `permutation`. Import accepts both `permutation`
//! and arrow-rs's `permutations` key, validating through arrow-rs. Import also accepts the forms
//! that other producers write: unset options as `null` (arrow-rs) and a nullable list item
//! (pyarrow and Arrow C++).

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
use arrow_array::cast::AsArray;
use arrow_array::make_array;
use arrow_buffer::NullBuffer;
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
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExtVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowMetadataPolicy;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::types::fixed_shape_tensor::FixedShapeTensor;
use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;
use crate::types::fixed_shape_tensor::FixedShapeTensorStoredMetadata;

/// Canonical Arrow extension name of fixed-shape tensors.
const ARROW_EXTENSION_NAME: &str = "arrow.fixed_shape_tensor";

static ARROW_FIXED_SHAPE_TENSOR: CachedId = CachedId::new(ARROW_EXTENSION_NAME);

/// The `shape` key of the Arrow metadata. arrow-rs validates the metadata but has no public
/// accessor for the shape.
#[derive(Deserialize)]
struct ArrowShape {
    shape: Vec<usize>,
}

/// The optional keys of the Arrow metadata, in both the spec and the arrow-rs spelling.
const OPTIONAL_KEYS: [&str; 3] = ["dim_names", "permutation", "permutations"];

/// Converts Arrow metadata to the form that the arrow-rs reader accepts.
///
/// - Removes optional keys whose value is `null`. arrow-rs writes unset options as `null`, but its
///   reader rejects them. Example: `{"shape": [2, 2], "dim_names": null}` becomes
///   `{"shape": [2, 2]}`.
/// - Renames the spec key `permutation` to the arrow-rs key `permutations`. Example:
///   `{"shape": [2, 2], "permutation": [1, 0]}` becomes
///   `{"shape": [2, 2], "permutations": [1, 0]}`. Both keys at once is an error.
fn normalize_metadata(json: &str) -> VortexResult<String> {
    let mut map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(json).map_err(|e| vortex_err!("{e}"))?;

    for key in OPTIONAL_KEYS {
        if map.get(key).is_some_and(serde_json::Value::is_null) {
            map.remove(key);
        }
    }

    if let Some(permutation) = map.remove("permutation") {
        if map.contains_key("permutations") {
            vortex_bail!("{ARROW_EXTENSION_NAME} metadata has both permutation and permutations");
        }
        map.insert("permutations".to_string(), permutation);
    }

    serde_json::to_string(&map).map_err(|e| vortex_err!("{e}"))
}

/// Returns `data_type` with a non-nullable list item, for validation by arrow-rs.
///
/// arrow-rs accepts only a non-nullable item, but pyarrow and Arrow C++ always declare the item
/// nullable. The import of the array checks the actual nulls.
fn non_nullable_item(data_type: &DataType) -> DataType {
    match data_type {
        DataType::FixedSizeList(item, size) if item.is_nullable() => {
            DataType::FixedSizeList(Arc::new(item.as_ref().clone().with_nullable(false)), *size)
        }
        _ => data_type.clone(),
    }
}

/// Arrow metadata as written on export: canonical Arrow spec keys, unset fields omitted.
///
/// arrow-rs writes unset fields as `null` (which it then rejects when reading) and spells
/// `permutation` as `permutations`, so we serialize the spec-compliant JSON directly.
#[derive(Serialize)]
struct ArrowMetadataOut<'a> {
    shape: Vec<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    dim_names: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    permutation: Option<&'a [usize]>,
}

/// Converts validated Arrow metadata (physical order) to Vortex metadata (logical order).
fn to_vortex(
    tensor: &ArrowFixedShapeTensor,
    element_ptype: PType,
    shape: &[usize],
) -> VortexResult<FixedShapeTensorMetadata> {
    let stored = FixedShapeTensorStoredMetadata::from_physical(
        shape,
        tensor.dimension_names(),
        tensor.permutations(),
    )?;

    Ok(FixedShapeTensorMetadata::from_stored_metadata(
        element_ptype,
        stored,
    ))
}

impl ArrowExtVTable for FixedShapeTensor {
    /// The Arrow metadata holds the shape, the dimension names, and the permutation, so the Arrow
    /// field alone restores the full Vortex dtype.
    fn metadata_policy() -> ArrowMetadataPolicy {
        ArrowMetadataPolicy::ArrowNative
    }

    fn arrow_ext_id(&self) -> Id {
        *ARROW_FIXED_SHAPE_TENSOR
    }

    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let metadata = ext_dtype.metadata();

        // The storage FixedSizeList is already the Arrow storage type.
        let mut field = session.to_arrow_field(name, ext_dtype.storage_dtype())?;
        let DataType::FixedSizeList(elem, _) = field.data_type() else {
            vortex_bail!("FixedShapeTensor storage must export as FixedSizeList");
        };

        let permutation = metadata.permutation();
        let out = ArrowMetadataOut {
            shape: metadata.physical_shape().collect(),
            dim_names: metadata.physical_dim_names(),
            permutation,
        };

        // Validate with arrow-rs, then replace its metadata JSON with canonical spec JSON.
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
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let executed = array.execute::<ExtensionArray>(ctx)?;
        let Some(source_metadata) = executed.ext_dtype().metadata_opt::<FixedShapeTensor>() else {
            return Ok(ArrowExport::Unsupported(executed.into_array()));
        };
        let target_dtype =
            Self::from_arrow_field(target, &ctx.session().arrow())?.ok_or_else(|| {
                vortex_err!(
                    "Arrow field {:?} is not a supported {ARROW_EXTENSION_NAME}",
                    target.name()
                )
            })?;
        vortex_ensure_eq!(
            target_dtype.metadata(),
            source_metadata,
            "Arrow field {:?} tensor metadata does not match source dtype",
            target.name()
        );

        let storage = executed.storage_array().clone();

        let arrow_storage = storage.execute_arrow_with_options(Some(target), options, ctx)?;

        Ok(ArrowExport::Exported(arrow_storage))
    }

    fn from_arrow_field(
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<ExtDType<Self>>> {
        if field.extension_type_name() != Some(ARROW_EXTENSION_NAME) {
            return Ok(None);
        }
        // Validates the data type and metadata with arrow-rs. Before that, normalizes the metadata
        // and the item nullability to the form that arrow-rs accepts, and checks that the shape
        // product fits in `i32` (arrow-rs panics on `i32` overflow).
        let mut field_metadata = field.metadata().clone();
        let shape = if let Some(json) = field_metadata.get(EXTENSION_TYPE_METADATA_KEY) {
            let json = normalize_metadata(json)?;
            let ArrowShape { shape } =
                serde_json::from_str(&json).map_err(|e| vortex_err!("{e}"))?;
            field_metadata.insert(EXTENSION_TYPE_METADATA_KEY, json);
            shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .and_then(|product| i32::try_from(product).ok())
                .ok_or_else(|| {
                    vortex_err!("{ARROW_EXTENSION_NAME} shape {shape:?} overflows i32 list size")
                })?;
            Some(shape)
        } else {
            None
        };
        let tensor = ArrowFixedShapeTensor::try_new_from_field_metadata(
            &non_nullable_item(field.data_type()),
            &field_metadata,
        )?;

        let elem_dtype =
            session.from_arrow_datatype(tensor.value_type(), Nullability::NonNullable)?;
        let DType::Primitive(element_ptype, _) = elem_dtype else {
            vortex_bail!(
                "{ARROW_EXTENSION_NAME} elements must be primitive, got {}",
                tensor.value_type()
            );
        };

        let shape = shape.vortex_expect("arrow-rs requires fixed shape tensor metadata");

        Ok(Some(ExtDType::try_with_vtable(
            FixedShapeTensor,
            to_vortex(&tensor, element_ptype, &shape)?,
            field.is_nullable().into(),
        )?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        _field: &Field,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
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
        // The length is explicit because a zero-size dimension makes the list size zero.
        let fsl = ArrowFixedSizeListArray::try_new_with_length(
            Arc::new(elem),
            fsl.value_length(),
            values,
            fsl.nulls().cloned(),
            fsl.len(),
        )?;

        let storage =
            session.from_arrow_array(Arc::new(fsl), ext_dtype.storage_dtype().is_nullable())?;

        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(ext_dtype.clone().erased(), storage)?.into_array(),
        ))
    }
}

/// Removes the nulls of `child` that sit under null `parent` rows.
///
/// Vortex tensor children are non-nullable, but Arrow producers (e.g. pyarrow) write nulls under
/// null parent rows. Those slots are undefined, so the nulls are dropped. A null under a valid
/// parent row is an error.
///
/// `width` is the number of child slots per parent row: 1 for struct children, the list size for
/// fixed-size list values.
fn drop_masked_nulls(
    child: ArrowArrayRef,
    parent: Option<&NullBuffer>,
    width: usize,
) -> VortexResult<ArrowArrayRef> {
    let Some(child_nulls) = child.nulls().filter(|nulls| nulls.null_count() > 0) else {
        return Ok(child);
    };

    for i in (0..child.len()).filter(|&i| child_nulls.is_null(i)) {
        vortex_ensure!(
            parent.is_some_and(|parent| parent.is_null(i / width)),
            "tensor child value at index {i} is null under a valid row"
        );
    }

    let data = child.to_data().into_builder().nulls(None).build()?;
    Ok(make_array(data))
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
    use arrow_schema::Metadata;
    use arrow_schema::extension::FixedShapeTensor as ArrowFixedShapeTensor;
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
    use vortex_arrow::ExecuteArrowExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::ARROW_EXTENSION_NAME;
    use crate::tests::SESSION;
    use crate::types::fixed_shape_tensor::FixedShapeTensor;
    use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

    /// Logical `[3, 2]` (names `[W, H]`) stored physically as `[2, 3]` (names `[H, W]`).
    fn permuted_metadata() -> VortexResult<FixedShapeTensorMetadata> {
        FixedShapeTensorMetadata::new(PType::F32, vec![3, 2])
            .with_dim_names(vec!["W".into(), "H".into()])?
            .with_permutation(vec![1, 0])
    }

    fn tensor_dtype(metadata: FixedShapeTensorMetadata, nullable: bool) -> VortexResult<DType> {
        Ok(DType::Extension(
            ExtDType::try_with_vtable(FixedShapeTensor, metadata, nullable.into())?.erased(),
        ))
    }

    /// An Arrow tensor column with element nulls under its null row.
    fn arrow_tensors(metadata: &str) -> (Field, ArrowArrayRef) {
        arrow_tensors_with_item(metadata, false)
    }

    fn arrow_tensors_with_item(metadata: &str, nullable_item: bool) -> (Field, ArrowArrayRef) {
        let item = Arc::new(Field::new("item", DataType::Float32, nullable_item));
        let mut field = Field::new("t", DataType::FixedSizeList(Arc::clone(&item), 4), true);
        field.set_metadata(Metadata::from_iter([
            (
                "ARROW:extension:name".to_string(),
                ARROW_EXTENSION_NAME.to_string(),
            ),
            ("ARROW:extension:metadata".to_string(), metadata.to_string()),
        ]));

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
            json!({"shape": [2, 3], "dim_names": ["H", "W"], "permutation": [1, 0]})
        );
        Ok(())
    }

    #[rstest]
    #[case::permuted(permuted_metadata())]
    #[case::shape_only(Ok(FixedShapeTensorMetadata::new(PType::F32, vec![2, 3])))]
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
        let arrow = original.clone().execute_arrow(Some(&field), &mut ctx)?;
        let imported = session.from_arrow_array(arrow, &field)?;

        assert_eq!(imported.dtype(), original.dtype());
        vortex_array::assert_arrays_eq!(imported, original, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::non_nullable_item(false)]
    #[case::nullable_item(true)]
    fn imports_null_rows(#[case] nullable_item: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let session = SESSION.arrow();
        let (field, arrow) = arrow_tensors_with_item(
            r#"{"shape": [2, 2], "permutations": [1, 0]}"#,
            nullable_item,
        );

        let imported = session.from_arrow_array(arrow, &field)?;

        let metadata =
            FixedShapeTensorMetadata::new(PType::F32, vec![2, 2]).with_permutation(vec![1, 0])?;
        let expected = DType::Extension(
            ExtDType::try_with_vtable(FixedShapeTensor, metadata, Nullability::Nullable)?.erased(),
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
    #[case::null_dim_names(r#"{"shape": [2, 2], "dim_names": null}"#, None)]
    #[case::null_options(r#"{"shape": [2, 2], "dim_names": null, "permutations": null}"#, None)]
    #[case::null_dim_names_with_permutations(
        r#"{"shape": [2, 2], "dim_names": null, "permutations": [1, 0]}"#,
        Some(vec![1, 0])
    )]
    #[case::null_spelling_with_spec_key(
        r#"{"shape": [2, 2], "permutation": [1, 0], "permutations": null}"#,
        Some(vec![1, 0])
    )]
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
    #[case::null_shape(r#"{"shape": null}"#)]
    #[case::shape_overflows_i32(r#"{"shape": [2147483648]}"#)]
    #[case::shape_product_overflows_i32(r#"{"shape": [65536, 65536]}"#)]
    fn rejects_metadata(#[case] json: &str) {
        let session = SESSION.arrow();
        let (field, _) = arrow_tensors(json);
        assert!(session.from_arrow_field(&field).is_err());
    }

    #[test]
    fn rejects_non_primitive_element_type() {
        let session = SESSION.arrow();
        let item = Arc::new(Field::new("item", DataType::Boolean, false));
        let mut field = Field::new("t", DataType::FixedSizeList(item, 4), true);
        field.set_metadata(Metadata::from_iter([
            (
                "ARROW:extension:name".to_string(),
                ARROW_EXTENSION_NAME.to_string(),
            ),
            (
                "ARROW:extension:metadata".to_string(),
                r#"{"shape": [2, 2]}"#.to_string(),
            ),
        ]));
        assert!(session.from_arrow_field(&field).is_err());
    }

    #[test]
    fn rejects_mismatched_export_target() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let session = SESSION.arrow();

        let values = buffer![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage =
            FixedSizeListArray::try_new(values.into_array(), 6, Validity::NonNullable, 1)?;
        let array = ExtensionArray::try_new_from_vtable(
            FixedShapeTensor,
            FixedShapeTensorMetadata::new(PType::F32, vec![2, 3]),
            storage.into_array(),
        )?
        .into_array();

        let mismatched_target = session.to_arrow_field(
            "t",
            &tensor_dtype(FixedShapeTensorMetadata::new(PType::F32, vec![3, 2]), false)?,
        )?;
        assert!(
            array
                .execute_arrow(Some(&mismatched_target), &mut ctx)
                .is_err()
        );
        Ok(())
    }

    /// arrow-rs writes unset options as `null`.
    #[rstest]
    #[case::shape_only(None, None)]
    #[case::dim_names(Some(vec!["H".to_string(), "W".to_string()]), None)]
    #[case::permutations(None, Some(vec![1, 0]))]
    #[case::both(Some(vec!["H".to_string(), "W".to_string()]), Some(vec![1, 0]))]
    fn imports_arrow_rs_metadata(
        #[case] dim_names: Option<Vec<String>>,
        #[case] permutations: Option<Vec<usize>>,
    ) -> VortexResult<()> {
        let session = SESSION.arrow();
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let tensor = ArrowFixedShapeTensor::try_new(
            DataType::Float32,
            vec![2, 2],
            dim_names,
            permutations.clone(),
        )?;
        let mut field = Field::new("t", DataType::FixedSizeList(item, 4), true);
        field.try_with_extension_type(tensor)?;

        let dtype = session.from_arrow_field(&field)?;
        let metadata = dtype
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<FixedShapeTensor>().cloned())
            .unwrap();
        assert_eq!(metadata.permutation(), permutations.as_deref());
        Ok(())
    }

    /// pyarrow and Arrow C++ declare the list item nullable.
    #[test]
    fn accepts_nullable_item() -> VortexResult<()> {
        let session = SESSION.arrow();
        let (nullable, _) = arrow_tensors_with_item(r#"{"shape": [2, 2]}"#, true);
        let (non_nullable, _) = arrow_tensors_with_item(r#"{"shape": [2, 2]}"#, false);

        assert_eq!(
            session.from_arrow_field(&nullable)?,
            session.from_arrow_field(&non_nullable)?
        );
        Ok(())
    }

    #[test]
    fn exports_to_nullable_item_target() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let session = SESSION.arrow();

        let values = buffer![1.0_f32, 2.0, 3.0, 4.0];
        let storage = FixedSizeListArray::try_new(values.into_array(), 4, Validity::AllValid, 1)?;
        let original = ExtensionArray::try_new_from_vtable(
            FixedShapeTensor,
            FixedShapeTensorMetadata::new(PType::F32, vec![2, 2]),
            storage.into_array(),
        )?
        .into_array();
        let (target, _) = arrow_tensors_with_item(r#"{"shape": [2, 2]}"#, true);

        let arrow = original.clone().execute_arrow(Some(&target), &mut ctx)?;
        assert_eq!(arrow.data_type(), target.data_type());

        let imported = session.from_arrow_array(arrow, &target)?;
        vortex_array::assert_arrays_eq!(imported, original, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::non_nullable_item(false)]
    #[case::nullable_item(true)]
    fn rejects_null_element_in_valid_row(#[case] nullable_item: bool) {
        let session = SESSION.arrow();
        let (field, _) = arrow_tensors_with_item(r#"{"shape": [4]}"#, nullable_item);

        let values = Float32Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0)]);
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let arrow = ArrowFixedSizeListArray::new(item, 4, Arc::new(values), None);

        assert!(session.from_arrow_array(Arc::new(arrow), &field).is_err());
    }
}
