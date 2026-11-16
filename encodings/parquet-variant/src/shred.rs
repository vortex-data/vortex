// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shreds unshredded Parquet Variant storage.
//!
//! A reader of an unshredded Variant decodes every value to extract one field of it. Shredding
//! the fields that the values have in common moves each into a typed column of its own, which a
//! reader extracts without touching the rest.
//!
//! Shredding never changes what a Variant holds: a row whose value does not fit the schema keeps
//! it in the residual binary value. So the schema only has to fit most rows to be worth it, and
//! [`InferredShredding`] infers it from a sample.
//!
//! Only strings are shredded for now. The Arrow shredder that does the work casts a value into
//! a typed column when the cast succeeds, so that a double in an integer column or a number in a
//! boolean column would lose its value. A string column takes strings alone, as the Parquet
//! Variant specification requires.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use parquet_variant::Variant as PqVariant;
use parquet_variant_compute::shred_variant;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_layout::layouts::variant::VariantShreddingPolicy;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::arrow::prepare_variant_for_export;

/// Shreds the unshredded storage `array` by `schema`.
///
/// The result holds the same values as `array`, in Parquet Variant storage with a shredded
/// child, and is `nullable` or not.
pub(crate) fn shred(
    array: &ArrayView<'_, ParquetVariant>,
    schema: &DType,
    nullable: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let shredding = shredding_type(schema)?;
    let variants = array.to_arrow(&ArrowExportOptions::default(), ctx)?;
    let shredded = shred_variant(&variants, &shredding)?;

    let session = ctx.session().arrow();
    if nullable {
        ParquetVariant::from_arrow_variant_nullable(&shredded, &session)
    } else {
        ParquetVariant::from_arrow_variant(&shredded, &session)
    }
}

/// The Arrow type that Parquet Variant shredding takes for the schema `dtype`.
///
/// The leaves must be strings; see the module documentation.
fn shredding_type(dtype: &DType) -> VortexResult<DataType> {
    Ok(match dtype {
        DType::Utf8(_) => DataType::Utf8,
        DType::Struct(fields, _) => DataType::Struct(
            fields
                .names()
                .iter()
                .zip(fields.fields())
                .map(|(name, field)| Ok(Field::new(name.as_ref(), shredding_type(&field)?, true)))
                .collect::<VortexResult<Fields>>()?,
        ),
        DType::List(element, _) => DataType::List(Arc::new(Field::new_list_field(
            shredding_type(element)?,
            true,
        ))),
        _ => vortex_bail!(
            "a Parquet Variant can only be shredded into strings, structs and lists, not {dtype}"
        ),
    })
}

/// Shreds the object fields that a sample of the values has in common.
///
/// The schema is inferred from the first chunk of a stream. It shreds each object field that
/// holds a string in every sampled row that has it, and that enough sampled rows have. A field
/// that also holds values of other types is not shredded, and stays in the residual binary
/// value.
#[derive(Clone, Debug)]
pub struct InferredShredding {
    max_sampled_rows: usize,
    max_depth: usize,
    max_fields: usize,
    min_presence_percent: usize,
}

impl Default for InferredShredding {
    fn default() -> Self {
        Self {
            max_sampled_rows: 1024,
            max_depth: 4,
            max_fields: 64,
            min_presence_percent: 1,
        }
    }
}

impl InferredShredding {
    /// Reads at most `rows` rows of the first chunk, spread evenly over it. The default is 1024.
    pub fn with_max_sampled_rows(mut self, rows: usize) -> Self {
        self.max_sampled_rows = rows.max(1);
        self
    }

    /// Shreds fields at most `depth` objects deep. The default is 4.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Shreds at most `fields` fields, those in the most sampled rows. The default is 64.
    pub fn with_max_fields(mut self, fields: usize) -> Self {
        self.max_fields = fields;
        self
    }

    /// Shreds a field only if at least `percent` percent of the sampled rows have it. The
    /// default is 1.
    pub fn with_min_presence_percent(mut self, percent: usize) -> Self {
        self.min_presence_percent = percent.min(100);
        self
    }
}

impl VariantShreddingPolicy for InferredShredding {
    fn schema(&self, first: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<DType>> {
        let storage = prepare_variant_for_export(first.clone(), None, ctx)?;
        let variants = storage
            .as_::<ParquetVariant>()
            .to_arrow(&ArrowExportOptions::default(), ctx)?;

        let len = variants.len();
        let step = len.div_ceil(self.max_sampled_rows).max(1);
        let mut root = Node::default();
        let mut sampled = 0usize;
        for idx in (0..len).step_by(step) {
            if variants.is_valid(idx) {
                sampled += 1;
                root.observe(&variants.try_value(idx)?, 0, self.max_depth);
            }
        }

        let min_rows = (sampled * self.min_presence_percent).div_ceil(100).max(1);
        let mut fields = Vec::new();
        root.collect(&mut Vec::new(), min_rows, &mut fields);

        // The fields in the most rows, and of those the ones that sort first.
        fields.sort_by(|(left_path, left), (right_path, right)| {
            right.cmp(left).then_with(|| left_path.cmp(right_path))
        });
        fields.truncate(self.max_fields);
        if fields.is_empty() {
            return Ok(None);
        }

        let mut shredded = Shredded::default();
        for (path, _) in fields {
            shredded.insert(&path);
        }
        Ok(Some(shredded.dtype()))
    }
}

/// The tree of the fields to shred, which becomes the schema.
#[derive(Default)]
struct Shredded {
    /// The fields of the object at this path. A path without fields is a string.
    fields: BTreeMap<String, Shredded>,
}

impl Shredded {
    fn insert(&mut self, path: &[String]) {
        if let Some((name, rest)) = path.split_first() {
            self.fields.entry(name.clone()).or_default().insert(rest);
        }
    }

    /// A field is only shredded if no sampled row holds an object at it, so a path with fields
    /// is never a string as well.
    fn dtype(&self) -> DType {
        if self.fields.is_empty() {
            return DType::Utf8(Nullability::Nullable);
        }
        DType::Struct(
            StructFields::from_iter(
                self.fields
                    .iter()
                    .map(|(name, field)| (name.as_str(), field.dtype())),
            ),
            Nullability::Nullable,
        )
    }
}

/// What the sampled rows hold at one path.
#[derive(Default)]
struct Node {
    /// How many rows hold a string.
    strings: usize,
    /// How many rows hold a value of another type, objects included.
    others: usize,
    /// The fields of the objects at this path.
    fields: BTreeMap<String, Node>,
}

impl Node {
    fn observe(&mut self, variant: &PqVariant<'_, '_>, depth: usize, max_depth: usize) {
        match variant {
            // A Variant null is in no typed column, and fits beside any.
            PqVariant::Null => {}
            PqVariant::String(_) | PqVariant::ShortString(_) => self.strings += 1,
            PqVariant::Object(object) => {
                self.others += 1;
                if depth < max_depth {
                    for (name, field) in object.iter() {
                        self.fields.entry(name.to_owned()).or_default().observe(
                            &field,
                            depth + 1,
                            max_depth,
                        );
                    }
                }
            }
            _ => self.others += 1,
        }
    }

    /// Appends the fields to shred under this path, each with the number of rows that have it.
    fn collect(
        &self,
        path: &mut Vec<String>,
        min_rows: usize,
        fields: &mut Vec<(Vec<String>, usize)>,
    ) {
        // A path that holds a string in every row that has it. The root is not a field.
        if !path.is_empty() && self.others == 0 && self.strings >= min_rows {
            fields.push((path.clone(), self.strings));
        }

        for (name, field) in &self.fields {
            path.push(name.clone());
            field.collect(path, min_rows, fields);
            path.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::StringArray;
    use parquet_variant_compute::json_to_variant;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::arrays::variant::VariantArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::expr::variant_get;
    use vortex_array::scalar_fn::fns::variant_get::VariantPath;
    use vortex_array::scalar_fn::fns::variant_shred::VariantShred;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    const ROWS: [&str; 4] = [
        r#"{"kind":"commit","time_us":1,"commit":{"collection":"post","rev":"a"},"tags":["x"]}"#,
        r#"{"kind":"commit","time_us":2,"commit":{"collection":"like","rev":7}}"#,
        r#"{"kind":"identity","time_us":3,"mixed":"text"}"#,
        r#"{"kind":"account","time_us":"late","mixed":4}"#,
    ];

    fn variants() -> VortexResult<ArrayRef> {
        let json: ArrowArrayRef = Arc::new(StringArray::from_iter_values(ROWS));
        ParquetVariant::from_arrow_variant(&json_to_variant(&json)?, &SESSION.arrow())
    }

    fn utf8() -> DType {
        DType::Utf8(Nullability::Nullable)
    }

    fn object(fields: &[(&str, DType)]) -> DType {
        DType::Struct(
            StructFields::from_iter(fields.iter().map(|(name, dtype)| (*name, dtype.clone()))),
            Nullability::Nullable,
        )
    }

    fn shred_into(array: ArrayRef, dtype: DType) -> VortexResult<ArrayRef> {
        VariantShred::try_new(array, dtype)?
            .into_array()
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
    }

    fn inferred(policy: &InferredShredding) -> VortexResult<Option<DType>> {
        policy.schema(&variants()?, &mut SESSION.create_execution_ctx())
    }

    fn shredded() -> VortexResult<ArrayRef> {
        let dtype = inferred(&InferredShredding::default())?
            .ok_or_else(|| vortex_err!("nothing to shred"))?;
        shred_into(variants()?, dtype)
    }

    /// The names of the fields of the shredded child of `array`.
    fn shredded_names(array: ArrayRef) -> VortexResult<Vec<String>> {
        let shredded = array
            .execute::<VariantArray>(&mut SESSION.create_execution_ctx())?
            .shredded()
            .ok_or_else(|| vortex_err!("not shredded"))?
            .dtype()
            .clone();
        let DType::Struct(fields, _) = shredded else {
            return Err(vortex_err!("shredded as {shredded}"));
        };
        Ok(fields.names().iter().map(|name| name.to_string()).collect())
    }

    fn get(array: ArrayRef, path: &[&str], dtype: DType) -> VortexResult<ArrayRef> {
        let path = VariantPath::new(path.iter().map(|name| (*name).into()));
        array
            .apply(&variant_get(root(), path, Some(dtype)))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
    }

    #[test]
    fn infers_the_fields_of_one_type() -> VortexResult<()> {
        // `time_us`, `mixed` and `commit.rev` hold two types, and `tags` holds a list.
        assert_eq!(
            inferred(&InferredShredding::default())?,
            Some(object(&[
                ("commit", object(&[("collection", utf8())])),
                ("kind", utf8()),
            ]))
        );
        Ok(())
    }

    #[test]
    fn infers_the_fields_in_the_most_rows() -> VortexResult<()> {
        let policy = InferredShredding::default().with_max_fields(1);
        assert_eq!(inferred(&policy)?, Some(object(&[("kind", utf8())])));
        Ok(())
    }

    #[test]
    fn infers_fields_that_enough_rows_have() -> VortexResult<()> {
        // `commit.collection` is in two rows of four.
        let policy = InferredShredding::default().with_min_presence_percent(75);
        assert_eq!(inferred(&policy)?, Some(object(&[("kind", utf8())])));
        Ok(())
    }

    #[test]
    fn infers_fields_no_deeper_than_the_depth() -> VortexResult<()> {
        let policy = InferredShredding::default().with_max_depth(1);
        assert_eq!(inferred(&policy)?, Some(object(&[("kind", utf8())])));
        Ok(())
    }

    #[test]
    fn infers_nothing_from_values_without_common_fields() -> VortexResult<()> {
        let json: ArrowArrayRef = Arc::new(StringArray::from_iter_values(["1", "[1,2]", "\"a\""]));
        let array = ParquetVariant::from_arrow_variant(&json_to_variant(&json)?, &SESSION.arrow())?;
        let inferred =
            InferredShredding::default().schema(&array, &mut SESSION.create_execution_ctx())?;
        assert!(inferred.is_none());
        Ok(())
    }

    #[test]
    fn shreds_by_the_given_schema() -> VortexResult<()> {
        let schema = object(&[("kind", utf8())]);
        assert_eq!(shredded_names(shred_into(variants()?, schema)?)?, ["kind"]);
        Ok(())
    }

    #[test]
    fn shreds_by_the_inferred_schema() -> VortexResult<()> {
        assert_eq!(shredded_names(shredded()?)?, ["commit", "kind"]);
        Ok(())
    }

    #[rstest]
    #[case::integer(DType::Primitive(PType::I64, Nullability::Nullable))]
    #[case::boolean(DType::Bool(Nullability::Nullable))]
    #[case::variant(DType::Variant(Nullability::Nullable))]
    fn shreds_strings_alone(#[case] leaf: DType) -> VortexResult<()> {
        // The Arrow shredder would cast a double or a boolean into an integer column.
        assert!(shred_into(variants()?, object(&[("time_us", leaf)])).is_err());
        Ok(())
    }

    #[test]
    fn does_not_shred_a_shredded_variant() -> VortexResult<()> {
        assert!(shred_into(shredded()?, object(&[("kind", utf8())])).is_err());
        Ok(())
    }

    #[rstest]
    #[case::shredded(&["kind"])]
    #[case::nested(&["commit", "collection"])]
    #[case::residual(&["mixed"])]
    #[case::residual_nested(&["commit", "rev"])]
    #[case::missing(&["nope"])]
    fn shredding_keeps_the_strings(#[case] path: &[&str]) -> VortexResult<()> {
        let expected = get(variants()?, path, utf8())?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(
            get(shredded()?, path, utf8())?,
            expected,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[rstest]
    #[case::residual(&["time_us"])]
    #[case::residual_nested(&["commit", "rev"])]
    fn shredding_keeps_the_integers(#[case] path: &[&str]) -> VortexResult<()> {
        let int64 = DType::Primitive(PType::I64, Nullability::Nullable);
        let expected = get(variants()?, path, int64.clone())?
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(
            get(shredded()?, path, int64)?,
            expected,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }
}
