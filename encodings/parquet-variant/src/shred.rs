// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shreds an unshredded Parquet Variant by a schema inferred from its values.
//!
//! A reader of an unshredded Variant decodes every value to extract one field of it. Shredding
//! the fields that the values have in common moves each into a typed column of its own, which a
//! reader extracts without touching the rest.
//!
//! Shredding never changes what a Variant holds: a row whose value does not fit the schema keeps
//! it in the residual binary value. So the schema only has to fit most rows to be worth it, and
//! is inferred from a sample.

use std::borrow::Cow;
use std::collections::BTreeMap;

use arrow_schema::DataType;
use parquet_variant::Variant as PqVariant;
use parquet_variant::VariantPath as PqVariantPath;
use parquet_variant::VariantPathElement as PqVariantPathElement;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::shred_variant;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::arrow::prepare_variant_for_export;

/// The most rows [`InferredShredding::infer`] reads, spread evenly over the sample.
const MAX_SAMPLED_ROWS: usize = 1024;

/// How many objects deep a shredded field can be.
const MAX_DEPTH: usize = 4;

/// The most fields a schema shreds. Those in the most rows are kept.
const MAX_FIELDS: usize = 64;

/// A field is shredded if at least one sampled row in this many has it.
const MIN_PRESENCE: usize = 100;

/// A shredding schema inferred from a sample of Variant values.
#[derive(Clone, Debug)]
pub struct InferredShredding {
    schema: DataType,
}

impl InferredShredding {
    /// Infers a shredding schema from the Variant array `sample`.
    ///
    /// The schema shreds the object fields that hold a string, an integer, a double or a boolean
    /// in every sampled row that has them. Returns `None` if there is no such field.
    pub fn infer(sample: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
        let storage = prepare_variant_for_export(sample.clone(), None, ctx)?;
        let variants = storage
            .as_::<ParquetVariant>()
            .to_arrow(&ArrowExportOptions::default(), ctx)?;

        let len = variants.len();
        let step = len.div_ceil(MAX_SAMPLED_ROWS).max(1);
        let mut root = Node::default();
        let mut sampled = 0usize;
        for idx in (0..len).step_by(step) {
            if variants.is_valid(idx) {
                sampled += 1;
                root.observe(&variants.try_value(idx)?, 0);
            }
        }

        let mut fields = Vec::new();
        root.collect(&mut Vec::new(), sampled.div_ceil(MIN_PRESENCE), &mut fields);
        if fields.is_empty() {
            return Ok(None);
        }
        // The fields in the most rows, and of those the ones that sort first.
        fields.sort_by(|(left_path, _, left), (right_path, _, right)| {
            right.cmp(left).then_with(|| left_path.cmp(right_path))
        });
        fields.truncate(MAX_FIELDS);

        let mut builder = ShreddedSchemaBuilder::new();
        for (path, kind, _) in fields {
            let path = PqVariantPath::new(
                path.into_iter()
                    .map(|name| PqVariantPathElement::field(Cow::Owned(name)))
                    .collect(),
            );
            builder = builder.with_path(path, &kind.data_type())?;
        }
        Ok(Some(Self {
            schema: builder.build(),
        }))
    }

    /// Shreds the Variant `array` by this schema.
    ///
    /// The result holds the same values as `array`, in Parquet Variant storage with a shredded
    /// child.
    pub fn shred(&self, array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let nullable = array.dtype().is_nullable();
        let storage = prepare_variant_for_export(array, None, ctx)?;
        let variants = storage
            .as_::<ParquetVariant>()
            .to_arrow(&ArrowExportOptions::default(), ctx)?;
        let shredded = shred_variant(&variants, &self.schema)?;
        let session = ctx.session().arrow();
        if nullable {
            ParquetVariant::from_arrow_variant_nullable(&shredded, &session)
        } else {
            ParquetVariant::from_arrow_variant(&shredded, &session)
        }
    }
}

/// The types a field is shredded as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Utf8,
    Int64,
    Float64,
    Bool,
}

impl Kind {
    const ALL: [Kind; 4] = [Kind::Utf8, Kind::Int64, Kind::Float64, Kind::Bool];

    fn data_type(self) -> DataType {
        match self {
            Kind::Utf8 => DataType::Utf8,
            Kind::Int64 => DataType::Int64,
            Kind::Float64 => DataType::Float64,
            Kind::Bool => DataType::Boolean,
        }
    }
}

/// What the sampled rows hold at one path.
#[derive(Default)]
struct Node {
    /// How many rows hold a value of each of [`Kind::ALL`].
    kinds: [usize; Kind::ALL.len()],
    /// How many rows hold a value that is not shredded, such as a list or an object.
    others: usize,
    /// The fields of the objects at this path.
    fields: BTreeMap<String, Node>,
}

impl Node {
    fn observe(&mut self, variant: &PqVariant<'_, '_>, depth: usize) {
        let kind = match variant {
            // A Variant null is in no typed column, and fits beside any.
            PqVariant::Null => return,
            PqVariant::String(_) | PqVariant::ShortString(_) => Kind::Utf8,
            PqVariant::Int8(_)
            | PqVariant::Int16(_)
            | PqVariant::Int32(_)
            | PqVariant::Int64(_) => Kind::Int64,
            PqVariant::Double(_) => Kind::Float64,
            PqVariant::BooleanTrue | PqVariant::BooleanFalse => Kind::Bool,
            PqVariant::Object(object) => {
                self.others += 1;
                if depth < MAX_DEPTH {
                    for (name, field) in object.iter() {
                        self.fields
                            .entry(name.to_owned())
                            .or_default()
                            .observe(&field, depth + 1);
                    }
                }
                return;
            }
            _ => {
                self.others += 1;
                return;
            }
        };
        self.kinds[Kind::ALL
            .iter()
            .position(|k| *k == kind)
            .unwrap_or_default()] += 1;
    }

    /// Appends the fields to shred under this path, each with the number of rows that have it.
    fn collect(
        &self,
        path: &mut Vec<String>,
        min_rows: usize,
        fields: &mut Vec<(Vec<String>, Kind, usize)>,
    ) {
        // A path that holds one type in every row that has it. The root is not a field.
        let typed = self.kinds.iter().filter(|rows| **rows > 0).count() == 1 && self.others == 0;
        if typed && !path.is_empty() {
            for (kind, rows) in Kind::ALL.iter().zip(self.kinds) {
                if rows >= min_rows.max(1) {
                    fields.push((path.clone(), *kind, rows));
                }
            }
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
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::arrays::variant::VariantArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::expr::variant_get;
    use vortex_array::scalar_fn::fns::variant_get::VariantPath;
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

    fn shredded() -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = variants()?;
        InferredShredding::infer(&array, &mut ctx)?
            .ok_or_else(|| vortex_err!("nothing to shred"))?
            .shred(array, &mut ctx)
    }

    fn get(array: ArrayRef, path: &[&str], dtype: DType) -> VortexResult<ArrayRef> {
        let path = VariantPath::new(path.iter().map(|name| (*name).into()));
        array
            .apply(&variant_get(root(), path, Some(dtype)))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
    }

    #[test]
    fn shreds_the_fields_of_one_type() -> VortexResult<()> {
        let shredded = shredded()?
            .execute::<VariantArray>(&mut SESSION.create_execution_ctx())?
            .shredded()
            .ok_or_else(|| vortex_err!("not shredded"))?
            .dtype()
            .clone();
        let DType::Struct(fields, _) = &shredded else {
            return Err(vortex_err!("shredded as {shredded}"));
        };
        // `time_us` and `mixed` hold two types, and `tags` a list.
        let names = fields
            .names()
            .iter()
            .map(|n| n.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(names, ["commit", "kind"]);
        Ok(())
    }

    #[rstest]
    #[case::shredded(&["kind"])]
    #[case::nested(&["commit", "collection"])]
    #[case::residual(&["mixed"])]
    #[case::residual_nested(&["commit", "rev"])]
    #[case::missing(&["nope"])]
    fn shredding_keeps_the_strings(#[case] path: &[&str]) -> VortexResult<()> {
        let utf8 = DType::Utf8(Nullability::Nullable);
        let expected = get(variants()?, path, utf8.clone())?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(
            get(shredded()?, path, utf8)?,
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

    #[test]
    fn leaves_values_without_common_fields_alone() -> VortexResult<()> {
        let json: ArrowArrayRef = Arc::new(StringArray::from_iter_values(["1", "[1,2]", "\"a\""]));
        let array = ParquetVariant::from_arrow_variant(&json_to_variant(&json)?, &SESSION.arrow())?;
        let inferred = InferredShredding::infer(&array, &mut SESSION.create_execution_ctx())?;
        assert!(inferred.is_none());
        Ok(())
    }
}
