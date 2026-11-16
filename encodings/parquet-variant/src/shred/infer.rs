// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Infers a shredding schema from a sample of the values.

use std::collections::BTreeMap;

use parquet_variant::Variant as PqVariant;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_arrow::ArrowExportOptions;
use vortex_error::VortexResult;
use vortex_layout::layouts::variant::VariantShreddingPolicy;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::arrow::prepare_variant_for_export;

/// Shreds the object fields that a sample of the values has in common.
///
/// The schema is inferred from the first chunk of a stream. It shreds each object field that
/// holds values of one type in every sampled row that has it, and that enough sampled rows have.
/// The types are strings, booleans, integers of any width, which shred as 64-bit integers, and
/// doubles. A field that also holds values of other types is not shredded, and stays in the
/// residual binary value.
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
        fields.sort_by(|(left_path, left, _), (right_path, right, _)| {
            right.cmp(left).then_with(|| left_path.cmp(right_path))
        });
        fields.truncate(self.max_fields);
        if fields.is_empty() {
            return Ok(None);
        }

        let mut shredded = Shredded::default();
        for (path, _, kind) in fields {
            shredded.insert(&path, kind);
        }
        Ok(Some(shredded.dtype()))
    }
}

/// The types of value that a field is shredded into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Utf8,
    Bool,
    Integer,
    Double,
}

impl Kind {
    const ALL: [Self; 4] = [Self::Utf8, Self::Bool, Self::Integer, Self::Double];

    fn of(variant: &PqVariant<'_, '_>) -> Option<Self> {
        Some(match variant {
            PqVariant::String(_) | PqVariant::ShortString(_) => Self::Utf8,
            PqVariant::BooleanTrue | PqVariant::BooleanFalse => Self::Bool,
            PqVariant::Int8(_)
            | PqVariant::Int16(_)
            | PqVariant::Int32(_)
            | PqVariant::Int64(_) => Self::Integer,
            PqVariant::Double(_) => Self::Double,
            _ => return None,
        })
    }

    fn dtype(self) -> DType {
        match self {
            Self::Utf8 => DType::Utf8(Nullability::Nullable),
            Self::Bool => DType::Bool(Nullability::Nullable),
            Self::Integer => DType::Primitive(PType::I64, Nullability::Nullable),
            Self::Double => DType::Primitive(PType::F64, Nullability::Nullable),
        }
    }
}

/// The tree of the fields to shred, which becomes the schema.
#[derive(Default)]
struct Shredded {
    /// The type of the values at this path, if it is a field to shred.
    kind: Option<Kind>,
    /// The fields of the object at this path.
    fields: BTreeMap<String, Shredded>,
}

impl Shredded {
    fn insert(&mut self, path: &[String], kind: Kind) {
        match path.split_first() {
            Some((name, rest)) => self
                .fields
                .entry(name.clone())
                .or_default()
                .insert(rest, kind),
            None => self.kind = Some(kind),
        }
    }

    /// A field is only shredded if no sampled row holds an object at it, so a path with fields
    /// never has a type as well.
    fn dtype(&self) -> DType {
        if let Some(kind) = self.kind {
            return kind.dtype();
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
    /// How many rows hold a value of each [`Kind`], in the order of [`Kind::ALL`].
    kinds: [usize; Kind::ALL.len()],
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
            other => match Kind::of(other) {
                Some(kind) => self.kinds[kind as usize] += 1,
                None => self.others += 1,
            },
        }
    }

    /// The type of the values at this path, if every row that has a value holds one of it.
    fn kind(&self) -> Option<(Kind, usize)> {
        if self.others != 0 {
            return None;
        }
        let mut found = None;
        for (kind, rows) in Kind::ALL.into_iter().zip(self.kinds) {
            if rows > 0 {
                if found.is_some() {
                    return None;
                }
                found = Some((kind, rows));
            }
        }
        found
    }

    /// Appends the fields to shred under this path, each with the number of rows that have it
    /// and the type of its values.
    fn collect(
        &self,
        path: &mut Vec<String>,
        min_rows: usize,
        fields: &mut Vec<(Vec<String>, usize, Kind)>,
    ) {
        // The root is not a field.
        if !path.is_empty()
            && let Some((kind, rows)) = self.kind()
            && rows >= min_rows
        {
            fields.push((path.clone(), rows, kind));
        }

        for (name, field) in &self.fields {
            path.push(name.clone());
            field.collect(path, min_rows, fields);
            path.pop();
        }
    }
}
