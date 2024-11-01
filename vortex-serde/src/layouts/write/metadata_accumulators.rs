//! Metadata accumulators track the per-chunk-of-a-column metadata, layout locations, and row counts.

use std::sync::Arc;

use vortex::array::StructArray;
use vortex::stats::{ArrayStatistics as _, Stat};
use vortex::validity::Validity;
use vortex::{Array, IntoArray};
use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::{match_each_native_ptype, DType, FieldName};
use vortex_error::{VortexError, VortexResult};
use vortex_scalar::Scalar;

pub fn new_metadata_accumulator(dtype: &DType) -> Box<dyn MetadataAccumulator> {
    match dtype {
        DType::Null => Box::new(BasicAccumulator::new()),
        DType::Bool(..) => Box::new(BoolAccumulator::new()),
        DType::Primitive(ptype, ..) => {
            match_each_native_ptype!(ptype, |$P| {
                Box::new(StandardAccumulator::<$P>::new())
            })
        }
        DType::Utf8(..) => Box::new(StandardAccumulator::<BufferString>::new()),
        DType::Binary(..) => Box::new(StandardAccumulator::<Buffer>::new()),
        DType::Struct(..) => Box::new(BasicAccumulator::new()),
        DType::List(..) => Box::new(BasicAccumulator::new()),
        DType::Extension(..) => Box::new(BasicAccumulator::new()),
    }
}

/// Accumulates zero or more series of metadata across the chunks of a column.
pub trait MetadataAccumulator {
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()>;

    fn into_array(self: Box<Self>) -> VortexResult<Array>;
}

/// Accumulator for bool-typed columns.
struct BoolAccumulator {
    row_offsets: RowOffsetsAccumulator,
    maxima: UnwrappedStatAccumulator<bool>,
    minima: UnwrappedStatAccumulator<bool>,
    true_count: UnwrappedStatAccumulator<u64>,
    null_count: UnwrappedStatAccumulator<u64>,
}

impl BoolAccumulator {
    fn new() -> Self {
        Self {
            row_offsets: RowOffsetsAccumulator::new(),
            maxima: UnwrappedStatAccumulator::new(Stat::Max, "max".into()),
            minima: UnwrappedStatAccumulator::new(Stat::Min, "min".into()),
            true_count: UnwrappedStatAccumulator::new(Stat::TrueCount, "true_count".into()),
            null_count: UnwrappedStatAccumulator::new(Stat::NullCount, "null_count".into()),
        }
    }
}

impl MetadataAccumulator for BoolAccumulator {
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()> {
        self.row_offsets.push_chunk(array)?;
        self.maxima.push_chunk(array)?;
        self.minima.push_chunk(array)?;
        self.true_count.push_chunk(array)?;
        self.null_count.push_chunk(array)
    }

    fn into_array(self: Box<Self>) -> VortexResult<Array> {
        let (names, fields): (Vec<FieldName>, Vec<Array>) = [
            self.row_offsets.into_column(),
            self.maxima.into_column(),
            self.minima.into_column(),
            self.true_count.into_column(),
            self.null_count.into_column(),
        ]
        .into_iter()
        .flatten()
        .unzip();
        let names = Arc::from(names);
        let n_chunks = fields[0].len();
        StructArray::try_new(names, fields, n_chunks, Validity::NonNullable)
            .map(IntoArray::into_array)
    }
}

/// An accumulator for the minima, maxima, null counts, and row offsets.
struct StandardAccumulator<T> {
    row_offsets: RowOffsetsAccumulator,
    maxima: UnwrappedStatAccumulator<T>,
    minima: UnwrappedStatAccumulator<T>,
    null_count: UnwrappedStatAccumulator<u64>,
}

impl<T> StandardAccumulator<T> {
    fn new() -> Self {
        Self {
            row_offsets: RowOffsetsAccumulator::new(),
            maxima: UnwrappedStatAccumulator::new(Stat::Max, "max".into()),
            minima: UnwrappedStatAccumulator::new(Stat::Min, "min".into()),
            null_count: UnwrappedStatAccumulator::new(Stat::NullCount, "null_count".into()),
        }
    }
}

impl<T> MetadataAccumulator for StandardAccumulator<T>
where
    T: TryFrom<Scalar, Error = VortexError>,
    Array: From<Vec<Option<T>>>,
{
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()> {
        self.row_offsets.push_chunk(array)?;
        self.maxima.push_chunk(array)?;
        self.minima.push_chunk(array)?;
        self.null_count.push_chunk(array)
    }

    fn into_array(self: Box<Self>) -> VortexResult<Array> {
        let (names, fields): (Vec<FieldName>, Vec<Array>) = [
            self.row_offsets.into_column(),
            self.maxima.into_column(),
            self.minima.into_column(),
            self.null_count.into_column(),
        ]
        .into_iter()
        .flatten()
        .unzip();
        let names = Arc::from(names);
        let n_chunks = fields[0].len();
        StructArray::try_new(names, fields, n_chunks, Validity::NonNullable)
            .map(IntoArray::into_array)
    }
}

/// A minimal accumulator which only tracks null counts and row offsets.
struct BasicAccumulator {
    row_offsets: RowOffsetsAccumulator,
    null_count: UnwrappedStatAccumulator<u64>,
}

impl BasicAccumulator {
    fn new() -> Self {
        Self {
            row_offsets: RowOffsetsAccumulator::new(),
            null_count: UnwrappedStatAccumulator::new(Stat::NullCount, "null_count".into()),
        }
    }
}

impl MetadataAccumulator for BasicAccumulator {
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()> {
        self.row_offsets.push_chunk(array)?;
        self.null_count.push_chunk(array)
    }

    fn into_array(self: Box<Self>) -> VortexResult<Array> {
        let (names, fields): (Vec<FieldName>, Vec<Array>) = [
            self.row_offsets.into_column(),
            self.null_count.into_column(),
        ]
        .into_iter()
        .flatten()
        .unzip();
        let names = Arc::from(names);
        let n_chunks = fields[0].len();
        StructArray::try_new(names, fields, n_chunks, Validity::NonNullable)
            .map(IntoArray::into_array)
    }
}

/// Accumulates a single series of values across the chunks of a column.
trait SingularAccumulator {
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()>;

    fn into_column(self) -> Option<(FieldName, Array)>;
}

struct UnwrappedStatAccumulator<T> {
    stat: Stat,
    name: FieldName,
    values: Vec<Option<T>>,
}

impl<T> UnwrappedStatAccumulator<T> {
    fn new(stat: Stat, name: FieldName) -> Self {
        Self {
            stat,
            name,
            values: Vec::new(),
        }
    }
}

impl<T> SingularAccumulator for UnwrappedStatAccumulator<T>
where
    T: TryFrom<Scalar, Error = VortexError>,
    Array: From<Vec<Option<T>>>,
{
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()> {
        self.values.push(
            array
                .statistics()
                .compute(self.stat)
                .map(T::try_from)
                .transpose()?,
        );
        Ok(())
    }

    fn into_column(self) -> Option<(FieldName, Array)> {
        if self.values.iter().any(Option::is_some) {
            return Some((self.name, Array::from(self.values)));
        }
        None
    }
}

struct RowOffsetsAccumulator {
    row_offsets: Vec<u64>,
    n_rows: u64,
}

impl RowOffsetsAccumulator {
    fn new() -> Self {
        Self {
            row_offsets: Vec::new(),
            n_rows: 0,
        }
    }
}

impl SingularAccumulator for RowOffsetsAccumulator {
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()> {
        self.row_offsets.push(self.n_rows);
        self.n_rows += array.len() as u64;

        Ok(())
    }

    fn into_column(self) -> Option<(FieldName, Array)> {
        // intentionally excluding the last n_rows, b/c it is just the total number of rows
        Some(("row_offsets".into(), Array::from(self.row_offsets)))
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::{BoolArray, ConstantArray, PrimitiveArray};
    use vortex::variants::StructArrayTrait;
    use vortex_dtype::Nullability;

    use super::*;

    fn assert_field_names(struct_array: &StructArray, names: &[&str]) {
        assert_eq!(
            struct_array.names(),
            &names
                .iter()
                .map(|s| FieldName::from(s.to_string()))
                .collect::<Vec<_>>()
                .into()
        );
    }

    #[test]
    fn test_bool_metadata_schema() {
        let mut bool_accumulator = BoolAccumulator::new();
        let chunk = BoolArray::from_vec(vec![true], Validity::AllValid).into_array();
        bool_accumulator.push_chunk(&chunk).unwrap();

        let struct_array =
            StructArray::try_from(Box::new(bool_accumulator).into_array().unwrap()).unwrap();
        assert_eq!(struct_array.len(), 1);
        assert_field_names(
            &struct_array,
            &["row_offsets", "max", "min", "true_count", "null_count"],
        );
    }

    #[test]
    fn test_standard_metadata_schema_nonnullable() {
        let mut standard_accumulator = StandardAccumulator::<u64>::new();
        let chunk = PrimitiveArray::from_nullable_vec(vec![Some(1u64)]).into_array();
        standard_accumulator.push_chunk(&chunk).unwrap();

        let struct_array =
            StructArray::try_from(Box::new(standard_accumulator).into_array().unwrap()).unwrap();
        assert_eq!(struct_array.len(), 1);
        assert_field_names(&struct_array, &["row_offsets", "max", "min", "null_count"]);
    }

    #[test]
    fn test_standard_metadata_schema_nullable() {
        let mut standard_accumulator = StandardAccumulator::<u64>::new();
        let chunk =
            ConstantArray::new(Scalar::primitive(1u64, Nullability::Nullable), 10).into_array();
        standard_accumulator.push_chunk(&chunk).unwrap();

        let struct_array =
            StructArray::try_from(Box::new(standard_accumulator).into_array().unwrap()).unwrap();
        assert_eq!(struct_array.len(), 1);
        assert_field_names(&struct_array, &["row_offsets", "max", "min", "null_count"]);
    }

    #[test]
    fn test_basic_metadata_schema() {
        let mut basic_accumulator = BasicAccumulator::new();
        let chunk = PrimitiveArray::from_nullable_vec(vec![Some(1u64)]).into_array();
        basic_accumulator.push_chunk(&chunk).unwrap();

        let struct_array =
            StructArray::try_from(Box::new(basic_accumulator).into_array().unwrap()).unwrap();
        assert_eq!(struct_array.len(), 1);
        assert_field_names(&struct_array, &["row_offsets", "null_count"]);
    }
}
