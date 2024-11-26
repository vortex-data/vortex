//! Metadata accumulators track the per-chunk-of-a-column metadata, layout locations, and row counts.

use std::sync::Arc;

use vortex_array::array::StructArray;
use vortex_array::stats::{ArrayStatistics as _, Stat};
use vortex_array::validity::Validity;
use vortex_array::{ArrayData, IntoArrayData};
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
pub trait MetadataAccumulator: Send {
    fn push_chunk(&mut self, array: &ArrayData);

    fn into_array(self: Box<Self>) -> VortexResult<Option<ArrayData>>;
}

/// Accumulator for bool-typed columns.
struct BoolAccumulator {
    maxima: UnwrappedStatAccumulator<bool>,
    minima: UnwrappedStatAccumulator<bool>,
    true_count: UnwrappedStatAccumulator<u64>,
    null_count: UnwrappedStatAccumulator<u64>,
}

impl BoolAccumulator {
    fn new() -> Self {
        Self {
            maxima: UnwrappedStatAccumulator::new(Stat::Max, "max".into()),
            minima: UnwrappedStatAccumulator::new(Stat::Min, "min".into()),
            true_count: UnwrappedStatAccumulator::new(Stat::TrueCount, "true_count".into()),
            null_count: UnwrappedStatAccumulator::new(Stat::NullCount, "null_count".into()),
        }
    }
}

impl MetadataAccumulator for BoolAccumulator {
    fn push_chunk(&mut self, array: &ArrayData) {
        self.maxima.push_chunk(array);
        self.minima.push_chunk(array);
        self.true_count.push_chunk(array);
        self.null_count.push_chunk(array);
    }

    fn into_array(self: Box<Self>) -> VortexResult<Option<ArrayData>> {
        let (names, fields): (Vec<FieldName>, Vec<ArrayData>) = [
            self.maxima.into_column(),
            self.minima.into_column(),
            self.true_count.into_column(),
            self.null_count.into_column(),
        ]
        .into_iter()
        .flatten()
        .unzip();

        if fields.is_empty() {
            Ok(None)
        } else {
            let names = Arc::from(names);
            let n_chunks = fields[0].len();
            StructArray::try_new(names, fields, n_chunks, Validity::NonNullable)
                .map(IntoArrayData::into_array)
                .map(Some)
        }
    }
}

/// An accumulator for the minima, maxima, null counts.
struct StandardAccumulator<T> {
    maxima: UnwrappedStatAccumulator<T>,
    minima: UnwrappedStatAccumulator<T>,
    null_count: UnwrappedStatAccumulator<u64>,
}

impl<T> StandardAccumulator<T> {
    fn new() -> Self {
        Self {
            maxima: UnwrappedStatAccumulator::new(Stat::Max, "max".into()),
            minima: UnwrappedStatAccumulator::new(Stat::Min, "min".into()),
            null_count: UnwrappedStatAccumulator::new(Stat::NullCount, "null_count".into()),
        }
    }
}

impl<T: Send> MetadataAccumulator for StandardAccumulator<T>
where
    Option<T>: TryFrom<Scalar, Error = VortexError>,
    ArrayData: FromIterator<Option<T>>,
{
    fn push_chunk(&mut self, array: &ArrayData) {
        self.maxima.push_chunk(array);
        self.minima.push_chunk(array);
        self.null_count.push_chunk(array);
    }

    fn into_array(self: Box<Self>) -> VortexResult<Option<ArrayData>> {
        let (names, fields): (Vec<FieldName>, Vec<ArrayData>) = [
            self.maxima.into_column(),
            self.minima.into_column(),
            self.null_count.into_column(),
        ]
        .into_iter()
        .flatten()
        .unzip();
        if fields.is_empty() {
            Ok(None)
        } else {
            let names = Arc::from(names);
            let n_chunks = fields[0].len();
            StructArray::try_new(names, fields, n_chunks, Validity::NonNullable)
                .map(IntoArrayData::into_array)
                .map(Some)
        }
    }
}

/// A minimal accumulator which only tracks null counts.
struct BasicAccumulator {
    null_count: UnwrappedStatAccumulator<u64>,
}

impl BasicAccumulator {
    fn new() -> Self {
        Self {
            null_count: UnwrappedStatAccumulator::new(Stat::NullCount, "null_count".into()),
        }
    }
}

impl MetadataAccumulator for BasicAccumulator {
    fn push_chunk(&mut self, array: &ArrayData) {
        self.null_count.push_chunk(array)
    }

    fn into_array(self: Box<Self>) -> VortexResult<Option<ArrayData>> {
        let (names, fields): (Vec<FieldName>, Vec<ArrayData>) = [self.null_count.into_column()]
            .into_iter()
            .flatten()
            .unzip();
        if fields.is_empty() {
            Ok(None)
        } else {
            let names = Arc::from(names);
            let n_chunks = fields[0].len();
            StructArray::try_new(names, fields, n_chunks, Validity::NonNullable)
                .map(IntoArrayData::into_array)
                .map(Some)
        }
    }
}

/// Accumulates a single series of values across the chunks of a column.
trait SingularAccumulator {
    fn push_chunk(&mut self, array: &ArrayData);

    fn into_column(self) -> Option<(FieldName, ArrayData)>;
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
    Option<T>: TryFrom<Scalar, Error = VortexError>,
    ArrayData: FromIterator<Option<T>>,
{
    fn push_chunk(&mut self, array: &ArrayData) {
        self.values.push(
            array
                .statistics()
                .compute(self.stat)
                .and_then(|s| Option::<T>::try_from(s).ok())
                .flatten(),
        )
    }

    fn into_column(self) -> Option<(FieldName, ArrayData)> {
        if self.values.iter().any(Option::is_some) {
            return Some((self.name, ArrayData::from_iter(self.values)));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::{BoolArray, ConstantArray, PrimitiveArray};
    use vortex_array::variants::StructArrayTrait;
    use vortex_array::ArrayLen;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

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
        let chunk = BoolArray::from_iter([true]).into_array();
        bool_accumulator.push_chunk(&chunk);

        let struct_array =
            StructArray::try_from(Box::new(bool_accumulator).into_array().unwrap().unwrap())
                .unwrap();
        assert_eq!(struct_array.len(), 1);
        assert_field_names(&struct_array, &["max", "min", "true_count", "null_count"]);
    }

    #[test]
    fn test_standard_metadata_schema_nonnullable() {
        let mut standard_accumulator = StandardAccumulator::<u64>::new();
        let chunk = PrimitiveArray::from_nullable_vec(vec![Some(1u64)]).into_array();
        standard_accumulator.push_chunk(&chunk);

        let struct_array = StructArray::try_from(
            Box::new(standard_accumulator)
                .into_array()
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(struct_array.len(), 1);
        assert_field_names(&struct_array, &["max", "min", "null_count"]);
    }

    #[test]
    fn test_standard_metadata_schema_nullable() {
        let mut standard_accumulator = StandardAccumulator::<u64>::new();
        let chunk =
            ConstantArray::new(Scalar::primitive(1u64, Nullability::Nullable), 10).into_array();
        standard_accumulator.push_chunk(&chunk);

        let struct_array = StructArray::try_from(
            Box::new(standard_accumulator)
                .into_array()
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(struct_array.len(), 1);
        assert_field_names(&struct_array, &["max", "min", "null_count"]);
    }

    #[test]
    fn test_standard_metadata_all_null() {
        let mut standard_accumulator = StandardAccumulator::<u64>::new();
        let chunk = ConstantArray::new(Scalar::null_typed::<u64>(), 10).into_array();
        standard_accumulator.push_chunk(&chunk);

        let metadata_array = StructArray::try_from(
            Box::new(standard_accumulator)
                .into_array()
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata_array.len(), 1);
        assert_field_names(&metadata_array, &["null_count"]);
    }

    #[test]
    fn test_standard_metadata_empty() {
        let mut standard_accumulator = StandardAccumulator::<u64>::new();
        let chunk = ConstantArray::new(Scalar::null_typed::<u64>(), 0).into_array();
        standard_accumulator.push_chunk(&chunk);

        let metadata_array = StructArray::try_from(
            Box::new(standard_accumulator)
                .into_array()
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata_array.len(), 1);
        assert_field_names(&metadata_array, &["null_count"]);
    }

    #[test]
    fn test_basic_metadata_schema() {
        let mut basic_accumulator = BasicAccumulator::new();
        let chunk = PrimitiveArray::from_nullable_vec(vec![Some(1u64)]).into_array();
        basic_accumulator.push_chunk(&chunk);

        let struct_array =
            StructArray::try_from(Box::new(basic_accumulator).into_array().unwrap().unwrap())
                .unwrap();
        assert_eq!(struct_array.len(), 1);
        assert_field_names(&struct_array, &["null_count"]);
    }
}
