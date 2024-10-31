//! Metadata accumulators track the per-chunk-of-a-column metadata, layout locations, and row counts.

use std::collections::VecDeque;
use std::mem;

use vortex::array::{BoolArray, NullArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex::stats::{ArrayStatistics as _, Stat};
use vortex::validity::Validity;
use vortex::{Array, IntoArray as _};
use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::{
    fieldnames_from_strings, match_each_native_ptype, DType, NativePType, Nullability,
};
use vortex_error::{vortex_bail, VortexError, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use super::layouts::Layout;
use crate::stream_writer::ByteRange;

pub fn new_metadata_accumulator(hint: usize, dtype: &DType) -> Box<dyn MetadataAccumulator> {
    match dtype {
        DType::Null => Box::new(ExtremaAccumulator::<()>::new(hint, into_null_array)),
        DType::Bool(..) => Box::new(ExtremaAccumulator::<bool>::new(hint, into_bool_array)),
        DType::Primitive(ptype, ..) => {
            match_each_native_ptype!(ptype, |$P| {
                Box::new(ExtremaAccumulator::<$P>::new(hint, into_primitive_array::<$P>))
            })
        }
        DType::Utf8(..) => Box::new(ExtremaAccumulator::<BufferString>::new(
            hint,
            into_utf8_array,
        )),
        DType::Binary(..) => Box::new(ExtremaAccumulator::<Buffer>::new(hint, into_binary_array)),
        DType::Struct(..) => Box::new(BasicAccumulator::new(hint)),
        DType::List(..) => Box::new(BasicAccumulator::new(hint)),
        DType::Extension(..) => Box::new(BasicAccumulator::new(hint)),
    }
}

pub trait MetadataAccumulator {
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()>;

    fn push_batch_byte_offsets(&mut self, batch_byte_offsets: Vec<u64>);

    fn into_layouts_and_metadata(self: Box<Self>) -> VortexResult<(VecDeque<Layout>, StructArray)>;
}

struct ExtremaAccumulator<T> {
    minima: Vec<Option<T>>,
    maxima: Vec<Option<T>>,
    to_array: fn(Vec<Option<T>>) -> Array,
    basic_metadata: BasicAccumulator,
}

impl<T> ExtremaAccumulator<T> {
    fn new(size_hint: usize, to_array: fn(Vec<Option<T>>) -> Array) -> Self {
        Self {
            minima: Vec::with_capacity(size_hint),
            maxima: Vec::with_capacity(size_hint),
            to_array,
            basic_metadata: BasicAccumulator::new(size_hint),
        }
    }
}

impl<T> MetadataAccumulator for ExtremaAccumulator<T>
where
    T: TryFrom<Scalar, Error = VortexError>,
{
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()> {
        self.minima.push(
            array
                .statistics()
                .compute(Stat::Min)
                .map(T::try_from)
                .transpose()?,
        );
        self.maxima.push(
            array
                .statistics()
                .compute(Stat::Max)
                .map(T::try_from)
                .transpose()?,
        );

        self.basic_metadata.push_chunk(array)
    }

    fn push_batch_byte_offsets(&mut self, batch_byte_offsets: Vec<u64>) {
        self.basic_metadata
            .push_batch_byte_offsets(batch_byte_offsets);
    }

    fn into_layouts_and_metadata(
        mut self: Box<Self>,
    ) -> VortexResult<(VecDeque<Layout>, StructArray)> {
        let (chunks, mut names, mut fields) =
            self.basic_metadata.into_layouts_and_metadata_parts()?;

        if self.minima.iter().any(Option::is_some) {
            names.push("min".into());
            fields.push((self.to_array)(mem::take(&mut self.minima)));
        }

        if self.maxima.iter().any(Option::is_some) {
            names.push("max".into());
            fields.push((self.to_array)(mem::take(&mut self.maxima)));
        }

        let n_chunks = chunks.len();
        let names = fieldnames_from_strings(names);
        Ok((
            chunks,
            StructArray::try_new(names, fields, n_chunks, Validity::NonNullable)?,
        ))
    }
}

struct BasicAccumulator {
    row_offsets: Vec<u64>,
    batch_byte_offsets: Vec<Vec<u64>>,
    null_counts: Vec<Option<u64>>,
    true_counts: Vec<Option<u64>>,
}

impl BasicAccumulator {
    pub fn new(size_hint: usize) -> Self {
        let mut row_offsets = Vec::with_capacity(size_hint + 1);
        row_offsets.push(0);
        Self {
            row_offsets,
            batch_byte_offsets: Vec::new(),
            null_counts: Vec::with_capacity(size_hint),
            true_counts: Vec::with_capacity(size_hint),
        }
    }

    fn n_rows_written(&self) -> u64 {
        *self
            .row_offsets
            .last()
            .vortex_expect("row offsets cannot be empty by construction")
    }
}

impl MetadataAccumulator for BasicAccumulator {
    fn push_chunk(&mut self, array: &Array) -> VortexResult<()> {
        self.row_offsets
            .push(self.n_rows_written() + array.len() as u64);

        self.null_counts.push(
            array
                .statistics()
                .compute(Stat::NullCount)
                .map(u64::try_from)
                .transpose()?,
        );

        self.true_counts.push(
            array
                .statistics()
                .compute(Stat::TrueCount)
                .map(u64::try_from)
                .transpose()?,
        );

        Ok(())
    }

    fn push_batch_byte_offsets(&mut self, batch_byte_offsets: Vec<u64>) {
        self.batch_byte_offsets.push(batch_byte_offsets);
    }

    fn into_layouts_and_metadata(self: Box<Self>) -> VortexResult<(VecDeque<Layout>, StructArray)> {
        let (chunks, names, fields) = self.into_layouts_and_metadata_parts()?;
        let n_chunks = chunks.len();
        let names = fieldnames_from_strings(names);
        Ok((
            chunks,
            StructArray::try_new(names, fields, n_chunks, Validity::NonNullable)?,
        ))
    }
}

impl BasicAccumulator {
    fn into_layouts_and_metadata_parts(
        mut self,
    ) -> VortexResult<(VecDeque<Layout>, Vec<String>, Vec<Array>)> {
        // we don't need the last row offset; that's just the total number of rows
        let length = self.row_offsets.len() - 1;
        self.row_offsets.truncate(length);

        let chunks: VecDeque<Layout> = self
            .batch_byte_offsets
            .iter()
            .flat_map(|byte_offsets| {
                byte_offsets
                    .iter()
                    .zip(byte_offsets.iter().skip(1))
                    .map(|(begin, end)| Layout::flat(ByteRange::new(*begin, *end)))
            })
            .collect();

        if chunks.len() != self.row_offsets.len() {
            vortex_bail!(
                "Expected {} chunks based on row offsets, found {} based on byte offsets",
                self.row_offsets.len(),
                chunks.len()
            );
        }

        let mut names: Vec<String> = vec!["row_offset".into()];
        let mut fields = vec![mem::take(&mut self.row_offsets).into_array()];

        if self.null_counts.iter().any(Option::is_some) {
            names.push("null_count".into());
            fields.push(
                PrimitiveArray::from_nullable_vec(mem::take(&mut self.null_counts)).into_array(),
            );
        }

        if self.true_counts.iter().any(Option::is_some) {
            names.push("true_count".into());
            fields.push(
                PrimitiveArray::from_nullable_vec(mem::take(&mut self.true_counts)).into_array(),
            );
        }

        Ok((chunks, names, fields))
    }
}

fn into_null_array(vec: Vec<Option<()>>) -> Array {
    NullArray::new(vec.len()).into_array()
}

fn into_bool_array(vec: Vec<Option<bool>>) -> Array {
    BoolArray::from_iter(vec).into_array()
}

fn into_primitive_array<P>(vec: Vec<Option<P>>) -> Array
where
    P: NativePType,
    P: TryFrom<Scalar, Error = VortexError>,
    P: 'static,
{
    PrimitiveArray::from_nullable_vec(vec).into_array()
}

fn into_utf8_array(x: Vec<Option<BufferString>>) -> Array {
    VarBinViewArray::from_iter(x, DType::Utf8(Nullability::Nullable)).into_array()
}

fn into_binary_array(x: Vec<Option<Buffer>>) -> Array {
    VarBinViewArray::from_iter(x, DType::Binary(Nullability::Nullable)).into_array()
}
