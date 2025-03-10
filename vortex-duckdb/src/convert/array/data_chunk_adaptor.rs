use duckdb::core::{ArrayVector, DataChunkHandle, FlatVector, ListVector, StructVector};
use duckdb::vtab::arrow::WritableVector;
use vortex_dtype::FieldNames;

pub struct DataChunkHandleSlice<'a> {
    chunk: &'a mut DataChunkHandle,
    column_index: usize,
}

/// A wrapper around a [`DataChunkHandle`] with extra info to create a vortex array
pub struct NamedDataChunk<'a> {
    pub chunk: &'a DataChunkHandle,
    pub nullable: Option<&'a [bool]>,
    pub names: Option<FieldNames>,
}

/// Since duckdb vectors only have a capacity, not a size this wrapper exists to allow the creation
/// of a vortex array from a duckdb vector.
/// Nullability is also included since the duckdb doesn't have this info its on the table.
pub struct SizedFlatVector {
    pub vector: FlatVector,
    pub nullable: bool,
    pub len: usize,
}

impl<'a> DataChunkHandleSlice<'a> {
    pub fn new(chunk: &'a mut DataChunkHandle, column_index: usize) -> Self {
        Self {
            chunk,
            column_index,
        }
    }
}

impl WritableVector for DataChunkHandleSlice<'_> {
    fn array_vector(&mut self) -> ArrayVector {
        self.chunk.array_vector(self.column_index)
    }

    fn flat_vector(&mut self) -> FlatVector {
        self.chunk.flat_vector(self.column_index)
    }

    fn struct_vector(&mut self) -> StructVector {
        self.chunk.struct_vector(self.column_index)
    }

    fn list_vector(&mut self) -> ListVector {
        self.chunk.list_vector(self.column_index)
    }
}

impl<'a> NamedDataChunk<'a> {
    pub fn from_chunk(chunk: &'a DataChunkHandle) -> Self {
        Self {
            chunk,
            nullable: None,
            names: None,
        }
    }

    pub fn named_chunk(chunk: &'a DataChunkHandle, names: FieldNames) -> Self {
        Self {
            chunk,
            nullable: None,
            names: Some(names),
        }
    }

    pub fn new(chunk: &'a DataChunkHandle, nullable: &'a [bool], names: FieldNames) -> Self {
        Self {
            chunk,
            nullable: Some(nullable),
            names: Some(names),
        }
    }
}
