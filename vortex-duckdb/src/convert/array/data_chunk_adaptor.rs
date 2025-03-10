use duckdb::core::{ArrayVector, DataChunkHandle, FlatVector, ListVector, StructVector};
use duckdb::vtab::arrow::WritableVector;

pub struct DataChunkHandleSlice<'a> {
    chunk: &'a mut DataChunkHandle,
    column_index: usize,
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
