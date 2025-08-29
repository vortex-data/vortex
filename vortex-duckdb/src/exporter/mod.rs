// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod cache;
mod constant;
mod decimal;
mod dict;
mod list;
mod primitive;
mod run_end;
mod sequence;
mod temporal;
mod varbinview;

use std::sync::Arc;

use bitvec::prelude::Lsb0;
use bitvec::view::BitView;
pub use cache::ConversionCache;
pub use decimal::precision_to_duckdb_storage_size;
use itertools::Itertools;
use vortex::arrays::{ConstantVTable, StructArray, TemporalArray};
use vortex::dtype::datetime::is_temporal_ext_type;
use vortex::encodings::dict::DictVTable;
use vortex::encodings::runend::RunEndVTable;
use vortex::encodings::sequence::SequenceVTable;
use vortex::error::{VortexExpect, VortexResult, vortex_bail};
use vortex::iter::ArrayIterator;
use vortex::mask::Mask;
use vortex::{Array, Canonical, ToCanonical};

use crate::duckdb::{DUCKDB_STANDARD_VECTOR_SIZE, DataChunk, Vector};

/// DuckDB exporter for an [`ArrayIterator`], sharing state and caches.
pub struct ArrayIteratorExporter {
    iter: Box<dyn ArrayIterator>,
    cache: Arc<ConversionCache>,
    array_exporter: Option<ArrayExporter>,
}

impl ArrayIteratorExporter {
    pub fn new(iter: Box<dyn ArrayIterator>, id: u64) -> Self {
        Self {
            iter,
            cache: Arc::new(ConversionCache::new(id)),
            array_exporter: None,
        }
    }

    /// Returns `true` if a chunk was exported, `false` if all data has been exported.
    pub fn export(&mut self, chunk: &mut DataChunk) -> VortexResult<bool> {
        loop {
            if self.array_exporter.is_none() {
                if let Some(array) = self.iter.next() {
                    // Create a new array exporter for the current array.
                    let array = array?.to_struct();
                    self.array_exporter = Some(ArrayExporter::try_new(&array, &self.cache)?);
                } else {
                    // No more arrays to export.
                    return Ok(false);
                }
            }

            if self
                .array_exporter
                .as_mut()
                .vortex_expect("must be present")
                .export(chunk)?
            {
                return Ok(true);
            } else {
                // This exporter is done, so we throw it away and loop.
                self.array_exporter = None;
            }
        }
    }
}

pub struct ArrayExporter {
    fields: Vec<Box<dyn ColumnExporter>>,
    array_len: usize,
    remaining: usize,
}

impl ArrayExporter {
    pub fn try_new(array: &StructArray, cache: &ConversionCache) -> VortexResult<Self> {
        let fields = array
            .fields()
            .iter()
            .map(|field| new_array_exporter(field.as_ref(), cache))
            .try_collect()?;
        Ok(Self {
            fields,
            array_len: array.len(),
            remaining: array.len(),
        })
    }

    /// Export the data into the next chunk.
    ///
    /// Returns `true` if a chunk was exported, `false` if all rows have been exported.
    pub fn export(&mut self, chunk: &mut DataChunk) -> VortexResult<bool> {
        if self.remaining == 0 {
            return Ok(false);
        }

        if self.fields.is_empty() {
            // No fields can occur in e.g. count(*) queries. In these cases, we just need to
            // set the length of the chunk and return.
            chunk.set_len(self.remaining);
            self.remaining = 0;

            return Ok(true);
        }

        let chunk_len = DUCKDB_STANDARD_VECTOR_SIZE.min(self.remaining);
        let position = self.array_len - self.remaining;
        self.remaining -= chunk_len;
        chunk.set_len(chunk_len);

        for (i, field) in self.fields.iter_mut().enumerate() {
            let mut vector = chunk.get_vector(i);
            field.export(position, chunk_len, &mut vector)?;
        }

        Ok(true)
    }
}

/// Exporter for a single column of a DuckDB data chunk.
///
/// NOTE(ngates): we could actually convert this into a Vortex compute function that takes
///  the offset, len and `WritableVector` as options. Not sure what it should return though?
///  This would allow Vortex extension authors to plug into the DuckDB exporter system.
pub trait ColumnExporter {
    /// Export the given range of data from the Vortex array to the DuckDB vector.
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()>;
}

/// Create a DuckDB exporter for the given Vortex array.
fn new_array_exporter(
    array: &dyn Array,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if let Some(array) = array.as_opt::<ConstantVTable>() {
        return constant::new_exporter(array);
    }

    if let Some(array) = array.as_opt::<RunEndVTable>() {
        return run_end::new_exporter(array, cache);
    }

    if let Some(array) = array.as_opt::<DictVTable>() {
        return dict::new_exporter(array, cache);
    }

    if let Some(array) = array.as_opt::<SequenceVTable>() {
        return sequence::new_exporter(array);
    }

    // Otherwise, we fall back to canonical
    match array.to_canonical() {
        Canonical::Null(_) => todo!("no null exporter"),
        Canonical::Bool(array) => bool::new_exporter(&array),
        Canonical::Primitive(array) => primitive::new_exporter(&array),
        Canonical::Decimal(array) => decimal::new_exporter(&array),
        Canonical::Struct(_) => {
            // The Arrow exporter does not support struct arrays yet, so we bail out.
            vortex_bail!("Struct arrays are not supported in DuckDB export yet");
        }
        Canonical::List(array) => list::new_exporter(&array, cache),
        Canonical::FixedSizeList(_) => todo!("TODO(connor)[FixedSizeList]"),
        Canonical::VarBinView(array) => varbinview::new_exporter(&array),
        Canonical::Extension(ext) => {
            if is_temporal_ext_type(ext.id()) {
                let temporal_array =
                    TemporalArray::try_from(ext).vortex_expect("id is a temporal array");
                return temporal::new_exporter(&temporal_array);
            }
            todo!("no non-temporal extension exporter")
        }
    }
}

pub(crate) trait VectorExt {
    /// Returns true if *all* values within the offset -> len slice are null.
    /// Since we're iterating these values anyway, then it's cheaper for us to check it inline.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `len` is less than or equal to the capacity of this vector.
    unsafe fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool;
}

impl VectorExt for Vector {
    unsafe fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        match mask {
            Mask::AllTrue(_) => {
                // We only need to blank out validity if there is already a slice allocated.
                // SAFETY: Caller guaranteees this.
                if let Some(validity) = unsafe { self.validity_slice_mut(len) } {
                    validity.fill(true);
                }
                false
            }
            Mask::AllFalse(_) => {
                // SAFETY: Caller guaranteees this.
                unsafe { self.ensure_validity_slice(len) }.fill(false);
                true
            }
            Mask::Values(arr) => {
                let arr_bits: &[u64] = {
                    let byte_slice = arr.boolean_buffer().inner().as_slice();
                    unsafe {
                        std::slice::from_raw_parts(
                            byte_slice.as_ptr() as _,
                            byte_slice.len().div_ceil(size_of::<u64>()),
                        )
                    }
                };
                let sliced_bits = &arr_bits.view_bits::<Lsb0>()[offset..][..len];
                let true_count = sliced_bits.count_ones();
                if true_count == len {
                    if let Some(validity) = unsafe { self.validity_slice_mut(len) } {
                        validity.fill(true);
                    }
                } else if true_count == 0 {
                    unsafe { self.ensure_validity_slice(len) }.fill(false);
                } else {
                    // SAFETY: Caller guaranteees this.
                    let validity = unsafe { self.ensure_validity_slice(len) };
                    validity[..len].copy_from_bitslice(sliced_bits);
                }
                true_count == 0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::buffer::BooleanBuffer;
    use vortex::mask::Mask;

    use super::VectorExt;
    use crate::cpp::DUCKDB_TYPE;
    use crate::duckdb::{LogicalType, Vector};

    #[test]
    fn test_set_validity_all_true() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let mask = Mask::AllTrue(10);
        let all_null = unsafe { vector.set_validity(&mask, 0, 10) };

        assert!(!all_null);
    }

    #[test]
    fn test_set_validity_all_false() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let mask = Mask::AllFalse(10);
        let all_null = unsafe { vector.set_validity(&mask, 0, 10) };

        assert!(all_null);

        let validity = unsafe { vector.validity_slice_mut(10).unwrap() };
        assert!(validity.iter().all(|v| !v));
    }

    #[test]
    fn test_set_validity_values_all_true() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let bits = vec![true; 10];
        let buffer = BooleanBuffer::from(bits.as_slice());
        let mask = Mask::from(buffer);

        let all_null = unsafe { vector.set_validity(&mask, 0, 10) };

        assert!(!all_null);

        // When all values are true, the mask may be optimized to AllTrue,
        // so validity_slice_mut may return None (no validity allocated)
        if let Some(validity) = unsafe { vector.validity_slice_mut(10) } {
            assert!(validity.iter().all(|v| *v));
        }
    }

    #[test]
    fn test_set_validity_values_all_false() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let bits = vec![false; 10];
        let buffer = BooleanBuffer::from(bits.as_slice());
        let mask = Mask::from(buffer);

        let all_null = unsafe { vector.set_validity(&mask, 0, 10) };

        assert!(all_null);

        let validity = unsafe { vector.validity_slice_mut(10).unwrap() };
        assert!(validity.iter().all(|v| !v));
    }

    #[test]
    fn test_set_validity_values_mixed() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let bits = vec![
            true, false, true, true, false, false, true, true, false, true,
        ];
        let buffer = BooleanBuffer::from(bits.as_slice());
        let mask = Mask::from(buffer);

        let all_null = unsafe { vector.set_validity(&mask, 0, 10) };

        assert!(!all_null);

        let validity = unsafe { vector.validity_slice_mut(10).unwrap() };
        for (i, bit) in bits.iter().enumerate() {
            assert_eq!(validity[i], *bit);
        }
    }

    #[test]
    fn test_set_validity_values_with_offset() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let bits = vec![
            false, false, true, true, false, true, false, true, true, false, true, true, false,
        ];
        let buffer = BooleanBuffer::from(bits.as_slice());
        let mask = Mask::from(buffer);

        let all_null = unsafe { vector.set_validity(&mask, 2, 8) };

        assert!(!all_null);

        let validity = unsafe { vector.validity_slice_mut(8).unwrap() };
        for i in 0..8 {
            assert_eq!(validity[i], bits[i + 2]);
        }
    }

    #[test]
    fn test_set_validity_values_with_offset_and_smaller_len() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let bits = vec![
            true, false, true, true, false, false, true, true, false, true, true, true, false,
            true, false,
        ];
        let buffer = BooleanBuffer::from(bits.as_slice());
        let mask = Mask::from(buffer);

        let all_null = unsafe { vector.set_validity(&mask, 3, 5) };

        assert!(!all_null);

        let validity = unsafe { vector.validity_slice_mut(5).unwrap() };
        for i in 0..5 {
            assert_eq!(validity[i], bits[i + 3]);
        }
    }

    #[test]
    fn test_set_validity_values_64bit_alignment() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let bits = (0..70).map(|i| i % 3 == 0).collect::<Vec<_>>();

        let buffer = BooleanBuffer::from(bits.as_slice());
        let mask = Mask::from(buffer);

        let all_null = unsafe { vector.set_validity(&mask, 5, 60) };

        assert!(!all_null);

        let validity = unsafe { vector.validity_slice_mut(60).unwrap() };
        for i in 0..60 {
            assert_eq!(validity[i], bits[i + 5]);
        }
    }
}
