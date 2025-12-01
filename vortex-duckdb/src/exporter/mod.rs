// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod all_invalid;
mod bool;
mod cache;
mod constant;
mod decimal;
mod dict;
mod fixed_size_list;
mod list;
mod primitive;
mod run_end;
mod sequence;
mod struct_;
mod temporal;
mod validity;
mod varbinview;

use std::sync::Arc;

use bitvec::prelude::Lsb0;
use bitvec::view::BitView;
pub use cache::ConversionCache;
pub use decimal::precision_to_duckdb_storage_size;
use itertools::Itertools;
use vortex::array::Array;
use vortex::array::Canonical;
use vortex::array::ToCanonical;
use vortex::array::arrays::ConstantVTable;
use vortex::array::arrays::DictVTable;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::iter::ArrayIterator;
use vortex::dtype::datetime::is_temporal_ext_type;
use vortex::encodings::runend::RunEndVTable;
use vortex::encodings::sequence::SequenceVTable;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::cpp::DUCKDB_TYPE;
use crate::duckdb::DUCKDB_STANDARD_VECTOR_SIZE;
use crate::duckdb::DataChunk;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;
use crate::duckdb::Vector;

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
            // In the case of a projection pushdown with zero columns duckdb will ask us for the
            // `EMPTY_COLUMN_IDX`, which we define as a bool column, we can leave the vector as
            // uninitialized and just return a DataChunk with the correct length.
            // One place no fields can occur is in count(*) queries.
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

fn new_array_exporter(
    array: &dyn Array,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    new_array_exporter_with_flatten(array, cache, false)
}

/// Create a DuckDB exporter for the given Vortex array.
fn new_array_exporter_with_flatten(
    array: &dyn Array,
    cache: &ConversionCache,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if let Some(array) = array.as_opt::<ConstantVTable>() {
        return constant::new_exporter(array);
    }

    if let Some(array) = array.as_opt::<RunEndVTable>() {
        return run_end::new_exporter(array, cache);
    }

    if let Some(array) = array.as_opt::<DictVTable>() {
        return dict::new_exporter_with_flatten(array, cache, flatten);
    }

    if let Some(array) = array.as_opt::<SequenceVTable>() {
        return sequence::new_exporter(array);
    }

    // Otherwise, we fall back to canonical
    match array.to_canonical() {
        Canonical::Null(_) => Ok(all_invalid::new_exporter(
            array.len(),
            &LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_SQLNULL),
        )),
        Canonical::Bool(array) => bool::new_exporter(&array),
        Canonical::Primitive(array) => primitive::new_exporter(&array),
        Canonical::Decimal(array) => decimal::new_exporter(&array),
        Canonical::Struct(array) => struct_::new_exporter(&array, cache),
        Canonical::List(array) => list::new_exporter(&array, cache),
        Canonical::FixedSizeList(array) => fixed_size_list::new_exporter(&array, cache),
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

impl Vector {
    unsafe fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        match mask {
            Mask::AllTrue(_) => {
                // We only need to blank out validity if there is already a slice allocated.
                // SAFETY: Caller guaranteees this.
                unsafe { self.set_all_true_validity(len) }
                false
            }
            Mask::AllFalse(_) => {
                // SAFETY: Caller guaranteees this.
                self.set_all_false_validity();
                true
            }
            Mask::Values(arr) => {
                let true_count = arr.bit_buffer().true_count();
                if true_count == len {
                    unsafe { self.set_all_true_validity(len) }
                } else if true_count == 0 {
                    self.set_all_false_validity()
                } else {
                    let source = arr.bit_buffer().inner().as_slice();
                    copy_from_slice(
                        unsafe { self.ensure_validity_slice(len) },
                        source,
                        offset,
                        len,
                    );
                }

                true_count == 0
            }
        }
    }

    unsafe fn set_all_true_validity(&mut self, len: usize) {
        if let Some(validity) = unsafe { self.validity_bitslice_mut(len) } {
            validity.fill(true);
        }
    }

    fn set_all_false_validity(&mut self) {
        self.reference_value(&Value::null(&self.logical_type()));
    }
}

/// Copy the sliced bits from source into target.
///
/// Offset and length are a _bit_ offset and a _bit_ length into source.
///
/// `target.len()` must equal `len`.
fn copy_from_slice(target: &mut [u64], source: &[u8], offset: usize, len: usize) {
    let (start, middle, end) = unsafe { target.align_to_mut::<u8>() };
    assert!(start.is_empty());
    assert!(end.is_empty());
    let target = &mut middle.view_bits_mut::<Lsb0>()[..len];
    target.copy_from_bitslice(&source.view_bits()[offset..][..len]);
}

#[cfg(test)]
mod tests {
    use vortex::buffer::BitBuffer;
    use vortex::mask::Mask;

    use crate::cpp::DUCKDB_TYPE;
    use crate::duckdb::LogicalType;
    use crate::duckdb::Vector;
    use crate::exporter::copy_from_slice;

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
        let len = 10;

        let mask = Mask::AllFalse(len);
        let all_null = unsafe { vector.set_validity(&mask, 0, len) };

        assert!(all_null);

        vector.flatten(len as u64);

        for i in 0..10 {
            assert!(vector.row_is_null(i));
        }
    }

    #[test]
    fn test_set_validity_values_all_true() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let mask = Mask::from(BitBuffer::from(vec![true; 10]));

        let all_null = unsafe { vector.set_validity(&mask, 0, 10) };

        assert!(!all_null);

        // When all values are true, the mask may be optimized to AllTrue,
        // so validity_slice_mut may return None (no validity allocated)
        if let Some(validity) = unsafe { vector.validity_bitslice_mut(10) } {
            assert!(validity.iter().all(|v| *v));
        }
    }

    #[test]
    fn test_set_validity_values_all_false() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        const LEN: usize = 10;
        let bits = vec![false; LEN];
        let mask = Mask::from(BitBuffer::from(bits));

        let all_null = unsafe { vector.set_validity(&mask, 0, LEN) };

        assert!(all_null);

        vector.flatten(LEN as u64);
        for i in 0..10 {
            assert!(vector.row_is_null(i));
        }
    }

    #[test]
    fn test_set_validity_values_mixed() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let bits = vec![
            true, false, true, true, false, false, true, true, false, true,
        ];
        let mask = Mask::from(BitBuffer::from(bits.as_slice()));

        let all_null = unsafe { vector.set_validity(&mask, 0, 10) };

        assert!(!all_null);

        let validity = unsafe { vector.validity_bitslice_mut(10).unwrap() };
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
        let mask = Mask::from(BitBuffer::from(bits.as_slice()));

        let all_null = unsafe { vector.set_validity(&mask, 2, 8) };

        assert!(!all_null);

        let validity = unsafe { vector.validity_bitslice_mut(8).unwrap() };
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
        let mask = Mask::from(BitBuffer::from(bits.as_slice()));

        let all_null = unsafe { vector.set_validity(&mask, 3, 5) };

        assert!(!all_null);

        let validity = unsafe { vector.validity_bitslice_mut(5).unwrap() };
        for i in 0..5 {
            assert_eq!(validity[i], bits[i + 3]);
        }
    }

    #[test]
    fn test_set_validity_values_64bit_alignment() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        let mut vector = Vector::with_capacity(logical_type, 100);

        let bits = (0..70).map(|i| i % 3 == 0).collect::<Vec<_>>();
        let mask = Mask::from(BitBuffer::from(bits.as_slice()));

        let all_null = unsafe { vector.set_validity(&mask, 5, 60) };

        assert!(!all_null);

        let validity = unsafe { vector.validity_bitslice_mut(60).unwrap() };
        for i in 0..60 {
            assert_eq!(validity[i], bits[i + 5]);
        }
    }

    #[test]
    fn test_copy_from_slice_empty_to_empty() {
        let target = &mut [];
        let source = Vec::<u8>::new();
        copy_from_slice(target, &source, 0, 0);
    }

    #[test]
    fn test_copy_from_slice_64_to_empty() {
        let target = &mut [];
        let source = [1u8, 2, 3, 50, 51, 52, 100, 101];
        copy_from_slice(target, &source, 0, 0);
        copy_from_slice(target, &source, 5, 0);
        copy_from_slice(target, &source, 8, 0);
    }

    #[test]
    fn test_copy_from_slice_64_to_64() {
        let mut target = vec![0u64];
        let source = [1u8, 2, 3, 50, 51, 52, 100, 101];
        copy_from_slice(&mut target, &source, 0, 64);
        assert_eq!(
            target[0], 0x65_64_34_33_32_03_02_01_u64,
            "{:#08x} == {:#08x}",
            target[0], 0x65_64_34_33_32_03_02_01_u64,
        );
    }

    #[test]
    fn test_copy_from_slice_80_to_0() {
        let target = &mut [];
        let source = [1u8, 2, 3, 50, 51, 52, 100, 101, 254, 255];
        copy_from_slice(target, &source, 0, 0);
        copy_from_slice(target, &source, 8, 0);
        copy_from_slice(target, &source, 10, 0);
    }

    #[test]
    fn test_copy_from_slice_80_to_64_case_1() {
        let mut target = [0u64];
        let source = [1u8, 2, 3, 50, 51, 52, 100, 101, 254, 255];
        copy_from_slice(&mut target, &source, 16, 64);
        assert_eq!(
            target[0], 0xff_fe_65_64_34_33_32_03_u64,
            "{:#08x} == {:#08x}",
            target[0], 0xff_fe_65_64_34_33_32_03_u64,
        );
    }

    #[test]
    fn test_copy_from_slice_80_to_64_case_2() {
        let mut target = [0u64];
        let source = [1u8, 2, 3, 50, 51, 52, 100, 101, 254, 255];
        copy_from_slice(&mut target, &source, 8, 64);
        assert_eq!(
            target[0], 0xfe_65_64_34_33_32_03_02_u64,
            "{:#08x} == {:#08x}",
            target[0], 0xfe_65_64_34_33_32_03_02_u64,
        );
    }

    #[test]
    fn test_copy_from_slice_80_to_64_case_3() {
        let mut target = [0u64];
        let source = [1u8, 2, 3, 50, 51, 52, 100, 101, 254, 255];
        copy_from_slice(&mut target, &source, 0, 64);
        assert_eq!(
            target[0], 0x65_64_34_33_32_03_02_01_u64,
            "{:#08x} == {:#08x}",
            target[0], 0x65_64_34_33_32_03_02_01_u64,
        );
    }

    #[test]
    fn test_copy_from_slice_80_to_64_case_4() {
        let mut target = [0u64];
        let source = [1u8, 2, 3, 50, 51, 52, 100, 101, 254, 255];
        copy_from_slice(&mut target, &source, 10, 64);
        assert_eq!(
            target[0],
            0xff_99_59_0d_0c_cc_80_c0_u64, // Python: hex(0xff_fe_65_64_34_33_32_03_02 >> 2), then remove the high two hexits
            "{:#08x} == {:#08x}",
            target[0],
            0xff_99_59_0d_0c_cc_80_c0_u64
        );
    }

    #[test]
    fn test_copy_from_slice_248_to_128_middle_non_empty() {
        let mut target = [0u64, 0u64];
        let source: [u8; 31] = [
            0x01, 0x02, 0x03, 0x04, 0xff, 0xfe, 0xfd, 0xfc, 0x05, 0x06, 0x07, 0x08, 0xfc, 0xfb,
            0xfa, 0xf9, 0x01, 0x02, 0x03, 0x04, 0xff, 0xfe, 0xfd, 0xfc, 0x05, 0x06, 0x07, 0x08,
            0xfc, 0xfb, 0xfa,
        ];
        // In a span of 248 bits (31 bytes) there should be at least one 8-byte aligned span.
        let (_, middle, _) = unsafe { source.align_to::<u64>() };
        assert!(!middle.is_empty());

        copy_from_slice(&mut target, &source, 0, 128);
        assert_eq!(
            target[0], 0xfc_fd_fe_ff_04_03_02_01_u64,
            "{:#08x} == {:#08x}",
            target[0], 0xfc_fd_fe_ff_04_03_02_01_u64,
        );
        assert_eq!(
            target[1], 0xf9_fa_fb_fc_08_07_06_05_u64,
            "{:#08x} == {:#08x}",
            target[1], 0xf9_fa_fb_fc_08_07_06_05_u64,
        );

        copy_from_slice(&mut target, &source, 8, 128);
        assert_eq!(
            target[0], 0x05_fc_fd_fe_ff_04_03_02_u64,
            "{:#08x} == {:#08x}",
            target[0], 0x05_fc_fd_fe_ff_04_03_02_u64,
        );
        assert_eq!(
            target[1], 0x01_f9_fa_fb_fc_08_07_06_u64,
            "{:#08x} == {:#08x}",
            target[1], 0x01_f9_fa_fb_fc_08_07_06_u64,
        );

        copy_from_slice(&mut target, &source, 8 * 8, 128);
        assert_eq!(
            target[0], 0xf9_fa_fb_fc_08_07_06_05_u64,
            "{:#08x} == {:#08x}",
            target[0], 0xf9_fa_fb_fc_08_07_06_05_u64,
        );
        assert_eq!(
            target[1], 0xfc_fd_fe_ff_04_03_02_01_u64,
            "{:#08x} == {:#08x}",
            target[1], 0xfc_fd_fe_ff_04_03_02_01_u64,
        );

        copy_from_slice(&mut target, &source, 8 * 12, 128);
        assert_eq!(
            target[0], 0x04_03_02_01_f9_fa_fb_fc_u64,
            "{:#08x} == {:#08x}",
            target[0], 0x04_03_02_01_f9_fa_fb_fc_u64,
        );
        assert_eq!(
            target[1], 0x08_07_06_05_fc_fd_fe_ff_u64,
            "{:#08x} == {:#08x}",
            target[1], 0x08_07_06_05_fc_fd_fe_ff_u64,
        );

        copy_from_slice(&mut target, &source, 8 * 12 + 4, 128);
        // Find the 12th byte, skip the first hexit, take the next 32 hexits (i.e. 16 bytesor 128
        // bits).
        assert_eq!(
            target[0], 0xf0_40_30_20_1f_9f_af_bf_u64,
            "{:#08x} == {:#08x}",
            target[0], 0xf0_40_30_20_1f_9f_af_bf_u64,
        );
        assert_eq!(
            target[1], 0xc0_80_70_60_5f_cf_df_ef_u64,
            "{:#08x} == {:#08x}",
            target[1], 0xc0_80_70_60_5f_cf_df_ef_u64,
        );

        // Take the above and shift one bit towards the right-hand-side.
        copy_from_slice(&mut target, &source, 8 * 12 + 4 + 1, 128);
        assert_eq!(
            target[0], 0xf8_20_18_10_0f_cf_d7_df_u64,
            "{:#08x} == {:#08x}",
            target[0], 0xf8_20_18_10_0f_cf_d7_df_u64,
        );
        assert_eq!(
            target[1], 0xe0_40_38_30_2f_e7_ef_f7_u64,
            "{:#08x} == {:#08x}",
            target[1], 0xe0_40_38_30_2f_e7_ef_f7_u64,
        );
    }
}
