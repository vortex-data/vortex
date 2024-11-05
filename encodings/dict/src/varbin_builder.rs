use std::hash::BuildHasher;

use hashbrown::HashTable;
use num_traits::AsPrimitive;
use vortex_array::aliases::hash_map::Hasher;
use vortex_array::array::builder::VarBinBuilder;
use vortex_array::array::{ConstantArray, PrimitiveArray, SparseArray, VarBinArray};
use vortex_array::validity::Validity;
use vortex_array::{ArrayDType, IntoArray};
use vortex_dtype::{DType, NativePType};
use vortex_error::VortexResult;
use vortex_scalar::ScalarValue;

use crate::{DictArray, NULL_CODE};

pub struct VarBinDictionaryBuilder {
    values: VarBinBuilder<i32>,
    codes: Vec<u64>,
    hasher: Hasher,
    lookup: HashTable<usize>,
}

impl Default for VarBinDictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VarBinDictionaryBuilder {
    pub fn new() -> Self {
        Self {
            values: VarBinBuilder::new(),
            codes: Vec::new(),
            hasher: Hasher::default(),
            lookup: HashTable::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: VarBinBuilder::new(),
            codes: Vec::with_capacity(capacity),
            hasher: Hasher::default(),
            lookup: HashTable::new(),
        }
    }

    #[inline]
    fn get_or_insert_key<V: AsRef<[u8]>>(&mut self, value: V) -> u64 {
        let byte_ref = value.as_ref();
        let value_hash = self.hasher.hash_one(byte_ref);
        *self
            .lookup
            .entry(
                value_hash,
                |idx| byte_ref == lookup_bytes(&self.values, *idx),
                |idx| self.hasher.hash_one(lookup_bytes(&self.values, *idx)),
            )
            .or_insert_with(|| {
                let idx = self.values.len();
                self.values.push_value(byte_ref);
                idx
            })
            .get() as u64
    }

    #[inline]
    pub fn append_value<T: AsRef<[u8]>>(&mut self, value: T) {
        let key = self.get_or_insert_key(value);
        self.codes.push(key);
    }

    /// Return built (codes, values) as tuple of PrimitiveArrays
    pub fn into_parts(self, dtype: DType) -> VortexResult<(PrimitiveArray, VarBinArray)> {
        Ok((PrimitiveArray::from(self.codes), self.values.finish(dtype)))
    }

    pub fn finish(self, dtype: DType) -> VortexResult<DictArray> {
        let (codes, values) = self.into_parts(dtype)?;
        DictArray::try_new(codes.into_array(), values.into_array())
    }
}

fn lookup_bytes<O: NativePType + AsPrimitive<usize>>(
    builder: &VarBinBuilder<O>,
    idx: usize,
) -> &[u8] {
    let begin: usize = builder.offsets_slice()[idx].as_();
    let end: usize = builder.offsets_slice()[idx + 1].as_();
    &builder.data_slice()[begin..end]
}

pub struct NullableVarBinDictionaryBuilder {
    builder: VarBinDictionaryBuilder,
}

impl Default for NullableVarBinDictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NullableVarBinDictionaryBuilder {
    pub fn new() -> Self {
        let mut builder = VarBinDictionaryBuilder::new();
        builder.values.push_null();
        Self { builder }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = VarBinDictionaryBuilder::with_capacity(capacity);
        builder.values.push_null();
        Self { builder }
    }

    #[inline]
    pub fn append<T: AsRef<[u8]>>(&mut self, value: Option<T>) {
        match value {
            None => self.builder.codes.push(NULL_CODE),
            Some(v) => self.append_value(v),
        }
    }

    #[inline]
    pub fn append_value<T: AsRef<[u8]>>(&mut self, value: T) {
        self.builder.append_value(value)
    }

    pub fn into_parts(self, dtype: DType) -> VortexResult<(PrimitiveArray, VarBinArray)> {
        let (codes, values) = self.builder.into_parts(dtype)?;
        let n_values = values.len();
        Ok((
            codes,
            VarBinArray::try_new(
                values.offsets(),
                values.bytes(),
                values.dtype().clone(),
                Validity::Array(
                    SparseArray::try_new(
                        ConstantArray::new(0u64, 1).into_array(),
                        ConstantArray::new(false, 1).into_array(),
                        n_values,
                        ScalarValue::from(true),
                    )?
                    .into_array(),
                ),
            )?,
        ))
    }

    pub fn finish(self, dtype: DType) -> VortexResult<DictArray> {
        let (codes, values) = self.into_parts(dtype)?;
        DictArray::try_new(codes.into_array(), values.into_array())
    }
}
