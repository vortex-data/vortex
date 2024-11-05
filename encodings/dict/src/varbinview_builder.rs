use std::hash::BuildHasher;

use arrow_array::builder::{ArrayBuilder, GenericByteViewBuilder};
use arrow_array::types::BinaryViewType;
use vortex_array::aliases::hash_map::{HashTable, Hasher};
use vortex_array::array::{ConstantArray, PrimitiveArray, SparseArray, VarBinViewArray};
use vortex_array::arrow::FromArrowArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayDType, IntoArray};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::ScalarValue;

use crate::{DictArray, NULL_CODE};

pub struct VarBinViewDictionaryBuilder {
    values: GenericByteViewBuilder<BinaryViewType>,
    codes: Vec<u64>,
    hasher: Hasher,
    lookup: HashTable<usize>,
}

impl Default for VarBinViewDictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VarBinViewDictionaryBuilder {
    pub fn new() -> Self {
        Self {
            values: GenericByteViewBuilder::new(),
            codes: Vec::new(),
            hasher: Hasher::default(),
            lookup: HashTable::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: GenericByteViewBuilder::new(),
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
                |idx| byte_ref == self.values.get_value(*idx),
                |idx| self.hasher.hash_one(self.values.get_value(*idx)),
            )
            .or_insert_with(|| {
                let idx = self.values.len();
                self.values.append_value(byte_ref);
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
    pub fn into_parts(mut self, dtype: DType) -> VortexResult<(PrimitiveArray, VarBinViewArray)> {
        let array = Array::from_arrow(&self.values.finish(), dtype.is_nullable());
        let varbinview = VarBinViewArray::try_from(array)?;

        let casted_array = match dtype {
            DType::Utf8(n) => VarBinViewArray::try_new(
                varbinview.views(),
                varbinview.buffers().collect(),
                DType::Utf8(n),
                varbinview.validity(),
            )?,
            DType::Binary(_) => varbinview,
            _ => vortex_bail!("DType can only be Utf8 or Binary"),
        };

        Ok((PrimitiveArray::from(self.codes), casted_array))
    }

    pub fn finish(self, dtype: DType) -> VortexResult<DictArray> {
        let (codes, values) = self.into_parts(dtype)?;
        DictArray::try_new(codes.into_array(), values.into_array())
    }
}

pub struct NullableVarBinViewDictionaryBuilder {
    builder: VarBinViewDictionaryBuilder,
}

impl Default for NullableVarBinViewDictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NullableVarBinViewDictionaryBuilder {
    pub fn new() -> Self {
        let mut builder = VarBinViewDictionaryBuilder::new();
        builder.values.append_null();
        Self { builder }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = VarBinViewDictionaryBuilder::with_capacity(capacity);
        builder.values.append_null();
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

    pub fn into_parts(self, dtype: DType) -> VortexResult<(PrimitiveArray, VarBinViewArray)> {
        let (codes, values) = self.builder.into_parts(dtype)?;
        let n_values = values.len();
        Ok((
            codes,
            VarBinViewArray::try_new(
                values.views(),
                values.buffers().collect(),
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
