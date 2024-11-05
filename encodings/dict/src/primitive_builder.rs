use std::hash::{Hash, Hasher};

use num_traits::AsPrimitive;
use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_array::array::{ConstantArray, PrimitiveArray, SparseArray};
use vortex_array::validity::Validity;
use vortex_array::IntoArray;
use vortex_dtype::{NativePType, ToBytes};
use vortex_error::VortexResult;
use vortex_scalar::ScalarValue;

use crate::DictArray;

/// Statically assigned code for a null value.
pub const NULL_CODE: u64 = 0;

#[derive(Debug)]
struct Value<T>(T);

impl<T: ToBytes> Hash for Value<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_le_bytes().hash(state)
    }
}

impl<T: ToBytes> PartialEq<Self> for Value<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_le_bytes().eq(other.0.to_le_bytes())
    }
}

impl<T: ToBytes> Eq for Value<T> {}

pub struct PrimitiveDictionaryBuilder<T> {
    values: Vec<T>,
    codes: Vec<u64>,
    lookup: HashMap<Value<T>, u64>,
}

impl<T: NativePType> Default for PrimitiveDictionaryBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativePType> PrimitiveDictionaryBuilder<T> {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            codes: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::new(),
            codes: Vec::with_capacity(capacity),
            lookup: HashMap::new(),
        }
    }

    #[inline]
    fn get_or_insert_key(&mut self, value: T) -> u64 {
        match self.lookup.entry(Value(value)) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(vac) => {
                let next_code = self.values.len() as u64;
                vac.insert(next_code.as_());
                self.values.push(value);
                next_code
            }
        }
    }

    #[inline]
    pub fn append_value(&mut self, value: T) {
        let key = self.get_or_insert_key(value);
        self.codes.push(key);
    }

    /// Return built (codes, values) as tuple of PrimitiveArrays
    pub fn into_parts(self) -> (Vec<u64>, Vec<T>) {
        (self.codes, self.values)
    }

    pub fn finish(self) -> VortexResult<DictArray> {
        let (codes, values) = self.into_parts();
        DictArray::try_new(
            PrimitiveArray::from(codes).into_array(),
            PrimitiveArray::from(values).into_array(),
        )
    }
}

pub struct NullablePrimitiveDictionaryBuilder<T> {
    builder: PrimitiveDictionaryBuilder<T>,
}

impl<T: NativePType> Default for NullablePrimitiveDictionaryBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativePType> NullablePrimitiveDictionaryBuilder<T> {
    pub fn new() -> Self {
        let mut builder = PrimitiveDictionaryBuilder::new();
        builder.values.push(T::zero());
        Self { builder }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = PrimitiveDictionaryBuilder::with_capacity(capacity);
        builder.values.push(T::zero());
        Self { builder }
    }

    #[inline]
    pub fn append(&mut self, value: Option<T>) {
        match value {
            None => self.builder.codes.push(NULL_CODE),
            Some(v) => self.append_value(v),
        }
    }

    #[inline]
    pub fn append_value(&mut self, value: T) {
        self.builder.append_value(value)
    }

    pub fn into_parts(self) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
        let (codes, values) = self.builder.into_parts();
        let n_values = values.len();
        Ok((
            PrimitiveArray::from(codes),
            PrimitiveArray::from_vec(
                values,
                Validity::Array(
                    SparseArray::try_new(
                        ConstantArray::new(0u64, 1).into_array(),
                        ConstantArray::new(false, 1).into_array(),
                        n_values,
                        ScalarValue::from(true),
                    )?
                    .into_array(),
                ),
            ),
        ))
    }

    pub fn finish(self) -> VortexResult<DictArray> {
        let (codes, values) = self.into_parts()?;
        DictArray::try_new(codes.into_array(), values.into_array())
    }
}
