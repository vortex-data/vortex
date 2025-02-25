use std::hash::BuildHasher;

use arrow_buffer::NullBufferBuilder;
use num_traits::AsPrimitive;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::aliases::hash_map::{DefaultHashBuilder, HashTable, RandomState};
use vortex_array::arrays::{BinaryView, PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayExt, ArrayRef};
use vortex_buffer::{BufferMut, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail};

use crate::builders::DictEncoder;

/// Dictionary encode varbin array. Specializes for primitive byte arrays to avoid double copying
pub struct BytesDictBuilder {
    lookup: Option<HashTable<u64>>,
    views: BufferMut<BinaryView>,
    values: ByteBufferMut,
    hasher: RandomState,
    dtype: DType,
}

impl BytesDictBuilder {
    pub fn new(dtype: DType) -> Self {
        Self {
            lookup: Some(HashTable::new()),
            views: BufferMut::<BinaryView>::empty(),
            values: BufferMut::empty(),
            hasher: DefaultHashBuilder::default(),
            dtype,
        }
    }

    #[inline]
    fn lookup_bytes(&self, idx: usize) -> &[u8] {
        let bin_view = &self.views[idx];
        if bin_view.is_inlined() {
            bin_view.as_inlined().value()
        } else {
            &self.values[bin_view.as_view().to_range()]
        }
    }

    #[inline]
    fn encode_value(&mut self, lookup: &mut HashTable<u64>, val: &[u8]) -> u64 {
        *lookup
            .entry(
                self.hasher.hash_one(val),
                |idx| val == self.lookup_bytes(idx.as_()),
                |idx| self.hasher.hash_one(self.lookup_bytes(idx.as_())),
            )
            .or_insert_with(|| {
                let next_code = self.views.len() as u64;
                if val.len() <= BinaryView::MAX_INLINED_SIZE {
                    self.views.push(BinaryView::new_inlined(val));
                } else {
                    self.views.push(BinaryView::new_view(
                        u32::try_from(val.len()).vortex_unwrap(),
                        val[0..4].try_into().vortex_unwrap(),
                        0,
                        u32::try_from(self.values.len()).vortex_unwrap(),
                    ));
                    self.values.extend_from_slice(val);
                }
                next_code
            })
            .get()
    }

    fn encode_bytes<A: ArrayAccessor<[u8]>>(
        &mut self,
        accessor: &A,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let mut local_lookup = self.lookup.take().vortex_expect("Must have a lookup dict");
        let mut codes: BufferMut<u64> = BufferMut::with_capacity(len);

        let (codes, validity) = if self.dtype.is_nullable() {
            let mut null_buf = NullBufferBuilder::new(len);

            accessor.with_iterator(|it| {
                for value in it {
                    let (code, validity) = value
                        .map(|v| (self.encode_value(&mut local_lookup, v), true))
                        .unwrap_or((0, false));
                    null_buf.append(validity);
                    unsafe { codes.push_unchecked(code) }
                }
            })?;
            (
                codes,
                null_buf
                    .finish()
                    .map(Validity::from)
                    .unwrap_or(Validity::AllValid),
            )
        } else {
            accessor.with_iterator(|it| {
                for value in it {
                    let code = self.encode_value(
                        &mut local_lookup,
                        value.vortex_expect("Dict encode null value in non-nullable array"),
                    );
                    unsafe { codes.push_unchecked(code) }
                }
            })?;
            (codes, Validity::NonNullable)
        };

        // Restore lookup dictionary back into the struct
        self.lookup = Some(local_lookup);
        Ok(PrimitiveArray::new(codes, validity).into_array())
    }
}

impl DictEncoder for BytesDictBuilder {
    fn encode(&mut self, array: &dyn Array) -> VortexResult<ArrayRef> {
        if &self.dtype != array.dtype() {
            vortex_bail!(
                "Array DType {} does not match builder dtype {}",
                array.dtype(),
                self.dtype
            );
        }

        let len = array.len();
        let codes = if let Some(varbinview) = array.as_opt::<VarBinViewArray>() {
            self.encode_bytes(varbinview, len)?
        } else if let Some(varbin) = array.as_opt::<VarBinArray>() {
            self.encode_bytes(varbin, len)?
        } else {
            vortex_bail!("Can only dictionary encode VarBin and VarBinView arrays");
        };

        Ok(codes)
    }

    fn values(&mut self) -> VortexResult<ArrayRef> {
        VarBinViewArray::try_new(
            self.views.clone().freeze(),
            vec![self.values.clone().freeze()],
            self.dtype.clone(),
            self.dtype.nullability().into(),
        )
        .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod test {
    use std::str;

    use vortex_array::ToCanonical;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::VarBinArray;

    use crate::builders::dict_encode;

    #[test]
    fn encode_varbin() {
        let arr = VarBinArray::from(vec!["hello", "world", "hello", "again", "world"]);
        let dict = dict_encode(&arr).unwrap();
        assert_eq!(
            dict.codes().to_primitive().unwrap().as_slice::<u64>(),
            &[0, 1, 0, 2, 1]
        );
        dict.values()
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| {
                assert_eq!(
                    iter.flatten()
                        .map(|b| unsafe { str::from_utf8_unchecked(b) })
                        .collect::<Vec<_>>(),
                    vec!["hello", "world", "again"]
                );
            })
            .unwrap();
    }

    #[test]
    fn encode_varbin_nulls() {
        let arr: VarBinArray = vec![
            Some("hello"),
            None,
            Some("world"),
            Some("hello"),
            None,
            Some("again"),
            Some("world"),
            None,
        ]
        .into_iter()
        .collect();
        let dict = dict_encode(&arr).unwrap();
        assert_eq!(
            dict.codes().to_primitive().unwrap().as_slice::<u64>(),
            &[0, 0, 1, 0, 0, 2, 1, 0]
        );
        dict.values()
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| {
                assert_eq!(
                    iter.map(|b| b.map(|v| unsafe { str::from_utf8_unchecked(v) }))
                        .collect::<Vec<_>>(),
                    vec![Some("hello"), Some("world"), Some("again")]
                );
            })
            .unwrap();
    }

    #[test]
    fn repeated_values() {
        let arr = VarBinArray::from(vec!["a", "a", "b", "b", "a", "b", "a", "b"]);
        let dict = dict_encode(&arr).unwrap();
        dict.values()
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| {
                assert_eq!(
                    iter.flatten()
                        .map(|b| unsafe { str::from_utf8_unchecked(b) })
                        .collect::<Vec<_>>(),
                    vec!["a", "b"]
                );
            })
            .unwrap();
        assert_eq!(
            dict.codes().to_primitive().unwrap().as_slice::<u64>(),
            &[0u64, 0, 1, 1, 0, 1, 0, 1]
        );
    }
}
