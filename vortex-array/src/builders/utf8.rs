use std::any::Any;

use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;

use crate::builders::varbinview_builder::VarBinViewBuilder;
use crate::builders::ArrayBuilder;
use crate::Array;

pub struct Utf8Builder {
    varbinview_builder: VarBinViewBuilder,
}

impl Utf8Builder {
    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            varbinview_builder: VarBinViewBuilder::with_capacity(
                DType::Utf8(nullability),
                capacity,
            ),
        }
    }

    #[inline]
    pub fn append_value<S: AsRef<str>>(&mut self, value: S) {
        self.varbinview_builder
            .append_value(value.as_ref().as_bytes());
    }

    #[inline]
    pub fn append_option<S: AsRef<str>>(&mut self, value: Option<S>) {
        self.varbinview_builder
            .append_option(value.as_ref().map(|s| s.as_ref().as_bytes()));
    }
}

impl ArrayBuilder for Utf8Builder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.varbinview_builder.dtype()
    }

    #[inline]
    fn len(&self) -> usize {
        self.varbinview_builder.len()
    }

    #[inline]
    fn append_zeros(&mut self, n: usize) {
        self.varbinview_builder.append_zeros(n);
    }

    #[inline]
    fn append_nulls(&mut self, n: usize) {
        self.varbinview_builder.append_nulls(n);
    }

    #[inline]
    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        self.varbinview_builder.extend_from_array(array)
    }

    fn finish(&mut self) -> VortexResult<Array> {
        self.varbinview_builder.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::str::from_utf8;

    use itertools::Itertools;
    use vortex_dtype::Nullability;

    use crate::accessor::ArrayAccessor;
    use crate::array::VarBinViewArray;
    use crate::builders::{ArrayBuilder, Utf8Builder};

    #[test]
    fn test_utf8_builder() {
        let mut builder = Utf8Builder::with_capacity(Nullability::Nullable, 10);

        builder.append_option(Some("Hello"));
        builder.append_option::<&str>(None);
        builder.append_value("World");

        builder.append_nulls(2);

        builder.append_zeros(2);
        builder.append_value("test");

        let arr = VarBinViewArray::try_from(builder.finish().unwrap()).unwrap();

        let arr = arr
            .with_iterator(|iter| {
                iter.map(|x| x.map(|x| from_utf8(x).unwrap().to_string()))
                    .collect_vec()
            })
            .unwrap();
        assert_eq!(arr.len(), 8);
        assert_eq!(
            arr,
            vec![
                Some("Hello".to_string()),
                None,
                Some("World".to_string()),
                None,
                None,
                Some("".to_string()),
                Some("".to_string()),
                Some("test".to_string()),
            ]
        );
    }
    #[test]
    fn test_utf8_builder_with_extend() {
        let array = {
            let mut builder = Utf8Builder::with_capacity(Nullability::Nullable, 10);
            builder.append_null();
            builder.append_value("Hello2");
            builder.finish().unwrap()
        };
        let mut builder = Utf8Builder::with_capacity(Nullability::Nullable, 10);

        builder.append_option(Some("Hello1"));
        builder.extend_from_array(array).unwrap();
        builder.append_nulls(2);
        builder.append_value("Hello3");

        let arr = VarBinViewArray::try_from(builder.finish().unwrap()).unwrap();

        let arr = arr
            .with_iterator(|iter| {
                iter.map(|x| x.map(|x| from_utf8(x).unwrap().to_string()))
                    .collect_vec()
            })
            .unwrap();
        assert_eq!(arr.len(), 6);
        assert_eq!(
            arr,
            vec![
                Some("Hello1".to_string()),
                None,
                Some("Hello2".to_string()),
                None,
                None,
                Some("Hello3".to_string()),
            ]
        );
    }
}
