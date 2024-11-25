use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use vortex_array::array::{PrimitiveArray, SparseArray};
use vortex_array::encoding::ids;
use vortex_array::stats::{StatisticsVTable, StatsSet};
use vortex_array::validity::{ArrayValidity, LogicalValidity, ValidityVTable};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoCanonical,
};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::alp_rd::alp_rd_decode;

impl_encoding!("vortex.alprd", ids::ALP_RD, ALPRD);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPRDMetadata {
    right_bit_width: u8,
    dict_len: u8,
    dict: [u16; 8],
    left_parts_ptype: PType,
    has_exceptions: bool,
}

impl Display for ALPRDMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ALPRDArray {
    pub fn try_new(
        dtype: DType,
        left_parts: ArrayData,
        left_parts_dict: impl AsRef<[u16]>,
        right_parts: ArrayData,
        right_bit_width: u8,
        left_parts_exceptions: Option<ArrayData>,
    ) -> VortexResult<Self> {
        if !dtype.is_float() {
            vortex_bail!("ALPRDArray given invalid DType ({dtype})");
        }

        let len = left_parts.len();
        if right_parts.len() != len {
            vortex_bail!(
                "left_parts (len {}) and right_parts (len {}) must be of same length",
                len,
                right_parts.len()
            );
        }

        if !left_parts.dtype().is_unsigned_int() {
            vortex_bail!("left_parts dtype must be uint");
        }
        // we delegate array validity to the left_parts child
        if dtype.is_nullable() != left_parts.dtype().is_nullable() {
            vortex_bail!(
                "ALPRDArray dtype nullability ({}) must match left_parts dtype nullability ({})",
                dtype,
                left_parts.dtype()
            );
        }
        let left_parts_ptype =
            PType::try_from(left_parts.dtype()).vortex_expect("left_parts dtype must be uint");

        // we enforce right_parts to be non-nullable uint
        if right_parts.dtype().is_nullable() {
            vortex_bail!("right_parts dtype must be non-nullable");
        }
        if !right_parts.dtype().is_unsigned_int() || right_parts.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable uint", right_parts.dtype());
        }

        let mut children = vec![left_parts, right_parts];
        let has_exceptions = left_parts_exceptions.is_some();

        if let Some(exceptions) = left_parts_exceptions {
            // Enforce that the exceptions are SparseArray so that we have access to indices and values.
            if exceptions.encoding().id().code() != ids::SPARSE {
                vortex_bail!("left_parts_exceptions must be SparseArray encoded");
            }
            children.push(exceptions);
        }

        let mut dict = [0u16; 8];
        for (idx, v) in left_parts_dict.as_ref().iter().enumerate() {
            dict[idx] = *v;
        }

        Self::try_from_parts(
            dtype,
            len,
            ALPRDMetadata {
                right_bit_width,
                dict_len: left_parts_dict.as_ref().len() as u8,
                dict,
                left_parts_ptype,
                has_exceptions,
            },
            children.into(),
            StatsSet::default(),
        )
    }

    /// Returns true if logical type of the array values is f32.
    ///
    /// Returns false if the logical type of the array values is f64.
    #[inline]
    pub fn is_f32(&self) -> bool {
        PType::try_from(self.dtype()).vortex_expect("ALPRDArray must have primitive type")
            == PType::F32
    }

    /// The dtype of the left parts of the array.
    #[inline]
    fn left_parts_dtype(&self) -> DType {
        DType::Primitive(self.metadata().left_parts_ptype, self.dtype().nullability())
    }

    /// The dtype of the right parts of the array.
    #[inline]
    fn right_parts_dtype(&self) -> DType {
        DType::Primitive(
            if self.is_f32() {
                PType::U32
            } else {
                PType::U64
            },
            Nullability::NonNullable,
        )
    }

    /// The dtype of the exceptions of the left parts of the array.
    #[inline]
    fn left_parts_exceptions_dtype(&self) -> DType {
        DType::Primitive(self.metadata().left_parts_ptype, Nullability::Nullable)
    }

    /// The leftmost (most significant) bits of the floating point values stored in the array.
    ///
    /// These are bit-packed and dictionary encoded, and cannot directly be interpreted without
    /// the metadata of this array.
    pub fn left_parts(&self) -> ArrayData {
        self.as_ref()
            .child(0, &self.left_parts_dtype(), self.len())
            .vortex_expect("ALPRDArray: left_parts child")
    }

    /// The rightmost (least significant) bits of the floating point values stored in the array.
    pub fn right_parts(&self) -> ArrayData {
        self.as_ref()
            .child(1, &self.right_parts_dtype(), self.len())
            .vortex_expect("ALPRDArray: right_parts child")
    }

    /// Patches of left-most bits.
    pub fn left_parts_exceptions(&self) -> Option<ArrayData> {
        self.metadata().has_exceptions.then(|| {
            self.as_ref()
                .child(2, &self.left_parts_exceptions_dtype(), self.len())
                .vortex_expect("ALPRDArray: left_parts_exceptions child")
        })
    }

    /// The dictionary that maps the codes in `left_parts` into bit patterns.
    #[inline]
    pub fn left_parts_dict(&self) -> &[u16] {
        &self.metadata().dict[0..self.metadata().dict_len as usize]
    }

    #[inline]
    pub(crate) fn right_bit_width(&self) -> u8 {
        self.metadata().right_bit_width
    }
}

impl IntoCanonical for ALPRDArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        let left_parts = self.left_parts().into_canonical()?.into_primitive()?;
        let right_parts = self.right_parts().into_canonical()?.into_primitive()?;

        // Decode the left_parts using our builtin dictionary.
        let left_parts_dict = &self.metadata().dict[0..self.metadata().dict_len as usize];

        let exc_pos: Vec<u64>;
        let exc_u16: PrimitiveArray;

        if let Some(left_parts_exceptions) = self.left_parts_exceptions() {
            let left_parts_exceptions = SparseArray::try_from(left_parts_exceptions)
                .vortex_expect("ALPRDArray: exceptions must be SparseArray encoded");
            exc_pos = left_parts_exceptions
                .resolved_indices()
                .into_iter()
                .map(|v| v as _)
                .collect();
            exc_u16 = left_parts_exceptions
                .values()
                .into_canonical()?
                .into_primitive()?;
        } else {
            exc_pos = Vec::new();
            exc_u16 = PrimitiveArray::from(Vec::<u16>::new());
        }

        let decoded_array = if self.is_f32() {
            PrimitiveArray::from_vec(
                alp_rd_decode::<f32>(
                    left_parts.maybe_null_slice::<u16>(),
                    left_parts_dict,
                    self.metadata().right_bit_width,
                    right_parts.maybe_null_slice::<u32>(),
                    &exc_pos,
                    exc_u16.maybe_null_slice::<u16>(),
                ),
                self.logical_validity().into_validity(),
            )
        } else {
            PrimitiveArray::from_vec(
                alp_rd_decode::<f64>(
                    left_parts.maybe_null_slice::<u16>(),
                    left_parts_dict,
                    self.metadata().right_bit_width,
                    right_parts.maybe_null_slice::<u64>(),
                    &exc_pos,
                    exc_u16.maybe_null_slice::<u16>(),
                ),
                self.logical_validity().into_validity(),
            )
        };

        Ok(Canonical::Primitive(decoded_array))
    }
}

impl ValidityVTable<ALPRDArray> for ALPRDEncoding {
    fn is_valid(&self, array: &ALPRDArray, index: usize) -> bool {
        // Use validity from left_parts
        array.left_parts().is_valid(index)
    }

    fn logical_validity(&self, array: &ALPRDArray) -> LogicalValidity {
        // Use validity from left_parts
        array.left_parts().logical_validity()
    }
}

impl VisitorVTable<ALPRDArray> for ALPRDEncoding {
    fn accept(&self, array: &ALPRDArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("left_parts", &array.left_parts())?;
        visitor.visit_child("right_parts", &array.right_parts())?;
        if let Some(left_parts_exceptions) = array.left_parts_exceptions() {
            visitor.visit_child("left_parts_exceptions", &left_parts_exceptions)
        } else {
            Ok(())
        }
    }
}

impl StatisticsVTable<ALPRDArray> for ALPRDEncoding {}

impl ArrayTrait for ALPRDArray {}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::{IntoArrayData, IntoCanonical};

    use crate::{alp_rd, ALPRDFloat};

    #[rstest]
    #[case(vec![0.1f32.next_up(); 1024], 1.123_848_f32)]
    #[case(vec![0.1f64.next_up(); 1024], 1.123_848_591_110_992_f64)]
    fn test_array_encode_with_nulls_and_exceptions<T: ALPRDFloat>(
        #[case] reals: Vec<T>,
        #[case] seed: T,
    ) {
        assert_eq!(reals.len(), 1024, "test expects 1024-length fixture");
        // Null out some of the values.
        let mut reals: Vec<Option<T>> = reals.into_iter().map(Some).collect();
        reals[1] = None;
        reals[5] = None;
        reals[900] = None;

        // Create a new array from this.
        let real_array = PrimitiveArray::from_nullable_vec(reals.clone());

        // Pick a seed that we know will trigger lots of exceptions.
        let encoder: alp_rd::RDEncoder = alp_rd::RDEncoder::new(&[seed.powi(-2)]);

        let rd_array = encoder.encode(&real_array);

        let decoded = rd_array
            .into_array()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap();

        let maybe_null_reals: Vec<T> = reals.into_iter().map(|v| v.unwrap_or_default()).collect();
        assert_eq!(decoded.maybe_null_slice::<T>(), &maybe_null_reals);
    }
}
