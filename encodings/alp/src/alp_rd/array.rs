use std::fmt::Debug;

use vortex_array::arrays::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::VTableRef;
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl,
    Canonical, Encoding, ProstMetadata, ToCanonical,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::alp_rd::alp_rd_decode;
use crate::alp_rd::serde::ALPRDMetadata;

#[derive(Clone, Debug)]
pub struct ALPRDArray {
    dtype: DType,
    left_parts: ArrayRef,
    left_parts_patches: Option<Patches>,
    left_parts_dictionary: Buffer<u16>,
    right_parts: ArrayRef,
    right_bit_width: u8,
    stats_set: ArrayStats,
}

pub struct ALPRDEncoding;
impl Encoding for ALPRDEncoding {
    type Array = ALPRDArray;
    type Metadata = ProstMetadata<ALPRDMetadata>;
}

impl ALPRDArray {
    pub fn try_new(
        dtype: DType,
        left_parts: ArrayRef,
        left_parts_dictionary: Buffer<u16>,
        right_parts: ArrayRef,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
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

        // we enforce right_parts to be non-nullable uint
        if !right_parts.dtype().is_unsigned_int() || right_parts.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable uint", right_parts.dtype());
        }

        let left_parts_patches = left_parts_patches
            .map(|patches| {
                if !patches.values().all_valid()? {
                    vortex_bail!("patches must be all valid: {}", patches.values());
                }
                // TODO(ngates): assert the DType, don't cast it.
                patches.cast_values(left_parts.dtype())
            })
            .transpose()?;

        Ok(Self {
            dtype,
            left_parts,
            left_parts_patches,
            left_parts_dictionary,
            right_parts,
            right_bit_width,
            stats_set: Default::default(),
        })
    }

    /// Returns true if logical type of the array values is f32.
    ///
    /// Returns false if the logical type of the array values is f64.
    #[inline]
    pub fn is_f32(&self) -> bool {
        matches!(&self.dtype, DType::Primitive(PType::F32, _))
    }

    /// The leftmost (most significant) bits of the floating point values stored in the array.
    ///
    /// These are bit-packed and dictionary encoded, and cannot directly be interpreted without
    /// the metadata of this array.
    pub fn left_parts(&self) -> &ArrayRef {
        &self.left_parts
    }

    /// The rightmost (least significant) bits of the floating point values stored in the array.
    pub fn right_parts(&self) -> &ArrayRef {
        &self.right_parts
    }

    #[inline]
    pub fn right_bit_width(&self) -> u8 {
        self.right_bit_width
    }

    /// Patches of left-most bits.
    pub fn left_parts_patches(&self) -> Option<&Patches> {
        self.left_parts_patches.as_ref()
    }

    /// The dictionary that maps the codes in `left_parts` into bit patterns.
    #[inline]
    pub fn left_parts_dictionary(&self) -> &Buffer<u16> {
        &self.left_parts_dictionary
    }

    pub fn replace_left_parts_patches(&mut self, patches: Option<Patches>) {
        self.left_parts_patches = patches;
    }
}

impl ArrayImpl for ALPRDArray {
    type Encoding = ALPRDEncoding;

    fn _len(&self) -> usize {
        self.left_parts.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&ALPRDEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        let left_parts = children[0].clone();
        let right_parts = children[1].clone();

        let left_part_patches = self.left_parts_patches().map(|existing| {
            let indices = children[2].clone();
            let values = children[3].clone();
            Patches::new(existing.array_len(), existing.offset(), indices, values)
        });

        ALPRDArray::try_new(
            self.dtype().clone(),
            left_parts,
            self.left_parts_dictionary().clone(),
            right_parts,
            self.right_bit_width(),
            left_part_patches,
        )
    }
}

impl ArrayCanonicalImpl for ALPRDArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        let left_parts = self.left_parts().to_primitive()?;
        let right_parts = self.right_parts().to_primitive()?;

        // Decode the left_parts using our builtin dictionary.
        let left_parts_dict = self.left_parts_dictionary();

        let decoded_array = if self.is_f32() {
            PrimitiveArray::new(
                alp_rd_decode::<f32>(
                    left_parts.into_buffer::<u16>(),
                    left_parts_dict,
                    self.right_bit_width,
                    right_parts.into_buffer_mut::<u32>(),
                    self.left_parts_patches(),
                )?,
                Validity::copy_from_array(self)?,
            )
        } else {
            PrimitiveArray::new(
                alp_rd_decode::<f64>(
                    left_parts.into_buffer::<u16>(),
                    left_parts_dict,
                    self.right_bit_width,
                    right_parts.into_buffer_mut::<u64>(),
                    self.left_parts_patches(),
                )?,
                Validity::copy_from_array(self)?,
            )
        };

        Ok(Canonical::Primitive(decoded_array))
    }
}

impl ArrayStatisticsImpl for ALPRDArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for ALPRDArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.left_parts().is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.left_parts().all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.left_parts().all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.left_parts().validity_mask()
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;

    use crate::{ALPRDFloat, alp_rd};

    #[rstest]
    #[case(vec![0.1f32.next_up(); 1024], 1.123_848_f32)]
    #[case(vec![0.1f64.next_up(); 1024], 1.123_848_591_110_992_f64)]
    fn test_array_encode_with_nulls_and_patches<T: ALPRDFloat>(
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
        let real_array = PrimitiveArray::from_option_iter(reals.iter().cloned());

        // Pick a seed that we know will trigger lots of patches.
        let encoder: alp_rd::RDEncoder = alp_rd::RDEncoder::new(&[seed.powi(-2)]);

        let rd_array = encoder.encode(&real_array);

        let decoded = rd_array.to_primitive().unwrap();

        let maybe_null_reals: Vec<T> = reals.into_iter().map(|v| v.unwrap_or_default()).collect();
        assert_eq!(decoded.as_slice::<T>(), &maybe_null_reals);
    }
}
