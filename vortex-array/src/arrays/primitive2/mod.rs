use arrow_buffer::BooleanBufferBuilder;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{DType, Nullability, PType};

use crate::stats::ArrayStats;
use crate::validity::Validity;

pub trait PrimitiveType<const NULLABLE: bool>: Sized {
    const DTYPE: DType;

    type Storage;

    fn zero() -> Self::Storage;

    fn width() -> usize {
        size_of::<Self::Storage>()
    }

    fn nullability() -> Nullability {
        if NULLABLE {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        }
    }
}

// Just a very easy type to use here, I expect some complications down the road but
// the built-in integers should be able to work with a macro.
struct U8DType<const NULLABLE: bool>;

impl PrimitiveType<false> for U8DType<false> {
    // In this world, does nullability still needed as part of the dtype?
    const DTYPE: DType = DType::Primitive(PType::U8, Nullability::NonNullable);
    type Storage = u8;

    fn zero() -> Self::Storage {
        0_u8
    }
}

impl PrimitiveType<true> for u8 {
    const DTYPE: DType = DType::Primitive(PType::U8, Nullability::Nullable);
    type Storage = u8;

    fn zero() -> Self {
        0
    }
}

/// Array of fixed sized primitive-typed items
/// This struct is 16 bytes smaller than the current `PrimitiveArray` (96 vs 80)
pub struct PrimitiveArray<const NULLABLE: bool, T: PrimitiveType<NULLABLE>> {
    // Typed buffer is nice!
    buffer: Buffer<T::Storage>,
    validity: Validity,
    stats_set: ArrayStats,
}

impl<T> From<Vec<T::Storage>> for PrimitiveArray<false, T>
where
    T: PrimitiveType<false>,
{
    fn from(value: Vec<T::Storage>) -> Self {
        Self {
            buffer: Buffer::from_iter(value),
            validity: Validity::AllValid,
            stats_set: Default::default(),
        }
    }
}

impl<T> FromIterator<Option<T::Storage>> for PrimitiveArray<true, T>
where
    T: PrimitiveType<true>,
{
    fn from_iter<I: IntoIterator<Item = Option<T::Storage>>>(iter: I) -> Self {
        let mut buffer = BufferMut::empty();
        let mut validity = BooleanBufferBuilder::new(0);
        for i in iter.into_iter() {
            match i {
                Some(v) => {
                    buffer.push(v);
                    validity.append(true);
                }
                None => {
                    buffer.push(T::zero());
                    validity.append(false);
                }
            }
        }

        Self {
            buffer: buffer.freeze(),
            validity: Validity::from(validity.finish()),
            stats_set: Default::default(),
        }
    }
}
