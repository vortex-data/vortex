// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVectorMut`].

use vortex_buffer::BitBufferMut;
use vortex_dtype::Nullability;
use vortex_error::vortex_panic;
use vortex_mask::MaskMut;

use super::BoolVector;
use crate::{VectorMutOps, VectorOps};

/// A mutable vector of boolean values.
///
/// Internally, the boolean values are stored as the bits of a [`BitBufferMut`] plus an optional
/// [`MaskMut`] for null booleans (where `true` represents a _valid_ boolean and `false` represents
/// a `null` boolean).
///
/// The immutable equivalent of this type is [`BoolVector`].
#[derive(Debug, Clone)]
pub struct BoolVectorMut {
    pub(super) bits: BitBufferMut,
    pub(super) validity: Option<MaskMut>,
}

impl BoolVectorMut {
    /// Creates a new mutable boolean vector with the given `capacity` and `nullability`.
    pub fn with_capacity(capacity: usize, nullability: Nullability) -> Self {
        let validity = nullability
            .is_nullable()
            .then(|| MaskMut::with_capacity(capacity));

        Self {
            bits: BitBufferMut::with_capacity(capacity),
            validity,
        }
    }
}

impl VectorMutOps for BoolVectorMut {
    type Immutable = BoolVector;

    fn nullability(&self) -> Nullability {
        Nullability::from(self.validity.is_some())
    }

    fn len(&self) -> usize {
        debug_assert!(
            self.validity
                .as_ref()
                .is_none_or(|mask| mask.len() == self.bits.len())
        );

        self.bits.len()
    }

    fn capacity(&self) -> usize {
        self.bits.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.bits.reserve(additional);

        if let Some(v) = self.validity.as_mut() {
            v.reserve(additional);
        }
    }

    fn extend_from_vector(&mut self, other: &BoolVector) {
        assert_eq!(
            self.nullability(),
            other.nullability(),
            "tried to extend a vector with nullability {} with another vector with nullability {}",
            self.nullability(),
            other.nullability(),
        );

        self.bits.append_buffer(&other.bits);

        match (&mut self.validity, &other.validity) {
            (Some(self_v), Some(other_v)) => self_v.append_mask(other_v),
            (Some(self_v), None) => self_v.append_n(true, other.bits.len()),
            (None, Some(other_v)) => {
                let mut new_validity = MaskMut::new_true(self.bits.len() - other.bits.len());
                new_validity.append_mask(other_v);
                self.validity = Some(new_validity);
            }
            (None, None) => {}
        }
    }

    fn append_nulls(&mut self, n: usize) {
        let Some(mask) = &mut self.validity else {
            vortex_panic!("tried to append nulls to a non-nullable vector")
        };

        mask.append_n(false, n);

        self.bits.append_n(false, n);
    }

    fn freeze(self) -> Self::Immutable {
        BoolVector {
            bits: self.bits.freeze(),
            validity: self.validity.map(|v| v.freeze()),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        BoolVectorMut {
            bits: self.bits.split_off(at),
            validity: self.validity.as_mut().map(|v| v.split_off(at)),
        }
    }

    fn unsplit(&mut self, other: Self) {
        // TODO(connor): We must check `other`'s nullability in relation to `self`.

        let other_len = other.bits.len();
        self.bits.unsplit(other.bits);

        match (&mut self.validity, other.validity) {
            (Some(self_v), Some(other_v)) => self_v.unsplit(other_v),
            (Some(self_v), None) => self_v.append_n(true, other_len),
            (None, Some(other_v)) => {
                let mut new_validity = MaskMut::new_true(self.bits.len() - other_len);
                new_validity.unsplit(other_v);
                self.validity = Some(new_validity);
            }
            (None, None) => {}
        }
    }
}
