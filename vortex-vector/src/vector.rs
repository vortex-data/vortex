// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of the [`Vector`] type, which represents immutable and fully decompressed (canonical)
//! array data.
//!
//! [`Vector`] can be transformed into the [`VectorMut`] type if it is owned.

use std::fmt::Debug;
use std::ops::RangeBounds;

use crate::Scalar;
use crate::VectorMut;
use crate::VectorOps;
use crate::binaryview::BinaryVector;
use crate::binaryview::StringVector;
use crate::bool::BoolVector;
use crate::decimal::DecimalVector;
use crate::fixed_size_list::FixedSizeListVector;
use crate::listview::ListViewVector;
use crate::match_each_vector;
use crate::null::NullVector;
use crate::primitive::PrimitiveVector;
use crate::struct_::StructVector;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

/// An enum over all kinds of immutable vectors, which represent fully decompressed (canonical)
/// array data.
///
/// Most of the behavior of `Vector` is described by the [`VectorOps`] trait. Note that vectors are
/// **always** considered as nullable, and it is the responsibility of the user to not add any
/// nullable data to a vector they want to keep as non-nullable.
///
/// The mutable equivalent of this type is [`VectorMut`], which implements the
/// [`VectorMutOps`](crate::VectorMutOps) trait.
#[derive(Debug, Clone)]
pub enum Vector {
    /// Null vectors.
    Null(NullVector),
    /// Boolean vectors.
    Bool(BoolVector),
    /// Decimal vectors.
    ///
    /// Note that [`DecimalVector`] is an enum over the different possible (generic)
    /// [`DVector<D>`](crate::decimal::DVector)s.
    ///
    /// See the [documentation](crate::decimal) for more information.
    Decimal(DecimalVector),
    /// Primitive vectors.
    ///
    /// Note that [`PrimitiveVector`] is an enum over the different possible (generic)
    /// [`PVector<T>`](crate::primitive::PVector)s.
    ///
    /// See the [documentation](crate::primitive) for more information.
    Primitive(PrimitiveVector),
    /// String vectors
    String(StringVector),
    /// Binary vectors
    Binary(BinaryVector),
    /// Vectors of Lists with variable sizes.
    List(ListViewVector),
    /// Vectors of Lists with fixed sizes.
    FixedSizeList(FixedSizeListVector),
    /// Vectors of Struct elements.
    Struct(StructVector),
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        // Validity patterns must match
        if self.validity() != other.validity() {
            return false;
        }
        // Delegate to the underlying vector type equality
        match (self, other) {
            (Vector::Null(a), Vector::Null(b)) => a == b,
            (Vector::Bool(a), Vector::Bool(b)) => a == b,
            (Vector::Decimal(a), Vector::Decimal(b)) => a == b,
            (Vector::Primitive(a), Vector::Primitive(b)) => a == b,
            (Vector::String(a), Vector::String(b)) => a == b,
            (Vector::Binary(a), Vector::Binary(b)) => a == b,
            (Vector::List(a), Vector::List(b)) => a == b,
            (Vector::FixedSizeList(a), Vector::FixedSizeList(b)) => a == b,
            (Vector::Struct(a), Vector::Struct(b)) => a == b,
            _ => false, // Different variants are not equal
        }
    }
}

impl VectorOps for Vector {
    type Mutable = VectorMut;
    type Scalar = Scalar;

    fn len(&self) -> usize {
        match_each_vector!(self, |v| { v.len() })
    }

    fn validity(&self) -> &Mask {
        match_each_vector!(self, |v| { v.validity() })
    }

    fn mask_validity(&mut self, mask: &Mask) {
        match_each_vector!(self, |v| { v.mask_validity(mask) })
    }

    fn scalar_at(&self, index: usize) -> Scalar {
        match_each_vector!(self, |v| { v.scalar_at(index).into() })
    }

    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        match_each_vector!(self, |v| { Vector::from(v.slice(range)) })
    }

    fn clear(&mut self) {
        match_each_vector!(self, |v| { v.clear() })
    }

    fn try_into_mut(self) -> Result<VectorMut, Self> {
        match_each_vector!(self, |v| {
            v.try_into_mut().map(VectorMut::from).map_err(Vector::from)
        })
    }

    fn into_mut(self) -> VectorMut {
        match_each_vector!(self, |v| { VectorMut::from(v.into_mut()) })
    }
}

impl Vector {
    /// Returns a reference to the inner [`NullVector`] if `self` is of that variant.
    pub fn as_null(&self) -> &NullVector {
        if let Vector::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVector, got {self:?}");
    }

    /// Returns a reference to the inner [`BoolVector`] if `self` is of that variant.
    pub fn as_bool(&self) -> &BoolVector {
        if let Vector::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVector, got {self:?}");
    }

    /// Returns a reference to the inner [`PrimitiveVector`] if `self` is of that variant.
    pub fn as_primitive(&self) -> &PrimitiveVector {
        if let Vector::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector, got {self:?}");
    }

    /// Returns a reference to the inner [`StringVector`] if `self` is of that variant.
    pub fn as_string(&self) -> &StringVector {
        if let Vector::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }

    /// Returns a reference to the inner [`BinaryVector`] if `self` is of that variant.
    pub fn as_binary(&self) -> &BinaryVector {
        if let Vector::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    /// Returns a reference to the inner [`ListViewVector`] if `self` is of that variant.
    pub fn as_list(&self) -> &ListViewVector {
        if let Vector::List(v) = self {
            return v;
        }
        vortex_panic!("Expected ListViewVector, got {self:?}");
    }

    /// Returns a reference to the inner [`FixedSizeListVector`] if `self` is of that variant.
    pub fn as_fixed_size_list(&self) -> &FixedSizeListVector {
        if let Vector::FixedSizeList(v) = self {
            return v;
        }
        vortex_panic!("Expected FixedSizeListVector, got {self:?}");
    }

    /// Returns a reference to the inner [`StructVector`] if `self` is of that variant.
    pub fn as_struct(&self) -> &StructVector {
        if let Vector::Struct(v) = self {
            return v;
        }
        vortex_panic!("Expected StructVector, got {self:?}");
    }
}

impl Vector {
    /// Returns a reference to the inner [`NullVector`] if `self` is of that variant.
    pub fn as_null_mut(&mut self) -> &mut NullVector {
        if let Vector::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVector, got {self:?}");
    }

    /// Returns a reference to the inner [`BoolVector`] if `self` is of that variant.
    pub fn as_bool_mut(&mut self) -> &mut BoolVector {
        if let Vector::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVector, got {self:?}");
    }

    /// Returns a reference to the inner [`PrimitiveVector`] if `self` is of that variant.
    pub fn as_primitive_mut(&mut self) -> &mut PrimitiveVector {
        if let Vector::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector, got {self:?}");
    }

    /// Returns a reference to the inner [`StringVector`] if `self` is of that variant.
    pub fn as_string_mut(&mut self) -> &mut StringVector {
        if let Vector::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }

    /// Returns a reference to the inner [`BinaryVector`] if `self` is of that variant.
    pub fn as_binary_mut(&mut self) -> &mut BinaryVector {
        if let Vector::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    /// Returns a reference to the inner [`ListViewVector`] if `self` is of that variant.
    pub fn as_list_mut(&mut self) -> &mut ListViewVector {
        if let Vector::List(v) = self {
            return v;
        }
        vortex_panic!("Expected ListViewVector, got {self:?}");
    }

    /// Returns a reference to the inner [`FixedSizeListVector`] if `self` is of that variant.
    pub fn as_fixed_size_list_mut(&mut self) -> &mut FixedSizeListVector {
        if let Vector::FixedSizeList(v) = self {
            return v;
        }
        vortex_panic!("Expected FixedSizeListVector, got {self:?}");
    }

    /// Returns a reference to the inner [`StructVector`] if `self` is of that variant.
    pub fn as_struct_mut(&mut self) -> &mut StructVector {
        if let Vector::Struct(v) = self {
            return v;
        }
        vortex_panic!("Expected StructVector, got {self:?}");
    }
}

impl Vector {
    /// Consumes `self` and returns the inner [`NullVector`] if `self` is of that variant.
    pub fn into_null(self) -> NullVector {
        if let Vector::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`BoolVector`] if `self` is of that variant.
    pub fn into_bool(self) -> BoolVector {
        if let Vector::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`PrimitiveVector`] if `self` is of that variant.
    pub fn into_primitive(self) -> PrimitiveVector {
        if let Vector::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`DecimalVector`] if `self` is of that variant.
    pub fn into_decimal(self) -> DecimalVector {
        if let Vector::Decimal(v) = self {
            return v;
        }
        vortex_panic!("Expected DecimalVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`StringVector`] if `self` is of that variant.
    #[expect(
        clippy::same_name_method,
        reason = "intentionally shadows VarBinTypeDowncast method"
    )]
    pub fn into_string(self) -> StringVector {
        if let Vector::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`BinaryVector`] if `self` is of that variant.
    #[expect(
        clippy::same_name_method,
        reason = "intentionally shadows VarBinTypeDowncast method"
    )]
    pub fn into_binary(self) -> BinaryVector {
        if let Vector::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`ListViewVector`] if `self` is of that variant.
    pub fn into_list(self) -> ListViewVector {
        if let Vector::List(v) = self {
            return v;
        }
        vortex_panic!("Expected ListViewVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`FixedSizeListVector`] if `self` is of that
    /// variant.
    pub fn into_fixed_size_list(self) -> FixedSizeListVector {
        if let Vector::FixedSizeList(v) = self {
            return v;
        }
        vortex_panic!("Expected FixedSizeListVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`StructVector`] if `self` is of that variant.
    pub fn into_struct(self) -> StructVector {
        if let Vector::Struct(v) = self {
            return v;
        }
        vortex_panic!("Expected StructVector, got {self:?}");
    }
}

impl Vector {
    /// Consumes `self` and returns the inner [`NullVector`] if `self` is of that variant.
    pub fn into_null_opt(self) -> Option<NullVector> {
        if let Vector::Null(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Consumes `self` and returns the inner [`BoolVector`] if `self` is of that variant.
    pub fn into_bool_opt(self) -> Option<BoolVector> {
        if let Vector::Bool(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Consumes `self` and returns the inner [`PrimitiveVector`] if `self` is of that variant.
    pub fn into_primitive_opt(self) -> Option<PrimitiveVector> {
        if let Vector::Primitive(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Consumes `self` and returns the inner [`DecimalVector`] if `self` is of that variant.
    pub fn into_decimal_opt(self) -> Option<DecimalVector> {
        if let Vector::Decimal(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Consumes `self` and returns the inner [`StringVector`] if `self` is of that variant.
    pub fn into_string_opt(self) -> Option<StringVector> {
        if let Vector::String(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Consumes `self` and returns the inner [`BinaryVector`] if `self` is of that variant.
    pub fn into_binary_opt(self) -> Option<BinaryVector> {
        if let Vector::Binary(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Consumes `self` and returns the inner [`ListViewVector`] if `self` is of that variant.
    pub fn into_list_opt(self) -> Option<ListViewVector> {
        if let Vector::List(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Consumes `self` and returns the inner [`FixedSizeListVector`] if `self` is of that variant.
    pub fn into_fixed_size_list_opt(self) -> Option<FixedSizeListVector> {
        if let Vector::FixedSizeList(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Consumes `self` and returns the inner [`StructVector`] if `self` is of that variant.
    pub fn into_struct_opt(self) -> Option<StructVector> {
        if let Vector::Struct(v) = self {
            Some(v)
        } else {
            None
        }
    }
}
