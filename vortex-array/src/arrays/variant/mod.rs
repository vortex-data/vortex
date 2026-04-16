// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vtable;

use vortex_error::VortexExpect;

pub use self::vtable::Variant;
pub use self::vtable::VariantArray;
use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::dtype::DType;

pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

pub trait VariantArrayExt: TypedArrayRef<Variant> {
    fn child(&self) -> &ArrayRef {
        self.as_ref().slots()[0]
            .as_ref()
            .vortex_expect("validated variant child slot")
    }
}
impl<T: TypedArrayRef<Variant>> VariantArrayExt for T {}

impl Array<Variant> {
    /// Creates a new `VariantArray`.
    pub fn new(child: ArrayRef) -> Self {
        let dtype = DType::Variant(child.dtype().nullability());
        let len = child.len();
        let stats = child.statistics().to_owned();
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Variant, dtype, len, EmptyArrayData).with_slots(vec![Some(child)]),
            )
        }
        .with_stats_set(stats)
    }
}
