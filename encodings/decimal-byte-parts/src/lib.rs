// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An encoding that compresses decimals with integer compression schemes.
//!
//! A decimal is split into parts. The _part layout_ is a signed most significant part (MSP) plus
//! up to three non-nullable `u64` lower parts. The MSP has the decimal's nullability and the signed
//! integer dtype at the canonical width of its precision, capped at `i64`. Lower parts exist only
//! above precision 18. Precision 19 to 38 allows at most one lower part, and precision 39 to 76
//! allows up to three.
//!
//! Narrow MSP storage uses an integer [`Widen`] over a child that can be compressed. The canonical
//! width of precision `p` is the smallest width that holds `10^p - 1`, so every value of a narrower
//! signed integer fits the precision. The same is true when the parts span fewer bits than the
//! canonical width, such as an `i64` MSP with no lower parts at precision 19, or one lower part at
//! precision 39. Decoding sign-extends such values to the canonical width. So the MSP never needs a
//! range scan, and its `Widen` does not need to record a precision. Operations work on the parts
//! when possible, and decoding produces the canonical decimal array only when a consumer needs it.
//!
//! The wire stores the child of an MSP `Widen`, so older readers need no new format. Readers wrap
//! a narrower stored MSP without scanning. Only a wider legacy MSP needs a host copy, which fails
//! on a valid value outside the precision and writes zeros in null slots.
//!
//! [`Widen`]: vortex_array::arrays::Widen

use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

use crate::decimal_byte_parts::compute::aggregate::DecimalBytePartsAggregateKernel;
use crate::decimal_byte_parts::compute::is_constant::DecimalBytePartsIsConstantKernel;

mod decimal_byte_parts;
pub use decimal_byte_parts::*;

/// Initialize decimal-byte-parts encoding in the given session.
pub fn initialize(session: &VortexSession) {
    // One plugin owns both serialized formats: registering it reads either ID and writes the
    // one that fits the array. Which of them a writer may emit is decided by its editions.
    session.arrays().register(DecimalBytePartsPlugin);
    compute::kernel::initialize(session);

    for aggregate_id in [MinMax.id(), Min.id(), Max.id(), IsSorted.id(), Sum.id()] {
        session.aggregate_fns().register_aggregate_kernel(
            DecimalByteParts.id(),
            Some(aggregate_id),
            &DecimalBytePartsAggregateKernel,
        );
    }

    session.aggregate_fns().register_aggregate_kernel(
        DecimalByteParts.id(),
        Some(IsConstant.id()),
        &DecimalBytePartsIsConstantKernel,
    );
}
