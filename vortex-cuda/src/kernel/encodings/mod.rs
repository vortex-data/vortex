// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executors for encoded arrays.
//!
//! Executors decode children through the CUDA execution context and return canonical arrays.

mod alp;
mod bitpacked;
mod date_time_parts;
mod decimal_byte_parts;
mod delta;
mod for_;
mod fsst;
mod onpair;
mod runend;
mod sequence;
mod widen;
mod zigzag;
mod zstd;
mod zstd_buffers;

pub(crate) use alp::ALPExecutor;
pub(crate) use bitpacked::BitPackedExecutor;
pub(crate) use bitpacked::bitpacked_slice_view;
pub(crate) use date_time_parts::DateTimePartsExecutor;
pub(crate) use decimal_byte_parts::DecimalBytePartsExecutor;
pub(crate) use delta::DeltaExecutor;
pub(crate) use for_::FoRExecutor;
pub(crate) use fsst::DecodedVarBin;
pub(crate) use fsst::FSSTExecutor;
pub(crate) use fsst::decode_fsst_varbin;
pub(crate) use onpair::OnPairExecutor;
pub(crate) use onpair::decode_onpair_varbin;
pub(crate) use runend::RunEndExecutor;
pub(crate) use sequence::SequenceExecutor;
pub(crate) use widen::WidenExecutor;
pub(crate) use zigzag::ZigZagExecutor;
pub(crate) use zstd::ZstdExecutor;
pub use zstd::ZstdKernelPrep;
pub use zstd::zstd_kernel_prepare;
pub(crate) use zstd_buffers::ZstdBuffersExecutor;
