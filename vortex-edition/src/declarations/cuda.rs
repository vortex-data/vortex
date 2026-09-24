// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Draft output contract for files read through the CUDA reader.
//!
//! This edition includes the host-side metadata and canonical fallback representations.
//! It describes wire compatibility. It does not guarantee that every dtype or operation executes
//! on the GPU, or that a particular compression scheme is fast on the GPU.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionFamily;
use crate::EditionId;
use crate::EditionMember;

/// The CUDA reader's independently versioned wire-format family.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "cuda",
    origin: "vortex-cuda",
    doc: "Serialized representations supported by the CUDA reader, including host-side metadata.",
};

/// The September 2026 draft CUDA output edition.
pub const CUDA_2026_09_0: EditionId = EditionId::new("cuda", 2026, 9, 0);

/// Complete output permissions for the CUDA reader.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CUDA_2026_09_0,
        min_library_version: None,
    },
    added: &[
        EditionMember::array(&"fastlanes.bitpacked"),
        EditionMember::array(&"fastlanes.for"),
        EditionMember::array(&"vortex.alp"),
        EditionMember::array(&"vortex.bool"),
        EditionMember::array(&"vortex.bytebool"),
        EditionMember::array(&"vortex.chunked"),
        EditionMember::array(&"vortex.constant"),
        EditionMember::array(&"vortex.datetimeparts"),
        EditionMember::array(&"vortex.decimal"),
        EditionMember::array(&"vortex.decimal_byte_parts"),
        EditionMember::array(&"vortex.dict"),
        EditionMember::array(&"vortex.ext"),
        EditionMember::array(&"vortex.fsst"),
        EditionMember::array(&"vortex.list"),
        EditionMember::array(&"vortex.null"),
        EditionMember::array(&"vortex.primitive"),
        EditionMember::array(&"vortex.runend"),
        EditionMember::array(&"vortex.struct"),
        EditionMember::array(&"vortex.varbin"),
        EditionMember::array(&"vortex.varbinview"),
        EditionMember::array(&"vortex.zigzag"),
        EditionMember::layout(&"vortex.chunked"),
        EditionMember::layout(&"vortex.dict"),
        EditionMember::layout(&"vortex.flat"),
        EditionMember::layout(&"vortex.stats"),
        EditionMember::layout(&"vortex.struct"),
        EditionMember::dtype(&"vortex.date"),
        EditionMember::dtype(&"vortex.time"),
        EditionMember::dtype(&"vortex.timestamp"),
        EditionMember::array(&"vortex.sequence"),
        EditionMember::array(&"vortex.zstd"),
        EditionMember::array(&"vortex.fixed_size_list"),
        EditionMember::array(&"vortex.listview"),
        EditionMember::array(&"vortex.masked"),
        EditionMember::layout(&"vortex.zoned"),
        EditionMember::aggregate(&"vortex.bounded_max"),
        EditionMember::aggregate(&"vortex.bounded_min"),
        EditionMember::aggregate(&"vortex.max"),
        EditionMember::aggregate(&"vortex.min"),
        EditionMember::aggregate(&"vortex.nan_count"),
        EditionMember::aggregate(&"vortex.null_count"),
        EditionMember::array(&"vortex.onpair"),
        EditionMember::array(&"vortex.map"),
        EditionMember::dtype(&"vortex.uuid"),
        EditionMember::array(&"fastlanes.delta"),
        EditionMember::array(&"vortex.zstd_buffers"),
        EditionMember::layout(&"vortex.cuda_flat"),
    ],
};
