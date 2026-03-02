//! Static GUIDs that bindgen generates incorrectly.
//! See [relevant issue](https://github.com/rust-lang/rust-bindgen/issues/1888).

use super::nvEncodeAPI::GUID;

// Search for `//.*\nstatic const\s+GUID\s+NV_ENC_\w+_GUID\s+=\n\{.*\};`

// =========================================================================================
// Encode Codec GUIDS supported by the NvEncodeAPI interface.
// =========================================================================================

/// GUID for the H.264 encoding.
/// {6BC82762-4E63-4ca4-AA85-1E50F321F6BF}
pub const NV_ENC_CODEC_H264_GUID: GUID = GUID {
    Data1: 0x6bc8_2762,
    Data2: 0x4e63,
    Data3: 0x4ca4,
    Data4: [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
};

/// GUID for the H.265 encoding.
/// {790CDC88-4522-4d7b-9425-BDA9975F7603}
pub const NV_ENC_CODEC_HEVC_GUID: GUID = GUID {
    Data1: 0x790c_dc88,
    Data2: 0x4522,
    Data3: 0x4d7b,
    Data4: [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x3],
};

/// GUID for the AV1 encoding.
/// {0A352289-0AA7-4759-862D-5D15CD16D254}
pub const NV_ENC_CODEC_AV1_GUID: GUID = GUID {
    Data1: 0x0a35_2289,
    Data2: 0x0aa7,
    Data3: 0x4759,
    Data4: [0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54],
};

// =========================================================================================
// * Encode Profile GUIDS supported by the NvEncodeAPI interface.
// =========================================================================================

/// GUID for the autoselect profile.
// {BFD6F8E7-233C-4341-8B3E-4818523803F4}
pub const NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID: GUID = GUID {
    Data1: 0xbfd6_f8e7,
    Data2: 0x233c,
    Data3: 0x4341,
    Data4: [0x8b, 0x3e, 0x48, 0x18, 0x52, 0x38, 0x3, 0xf4],
};

/// GUID for the H.264 encoding baseline profile.
/// {0727BCAA-78C4-4c83-8C2F-EF3DFF267C6A}
pub const NV_ENC_H264_PROFILE_BASELINE_GUID: GUID = GUID {
    Data1: 0x0727_bcaa,
    Data2: 0x78c4,
    Data3: 0x4c83,
    Data4: [0x8c, 0x2f, 0xef, 0x3d, 0xff, 0x26, 0x7c, 0x6a],
};

/// GUID for the H.264 encoding main profile.
/// {60B5C1D4-67FE-4790-94D5-C4726D7B6E6D}
pub const NV_ENC_H264_PROFILE_MAIN_GUID: GUID = GUID {
    Data1: 0x60b5_c1d4,
    Data2: 0x67fe,
    Data3: 0x4790,
    Data4: [0x94, 0xd5, 0xc4, 0x72, 0x6d, 0x7b, 0x6e, 0x6d],
};

/// GUID for the H.264 encoding, high quality profile.
/// {E7CBC309-4F7A-4b89-AF2A-D537C92BE310}
pub const NV_ENC_H264_PROFILE_HIGH_GUID: GUID = GUID {
    Data1: 0xe7cb_c309,
    Data2: 0x4f7a,
    Data3: 0x4b89,
    Data4: [0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10],
};

/// GUID for the H.264 high quality, `YCbCR444` digital color format profile.
/// {7AC663CB-A598-4960-B844-339B261A7D52}
pub const NV_ENC_H264_PROFILE_HIGH_444_GUID: GUID = GUID {
    Data1: 0x7ac6_63cb,
    Data2: 0xa598,
    Data3: 0x4960,
    Data4: [0xb8, 0x44, 0x33, 0x9b, 0x26, 0x1a, 0x7d, 0x52],
};

/// GUID for the H.264, stereo encoding profile.
/// {40847BF5-33F7-4601-9084-E8FE3C1DB8B7}
pub const NV_ENC_H264_PROFILE_STEREO_GUID: GUID = GUID {
    Data1: 0x4084_7bf5,
    Data2: 0x33f7,
    Data3: 0x4601,
    Data4: [0x90, 0x84, 0xe8, 0xfe, 0x3c, 0x1d, 0xb8, 0xb7],
};

/// GUID for the H.264, progressive encoding profile.
/// {B405AFAC-F32B-417B-89C4-9ABEED3E5978}
pub const NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID: GUID = GUID {
    Data1: 0xb405_afac,
    Data2: 0xf32b,
    Data3: 0x417b,
    Data4: [0x89, 0xc4, 0x9a, 0xbe, 0xed, 0x3e, 0x59, 0x78],
};

/// GUID for the H.264, constrained encoding profile.
/// {AEC1BD87-E85B-48f2-84C3-98BCA6285072}
pub const NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID: GUID = GUID {
    Data1: 0xaec1_bd87,
    Data2: 0xe85b,
    Data3: 0x48f2,
    Data4: [0x84, 0xc3, 0x98, 0xbc, 0xa6, 0x28, 0x50, 0x72],
};

/// GUID for the H.265 main (8-bit) encoding profile.
/// {B514C39A-B55B-40fa-878F-F1253B4DFDEC}
pub const NV_ENC_HEVC_PROFILE_MAIN_GUID: GUID = GUID {
    Data1: 0xb514_c39a,
    Data2: 0xb55b,
    Data3: 0x40fa,
    Data4: [0x87, 0x8f, 0xf1, 0x25, 0x3b, 0x4d, 0xfd, 0xec],
};

/// GUID for H.265 Main10 (10-bit) encoding profile.
/// {fa4d2b6c-3a5b-411a-8018-0a3f5e3c9be5}
pub const NV_ENC_HEVC_PROFILE_MAIN10_GUID: GUID = GUID {
    Data1: 0xfa4d_2b6c,
    Data2: 0x3a5b,
    Data3: 0x411a,
    Data4: [0x80, 0x18, 0x0a, 0x3f, 0x5e, 0x3c, 0x9b, 0xe5],
};

/// GUID for H.265, JM 16 (`FRExt`) encoding profile.
/// {51ec32b5-1b4c-453c-9cbd-b616bd621341}
pub const NV_ENC_HEVC_PROFILE_FREXT_GUID: GUID = GUID {
    Data1: 0x51ec_32b5,
    Data2: 0x1b4c,
    Data3: 0x453c,
    Data4: [0x9c, 0xbd, 0xb6, 0x16, 0xbd, 0x62, 0x13, 0x41],
};

/// GUID for the AV1 main encoding preset.
/// {5f2a39f5-f14e-4f95-9a9e-b76d568fcf97}
pub const NV_ENC_AV1_PROFILE_MAIN_GUID: GUID = GUID {
    Data1: 0x5f2a_39f5,
    Data2: 0xf14e,
    Data3: 0x4f95,
    Data4: [0x9a, 0x9e, 0xb7, 0x6d, 0x56, 0x8f, 0xcf, 0x97],
};

// =========================================================================================
// * Preset GUIDS supported by the NvEncodeAPI interface.
// =========================================================================================

// Performance degrades and quality improves as we move from P1 to P7. Presets
// P3 to P7 for H264 and Presets P2 to P7 for HEVC have B frames enabled by
// default for HIGH_QUALITY and LOSSLESS tuning info, and will not work with
// Weighted Prediction enabled. In case Weighted Prediction is required, disable
// B frames by setting frameIntervalP = 1

/// GUID for the P1 (highest performance) encoding preset.
/// {FC0A8D3E-45F8-4CF8-80C7-298871590EBF}
pub const NV_ENC_PRESET_P1_GUID: GUID = GUID {
    Data1: 0xfc0a_8d3e,
    Data2: 0x45f8,
    Data3: 0x4cf8,
    Data4: [0x80, 0xc7, 0x29, 0x88, 0x71, 0x59, 0xe, 0xbf],
};

/// GUID for the P2 (higher performance) encoding preset.
/// Has B-frames enabled by default for H.265 `HIGH_QUALITY` and `LOSSLESS`
/// tuning info, and will not work with Weighted Prediction enabled.
/// {F581CFB8-88D6-4381-93F0-DF13F9C27DAB}
pub const NV_ENC_PRESET_P2_GUID: GUID = GUID {
    Data1: 0xf581_cfb8,
    Data2: 0x88d6,
    Data3: 0x4381,
    Data4: [0x93, 0xf0, 0xdf, 0x13, 0xf9, 0xc2, 0x7d, 0xab],
};

/// GUID for the P3 (high performance) encoding preset.
/// Has B-frames enabled by default for H.264 and H.265: `HIGH_QUALITY` and
/// `LOSSLESS` tuning info, and will not work with Weighted Prediction enabled.
/// {36850110-3A07-441F-94D5-3670631F91F6}
pub const NV_ENC_PRESET_P3_GUID: GUID = GUID {
    Data1: 0x3685_0110,
    Data2: 0x3a07,
    Data3: 0x441f,
    Data4: [0x94, 0xd5, 0x36, 0x70, 0x63, 0x1f, 0x91, 0xf6],
};

/// GUID for the P4 (balanced) encoding preset.
/// Has B-frames enabled by default for H.264 and H.265: `HIGH_QUALITY` and
/// `LOSSLESS` tuning info, and will not work with Weighted Prediction enabled.
/// {90A7B826-DF06-4862-B9D2-CD6D73A08681}
pub const NV_ENC_PRESET_P4_GUID: GUID = GUID {
    Data1: 0x90a7_b826,
    Data2: 0xdf06,
    Data3: 0x4862,
    Data4: [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81],
};

/// GUID for the P5 (high quality) encoding preset.
/// Has B-frames enabled by default for H.264 and H.265: `HIGH_QUALITY` and
/// `LOSSLESS` tuning info, and will not work with Weighted Prediction enabled.
/// {21C6E6B4-297A-4CBA-998F-B6CBDE72ADE3}
pub const NV_ENC_PRESET_P5_GUID: GUID = GUID {
    Data1: 0x21c6_e6b4,
    Data2: 0x297a,
    Data3: 0x4cba,
    Data4: [0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3],
};

/// GUID for the P6 (higher quality) encoding preset.
/// Has B-frames enabled by default for H.264 and H.265: `HIGH_QUALITY` and
/// `LOSSLESS` tuning info, and will not work with Weighted Prediction enabled.
/// {8E75C279-6299-4AB6-8302-0B215A335CF5}
pub const NV_ENC_PRESET_P6_GUID: GUID = GUID {
    Data1: 0x8e75_c279,
    Data2: 0x6299,
    Data3: 0x4ab6,
    Data4: [0x83, 0x2, 0xb, 0x21, 0x5a, 0x33, 0x5c, 0xf5],
};

/// GUID for the P6 (highest quality) encoding preset.
/// Has B-frames enabled by default for H.264 and H.265: `HIGH_QUALITY` and
/// `LOSSLESS` tuning info, and will not work with Weighted Prediction enabled.
/// {84848C12-6F71-4C13-931B-53E283F57974}
pub const NV_ENC_PRESET_P7_GUID: GUID = GUID {
    Data1: 0x8484_8c12,
    Data2: 0x6f71,
    Data3: 0x4c13,
    Data4: [0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74],
};
