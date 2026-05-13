// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// FSST-8 (8-bit code) wrapper. Renames the upstream `extern "C" fsst_*` and
// `namespace libfsst` symbols so this can be linked alongside the FSST-12
// build in the same binary.

#define NONOPT_FSST 1
#define libfsst libfsst8
#define fsst_create fsst8_create
#define fsst_duplicate fsst8_duplicate
#define fsst_export fsst8_export
#define fsst_destroy fsst8_destroy
#define fsst_import fsst8_import
#define fsst_decoder fsst8_decoder
#define fsst_compress fsst8_compress
#define fsst_decompress fsst8_decompress
#define fsst_decoder_t fsst8_decoder_t
#define fsst_encoder_t fsst8_encoder_t
#define fsst_compressAVX512 fsst8_compressAVX512
#define fsst_hasAVX512 fsst8_hasAVX512

#include "../vendor/fsst_cpp/libfsst.cpp"

// Non-inline export of the otherwise inline `fsst_decompress`. Ensures the
// Rust FFI sees a stable C symbol.
extern "C" size_t fsst8_decompress_export(
    const fsst8_decoder_t *decoder,
    size_t lenIn,
    const unsigned char *strIn,
    size_t size,
    unsigned char *output)
{
    return fsst8_decompress(decoder, lenIn, strIn, size, output);
}
