// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// FSST-12 (12-bit code) wrapper. Renames the upstream `extern "C" fsst_*` and
// `namespace libfsst` symbols so this can be linked alongside the FSST-8
// build in the same binary.

#define NONOPT_FSST 1
#define libfsst libfsst12
#define fsst_create fsst12_create
#define fsst_duplicate fsst12_duplicate
#define fsst_export fsst12_export
#define fsst_destroy fsst12_destroy
#define fsst_import fsst12_import
#define fsst_decoder fsst12_decoder
#define fsst_compress fsst12_compress
#define fsst_decompress fsst12_decompress
#define fsst_decoder_t fsst12_decoder_t
#define fsst_encoder_t fsst12_encoder_t

#include "../vendor/fsst_cpp/libfsst12.cpp"

extern "C" size_t fsst12_decompress_export(
    const fsst12_decoder_t *decoder,
    size_t lenIn,
    const unsigned char *strIn,
    size_t size,
    unsigned char *output)
{
    return fsst12_decompress(decoder, lenIn, strIn, size, output);
}
