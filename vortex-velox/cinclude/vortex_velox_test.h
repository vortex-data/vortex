// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex_velox.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This interface exists only for Velox tests and benchmark fixture generation. */
typedef struct vx_velox_test_writer vx_velox_test_writer;

const vx_velox_array *vx_velox_test_array_from_arrow_apply(const vx_velox_session *session,
                                                           struct ArrowArray *array,
                                                           struct ArrowSchema *schema,
                                                           const vx_velox_expression *expression,
                                                           vx_velox_error **error_out);

vx_velox_test_writer *vx_velox_test_writer_new(vx_velox_view path, vx_velox_error **error_out);
int32_t vx_velox_test_writer_push(vx_velox_test_writer *writer,
                                  struct ArrowArray *array,
                                  struct ArrowSchema *schema,
                                  vx_velox_error **error_out);
int32_t vx_velox_test_writer_close(vx_velox_test_writer *writer, vx_velox_error **error_out);
void vx_velox_test_writer_abort(vx_velox_test_writer *writer);

#ifdef __cplusplus
}
#endif
