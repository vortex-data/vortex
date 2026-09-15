// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <stdio.h>
#include <vortex.h>

int main(void) {
    printf("%d %d\n", VORTEX_HEADER_VERSION, vx_fixture_value());
    return 0;
}
