// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.util.Iterator;

public interface ArrayIterator extends AutoCloseable, Iterator<Array> {
    DType getDataType();

    @Override
    void close();
}
