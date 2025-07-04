// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

public interface File extends AutoCloseable {
    DType getDType();

    long rowCount();

    ArrayIterator newScan(ScanOptions options);

    @Override
    void close();
}
