// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import org.apache.hadoop.conf.Configuration;

/**
 * Java-serializable wrapper around a Hadoop {@link Configuration}.
 *
 * <p>{@code org.apache.spark.util.SerializableConfiguration} does the same job but is {@code private[spark]}, so this
 * class keeps the connector off Spark internals.
 */
final class SerializableHadoopConf implements Serializable {
    private static final long serialVersionUID = 1L;

    private transient Configuration conf;

    SerializableHadoopConf(Configuration conf) {
        this.conf = conf;
    }

    Configuration value() {
        return conf;
    }

    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        conf.write(out);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        conf = new Configuration(false);
        conf.readFields(in);
    }
}
