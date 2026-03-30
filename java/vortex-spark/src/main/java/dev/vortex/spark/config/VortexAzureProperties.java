// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.config;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.util.Map;
import java.util.Optional;

public final class VortexAzureProperties {
    private static final String ACCOUNT_KEY = "azure_storage_account_key";
    private static final String SAS_KEY = "azure_storage_sas_key";
    private static final String SKIP_SIGNATURE = "azure_skip_signature";

    private final Map<String, String> properties = Maps.newHashMap();

    public Optional<String> accessKey() {
        return Optional.ofNullable(properties.get(ACCOUNT_KEY));
    }

    public Optional<String> sasKey() {
        return Optional.ofNullable(properties.get(SAS_KEY));
    }

    public boolean skipSignature() {
        return Boolean.parseBoolean(properties.getOrDefault(SKIP_SIGNATURE, "false"));
    }

    public VortexAzureProperties setAccessKey(String accountKey) {
        properties.put(ACCOUNT_KEY, accountKey);
        return this;
    }

    public VortexAzureProperties setSasKey(String sasKey) {
        properties.put(SAS_KEY, sasKey);
        return this;
    }

    public VortexAzureProperties setSkipSignature(boolean skipSignature) {
        properties.put(SKIP_SIGNATURE, String.valueOf(skipSignature));
        return this;
    }

    public Map<String, String> asProperties() {
        return ImmutableMap.copyOf(properties);
    }
}
