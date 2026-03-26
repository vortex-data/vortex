// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.config;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.util.Map;
import java.util.Optional;

public final class VortexS3Properties {
    private static final String ACCESS_KEY = "aws_access_key_id";
    private static final String SECRET_KEY = "aws_secret_access_key";
    private static final String SESSION_TOKEN = "aws_session_token";
    private static final String REGION = "aws_region";
    private static final String ENDPOINT = "aws_endpoint";
    private static final String SKIP_SIGNATURE = "aws_skip_signature";

    private final Map<String, String> properties = Maps.newHashMap();

    public Optional<String> accessKeyId() {
        return Optional.ofNullable(properties.get(ACCESS_KEY));
    }

    public Optional<String> secretAccessKey() {
        return Optional.ofNullable(properties.get(SECRET_KEY));
    }

    public Optional<String> sessionToken() {
        return Optional.ofNullable(properties.get(SESSION_TOKEN));
    }

    public Optional<String> region() {
        return Optional.ofNullable(properties.get(REGION));
    }

    public Optional<String> endpoint() {
        return Optional.ofNullable(properties.get(ENDPOINT));
    }

    public boolean skipSignature() {
        return Boolean.parseBoolean(properties.getOrDefault(SKIP_SIGNATURE, "false"));
    }

    public void setAccessKeyId(String accessKeyId) {
        properties.put(ACCESS_KEY, accessKeyId);
    }

    public void setSecretAccessKey(String secretAccessKey) {
        properties.put(SECRET_KEY, secretAccessKey);
    }

    public void setSessionToken(String sessionToken) {
        properties.put(SESSION_TOKEN, sessionToken);
    }

    public void setRegion(String region) {
        properties.put(REGION, region);
    }

    public void setEndpoint(String endpoint) {
        properties.put(ENDPOINT, endpoint);
    }

    public void setSkipSignature(boolean skipSignature) {
        properties.put(SKIP_SIGNATURE, Boolean.toString(skipSignature));
    }

    public Map<String, String> asProperties() {
        return ImmutableMap.copyOf(properties);
    }
}
