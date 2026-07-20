// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link HadoopUtils}, which forwards S3 and Azure credentials from a Hadoop {@link Configuration} to
 * the Vortex object-store property keys.
 *
 * <p>Characterizes the {@code fs.s3a.*} to {@code aws_*} key mapping (including https-qualification of bare endpoints),
 * the prefix-based Azure account key and SAS token extraction, and the skip-signature fallback for anonymous Azure
 * access.
 */
final class HadoopUtilsTest {
    /** A Configuration that does not load the default resources, so tests see only the keys they set. */
    private static Configuration emptyConf() {
        return new Configuration(/* loadDefaults= */ false);
    }

    // --- s3PropertiesFromHadoopConf ---

    @Test
    @DisplayName("Maps fs.s3a credentials, session token, and region to the aws_* property keys")
    void s3MapsCredentialKeys() {
        Configuration conf = emptyConf();
        conf.set(HadoopUtils.FS_S3A_ACCESS_KEY, "AKIAEXAMPLE");
        conf.set(HadoopUtils.FS_S3A_SECRET_KEY, "secret");
        conf.set(HadoopUtils.FS_S3A_SESSION_TOKEN, "token");
        conf.set(HadoopUtils.FS_S3A_ENDPOINT_REGION, "us-west-2");

        Map<String, String> properties = HadoopUtils.s3PropertiesFromHadoopConf(conf);

        assertEquals("AKIAEXAMPLE", properties.get("aws_access_key_id"));
        assertEquals("secret", properties.get("aws_secret_access_key"));
        assertEquals("token", properties.get("aws_session_token"));
        assertEquals("us-west-2", properties.get("aws_region"));
    }

    @Test
    @DisplayName("Qualifies a bare fs.s3a.endpoint with https://")
    void s3QualifiesBareEndpoint() {
        Configuration conf = emptyConf();
        conf.set(HadoopUtils.FS_S3A_ENDPOINT, "s3.us-west-2.amazonaws.com");

        Map<String, String> properties = HadoopUtils.s3PropertiesFromHadoopConf(conf);

        assertEquals("https://s3.us-west-2.amazonaws.com", properties.get("aws_endpoint"));
    }

    @Test
    @DisplayName("Preserves an fs.s3a.endpoint that already has an http or https scheme")
    void s3PreservesSchemedEndpoint() {
        Configuration httpConf = emptyConf();
        httpConf.set(HadoopUtils.FS_S3A_ENDPOINT, "http://localhost:9000");
        assertEquals(
                "http://localhost:9000",
                HadoopUtils.s3PropertiesFromHadoopConf(httpConf).get("aws_endpoint"));

        Configuration httpsConf = emptyConf();
        httpsConf.set(HadoopUtils.FS_S3A_ENDPOINT, "https://s3.example.com");
        assertEquals(
                "https://s3.example.com",
                HadoopUtils.s3PropertiesFromHadoopConf(httpsConf).get("aws_endpoint"));
    }

    @Test
    @DisplayName("Ignores Hadoop keys that are not S3-relevant")
    void s3IgnoresUnrelatedKeys() {
        Configuration conf = emptyConf();
        conf.set("fs.defaultFS", "hdfs://namenode:8020");
        conf.set("fs.s3a.connection.maximum", "64");

        assertTrue(HadoopUtils.s3PropertiesFromHadoopConf(conf).isEmpty());
    }

    @Test
    @DisplayName("Returns no properties for an empty configuration")
    void s3EmptyConfYieldsNoProperties() {
        assertTrue(HadoopUtils.s3PropertiesFromHadoopConf(emptyConf()).isEmpty());
    }

    // --- azurePropertiesFromHadoopConf ---

    @Test
    @DisplayName("Extracts the storage account key from any fs.azure.account.key-prefixed entry")
    void azureExtractsAccountKey() {
        Configuration conf = emptyConf();
        conf.set(HadoopUtils.ACCESS_KEY_PREFIX + ".myaccount.dfs.core.windows.net", "account-key");

        Map<String, String> properties = HadoopUtils.azurePropertiesFromHadoopConf(conf);

        assertEquals("account-key", properties.get("azure_storage_account_key"));
        assertFalse(properties.containsKey("azure_skip_signature"));
    }

    @Test
    @DisplayName("Extracts the SAS token from any fs.azure.sas.fixed.token-prefixed entry")
    void azureExtractsSasToken() {
        Configuration conf = emptyConf();
        conf.set(HadoopUtils.FIXED_TOKEN_PREFIX + "myaccount.dfs.core.windows.net", "sas-token");

        Map<String, String> properties = HadoopUtils.azurePropertiesFromHadoopConf(conf);

        assertEquals("sas-token", properties.get("azure_storage_sas_key"));
    }

    @Test
    @DisplayName("Falls back to skipping signatures when no account key is configured")
    void azureSkipsSignatureWithoutAccountKey() {
        Map<String, String> properties = HadoopUtils.azurePropertiesFromHadoopConf(emptyConf());

        assertEquals("true", properties.get("azure_skip_signature"));
        assertFalse(properties.containsKey("azure_storage_account_key"));
    }

    @Test
    @DisplayName("Skips signatures even when only a SAS token is configured")
    void azureSasOnlyStillSkipsSignature() {
        Configuration conf = emptyConf();
        conf.set(HadoopUtils.FIXED_TOKEN_PREFIX + "myaccount.dfs.core.windows.net", "sas-token");

        Map<String, String> properties = HadoopUtils.azurePropertiesFromHadoopConf(conf);

        assertEquals("sas-token", properties.get("azure_storage_sas_key"));
        assertEquals("true", properties.get("azure_skip_signature"));
    }

    @Test
    @DisplayName("Ignores Hadoop keys that are not Azure-relevant")
    void azureIgnoresUnrelatedKeys() {
        Configuration conf = emptyConf();
        conf.set("fs.azure.io.retry.max.retries", "5");
        conf.set("fs.defaultFS", "abfss://container@account.dfs.core.windows.net/");

        Map<String, String> properties = HadoopUtils.azurePropertiesFromHadoopConf(conf);

        assertFalse(properties.containsKey("azure_storage_account_key"));
        assertFalse(properties.containsKey("azure_storage_sas_key"));
    }
}
