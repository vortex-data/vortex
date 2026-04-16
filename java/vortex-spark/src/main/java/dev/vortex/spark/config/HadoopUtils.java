// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.config;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;

public final class HadoopUtils {
    private HadoopUtils() {}

    static final String FS_S3A_ACCESS_KEY = "fs.s3a.access.key";
    static final String FS_S3A_SECRET_KEY = "fs.s3a.secret.key";
    static final String FS_S3A_SESSION_TOKEN = "fs.s3a.session.token";
    static final String FS_S3A_ENDPOINT = "fs.s3a.endpoint";
    static final String FS_S3A_ENDPOINT_REGION = "fs.s3a.endpoint.region";

    public static Map<String, String> s3PropertiesFromHadoopConf(Configuration hadoopConf) {
        VortexS3Properties properties = new VortexS3Properties();

        for (Map.Entry<String, String> entry : hadoopConf) {
            switch (entry.getKey()) {
                case FS_S3A_ACCESS_KEY:
                    properties.setAccessKeyId(entry.getValue());
                    break;
                case FS_S3A_SECRET_KEY:
                    properties.setSecretAccessKey(entry.getValue());
                    break;
                case FS_S3A_SESSION_TOKEN:
                    properties.setSessionToken(entry.getValue());
                    break;
                case FS_S3A_ENDPOINT:
                    String qualified = entry.getValue();
                    if (!qualified.startsWith("http")) {
                        qualified = "https://" + qualified;
                    }
                    properties.setEndpoint(qualified);
                    break;
                case FS_S3A_ENDPOINT_REGION:
                    properties.setRegion(entry.getValue());
                    break;
                default:
                    break;
            }
        }

        return properties.asProperties();
    }

    static final String ACCESS_KEY_PREFIX = "fs.azure.account.key";
    static final String FIXED_TOKEN_PREFIX = "fs.azure.sas.fixed.token.";

    public static Map<String, String> azurePropertiesFromHadoopConf(Configuration hadoopConf) {
        VortexAzureProperties properties = new VortexAzureProperties();

        // TODO(aduffy): match on storage account name.
        for (Map.Entry<String, String> entry : hadoopConf) {
            String configKey = entry.getKey();
            if (configKey.startsWith(ACCESS_KEY_PREFIX)) {
                properties.setAccessKey(entry.getValue());
            } else if (configKey.startsWith(FIXED_TOKEN_PREFIX)) {
                properties.setSasKey(entry.getValue());
            }
        }

        if (properties.accessKey().isEmpty()) {
            properties.setSkipSignature(true);
        }

        return properties.asProperties();
    }
}
