#!/bin/bash

# Localstack setup script, mounted into containers to initialize state for tests.
# Reference - https://docs.localstack.cloud/references/init-hooks/

set -Eeu -o pipefail -x

export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=development

awslocal s3api create-bucket --bucket test-bucket
