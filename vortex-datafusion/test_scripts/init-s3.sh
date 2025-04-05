#!/bin/bash

export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=development

awslocal s3api create-bucket --bucket test-bucket
