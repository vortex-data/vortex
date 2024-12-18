#!/bin/bash

set -Eeu -o pipefail -x

function string_escape() {
    sed -e 's,\\,\\\\,g' -e 's,",\\",g'
}

commit_id=$GITHUB_SHA
commit_title=$(git log -1 --pretty=%B $GITHUB_SHA | head -n 1 | string_escape)
commit_timestamp=$(git log -1 --format=%cd --date=iso-strict $GITHUB_SHA | string_escape)

author_email=$(git log -1 --pretty=format:'%ae' $GITHUB_SHA | string_escape)
author_name=$(git log -1 --pretty=format:'%an' $GITHUB_SHA | string_escape)

committer_email=$(git log -1 --pretty=format:'%ce' $GITHUB_SHA | string_escape)
committer_name=$(git log -1 --pretty=format:'%cn' $GITHUB_SHA | string_escape)

tree_id=$(git rev-parse --verify ${GITHUB_SHA}^{tree})
repo_url=$(git config --get remote.origin.url | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/.git$//' | string_escape)

jq --compact-output '.' <<EOF
{
    "author": {
        "email": "$author_email",
        "name": "$author_name"
    },
    "committer": {
        "email": "$committer_email",
        "name": "$committer_name"
    },
    "id": "$commit_id",
    "message": "$commit_title",
    "timestamp": "$commit_timestamp",
    "tree_id": "$tree_id",
    "url": "$repo_url/commit/$commit_id"
}
EOF
