#!/bin/bash

set -Eeu -o pipefail -x

commit_id=$GITHUB_SHA
commit_title=$(git log -1 --pretty=%B $GITHUB_SHA | head -n 1)
commit_timestamp=$(git log -1 --format=%cd --date=iso-strict $GITHUB_SHA)

author_email=$(git log -1 --pretty=format:'%ae' $GITHUB_SHA)
author_name=$(git log -1 --pretty=format:'%an' $GITHUB_SHA)

committer_email=$(git log -1 --pretty=format:'%ce' $GITHUB_SHA)
committer_name=$(git log -1 --pretty=format:'%cn' $GITHUB_SHA)

tree_id=$(git rev-parse --verify ${GITHUB_SHA}^{tree})
repo_url=$(git config --get remote.origin.url | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/.git$//')

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
