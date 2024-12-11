#!/bin/bash

author_email=$(git log -1 --pretty=format:'%ae')
author_name=$(git log -1 --pretty=format:'%an')

committer_email=$(git log -1 --pretty=format:'%ce')
committer_name=$(git log -1 --pretty=format:'%cn')

commit_id=$(git rev-parse HEAD)
commit_title=$(git log -1 --pretty=%B | head -n 1)
commit_timestamp=$(git log -1 --format=%cd --date=iso-strict)
tree_id=$(git rev-parse --verify HEAD^{tree})
repo_url=$(git config --get remote.origin.url | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/.git$//')  # Convert to HTTPS format

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

