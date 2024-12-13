#!/bin/bash

set -Eeu -o pipefail -x

commit_id=${GITHUB_SHA}

jq --compact-output 'select(.reason == "benchmark-complete" or .reason == null)
    | if (.throughput | length) == 0
      then ([{
               name: (.name // .id),
               unit: .unit,
               value: (.value // .mean.estimate),
               commit_id: "'$commit_id'"
           }])
      else ([{
               name: .id,
               unit: .unit,
               value: .mean.estimate,
               commit_id: "'$commit_id'"
           }, {
               name: (.id + " throughput"),
               unit: (.throughput[0].unit + "/" + .unit),
               value: (.throughput[0].per_iteration / .mean.estimate),
               time: .mean.estimate,
               bytes: .throughput[0].per_iteration,
               commit_id: "'$commit_id'"
           }])
      end
    | .[]
'
