set -o pipefail
set -e

# https://spiraldb.slack.com/archives/C07BV3GKAJ2/p1732736281946729
for package in $(cargo check -p  2>&1 | grep '^    ')
do
  echo ---- $package ----
  # Capture the output of clippy in $output, and also tee it to stdout.
  { output=$(cargo clippy --package $package --no-default-features 2>&1 | tee /dev/fd/3); } 3>&1 || true
  # Check if the return value from clippy is 0 or there was a compile_error match invoked
  ([ $? -eq 0 ] || echo "$output" | grep -q "compile_error!") && echo "Success and string found." || (echo "Failed or string not found." && exit 1)

done
