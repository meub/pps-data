#!/usr/bin/env bash
# Password-protect https://ppsdata.info/preview/* with a CloudFront Function.
#
#   1. Renders function.js with the credentials in credentials.env (gitignored).
#   2. Creates or updates the CloudFront Function and publishes it to LIVE.
#   3. Adds a /preview/* cache behavior to the distribution (a copy of the
#      default behavior plus the function), or updates the function on it.
#
# Idempotent. Rerun after changing the password. Reads DISTRIBUTION_ID from
# the repo's deploy.env.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
# shellcheck disable=SC1091
source "$root/deploy.env"
# shellcheck disable=SC1091
source "$here/credentials.env"

name=ppsdata-preview-auth
pattern="/preview/*"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

token=$(printf '%s:%s' "$PREVIEW_USER" "$PREVIEW_PASSWORD" | base64)
sed "s|__BASIC_TOKEN__|$token|" "$here/function.js" > "$work/function.js"

if etag=$(aws cloudfront describe-function --name "$name" --query ETag --output text 2>/dev/null); then
  echo "==> updating function $name"
  etag=$(aws cloudfront update-function --name "$name" --if-match "$etag" \
    --function-config Comment="Basic auth for /preview/*",Runtime=cloudfront-js-2.0 \
    --function-code "fileb://$work/function.js" --query ETag --output text)
else
  echo "==> creating function $name"
  etag=$(aws cloudfront create-function --name "$name" \
    --function-config Comment="Basic auth for /preview/*",Runtime=cloudfront-js-2.0 \
    --function-code "fileb://$work/function.js" --query ETag --output text)
fi
echo "==> publishing $name"
aws cloudfront publish-function --name "$name" --if-match "$etag" >/dev/null
arn=$(aws cloudfront describe-function --name "$name" --stage LIVE \
  --query FunctionSummary.FunctionMetadata.FunctionARN --output text)

echo "==> adding $pattern behavior to $DISTRIBUTION_ID"
aws cloudfront get-distribution-config --id "$DISTRIBUTION_ID" > "$work/dist.json"
dist_etag=$(jq -r .ETag "$work/dist.json")
jq --arg p "$pattern" --arg arn "$arn" '
  .DistributionConfig
  | (.DefaultCacheBehavior + {
      PathPattern: $p,
      FunctionAssociations: {Quantity: 1, Items: [{FunctionARN: $arn, EventType: "viewer-request"}]}
    }) as $b
  | .CacheBehaviors.Items = ([(.CacheBehaviors.Items // [])[] | select(.PathPattern != $p)] + [$b])
  | .CacheBehaviors.Quantity = (.CacheBehaviors.Items | length)
' "$work/dist.json" > "$work/new.json"
aws cloudfront update-distribution --id "$DISTRIBUTION_ID" --if-match "$dist_etag" \
  --distribution-config "file://$work/new.json" --query Distribution.Status --output text

echo "==> done. The distribution takes a few minutes to deploy. Check with:"
echo "    curl -s -o /dev/null -w '%{http_code}\\n' https://ppsdata.info/preview/budget.html   # expect 401"
