// CloudFront Function (cloudfront-js-2.0), viewer-request, on the /preview/*
// behavior of the ppsdata.info distribution. Asks for HTTP Basic auth before
// serving drafts. setup.sh replaces __BASIC_TOKEN__ with base64("user:password")
// from credentials.env (gitignored); never commit the rendered token.
//
// Viewer-request runs before the cache, so cached copies are still gated.
function handler(event) {
  var auth = event.request.headers.authorization;
  if (auth && auth.value === "Basic __BASIC_TOKEN__") {
    return event.request;
  }
  return {
    statusCode: 401,
    statusDescription: "Unauthorized",
    headers: {
      "www-authenticate": { value: 'Basic realm="PPS Data Explorer preview"' },
      "cache-control": { value: "no-store" },
    },
  };
}
