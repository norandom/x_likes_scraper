/*
 * Scrubbed slice of the X.com main client-web bundle.
 *
 * This file is *not* the real bundle. It contains only the minimum bytes
 * the test suite needs the auth-extraction regexes in
 * x_likes_exporter/auth.py to match against:
 *
 *   - bearer_pattern = r'"(Bearer [\w%]+)"'
 *       -> matches the literal "Bearer REDACTEDBEARERTOKEN0000" below
 *   - query_pattern  = rf'{{queryId:"([^"]+)",operationName:"{operation_name}"'
 *       -> for operation_name="Likes", matches the literal
 *          {queryId:"REDACTEDQUERYID000000Likes",operationName:"Likes"} below
 *
 * Every real credential that the live bundle carries (the production bearer
 * token, the production GraphQL query ids for other operations, etc.) is
 * replaced with a REDACTED placeholder.
 */

var __MARKER_AUTH__ = "Bearer REDACTEDBEARERTOKEN0000";

var __MARKER_QUERIES__ = [
  {queryId:"REDACTEDQUERYID000000Likes",operationName:"Likes",operationType:"query",metadata:{featureSwitches:[],fieldToggles:[]}},
  {queryId:"REDACTEDQUERYID000001UserBy",operationName:"UserByScreenName",operationType:"query"}
];

// The rest of the bundle is irrelevant to the auth tests and would normally
// be ~5MB of minified webpack output. Stripped from this fixture.
function __noop__() { return null; }
