/**
 * Start/Sit ESPN relay (Cloudflare Worker).
 *
 * Lets the app read a PRIVATE ESPN fantasy league. Browsers can't attach ESPN login cookies to requests
 * from another site, so the phone sends its own cookies here in two headers and this worker adds them to
 * one request to ESPN's league endpoint, then returns ESPN's answer.
 *
 * It stores nothing and logs nothing. It only talks to ESPN's fantasy-league read endpoint, only for
 * GET requests from the app's own web address.
 *
 * Deploy: Cloudflare dashboard -> Workers & Pages -> Create -> Worker -> paste this file -> Deploy.
 */
const ALLOWED_ORIGINS = ['https://brockwilson132.github.io', 'http://localhost:8765'];
const VIEWS = new Set(['mTeam', 'mRoster', 'mSettings', 'mMatchup', 'mMatchupScore', 'mStatus']);

function cors(origin) {
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'X-ESPN-S2, X-ESPN-SWID',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}

export default {
  async fetch(request) {
    const origin = request.headers.get('Origin') || '';
    if (!ALLOWED_ORIGINS.includes(origin)) return new Response('Forbidden', { status: 403 });
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors(origin) });
    if (request.method !== 'GET') return new Response('Method not allowed', { status: 405, headers: cors(origin) });

    const url = new URL(request.url);
    const season = url.searchParams.get('season') || '';
    const league = url.searchParams.get('league') || '';
    const views = url.searchParams.getAll('view').filter((v) => VIEWS.has(v));
    if (!/^\d{4}$/.test(season) || !/^\d{1,12}$/.test(league) || !views.length) {
      return new Response('Bad request', { status: 400, headers: cors(origin) });
    }
    const s2 = request.headers.get('X-ESPN-S2') || '';
    const swid = request.headers.get('X-ESPN-SWID') || '';
    // ESPN's values: espn_s2 is URL-safe text; SWID is a GUID in braces
    if (!/^[A-Za-z0-9%+/=._-]{20,2000}$/.test(s2) || !/^\{?[0-9A-Fa-f-]{36}\}?$/.test(swid)) {
      return new Response('Missing or malformed ESPN cookies', { status: 401, headers: cors(origin) });
    }
    const target = `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${season}/segments/0/leagues/${league}?` +
      views.map((v) => 'view=' + v).join('&');
    const upstream = await fetch(target, {
      headers: { Cookie: `espn_s2=${s2}; SWID=${swid.startsWith('{') ? swid : '{' + swid + '}'}`, 'Accept': 'application/json' },
      cf: { cacheTtl: 0, cacheEverything: false },
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { ...cors(origin), 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    });
  },
};
