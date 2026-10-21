/**
 * Start/Sit update scheduler (Cloudflare Worker with a cron trigger).
 *
 * GitHub's own scheduled runs are best-effort and skip many hours, so this worker asks GitHub to run the
 * update workflow instead: every 10 minutes during NFL game windows (US Eastern time), hourly otherwise.
 * Needs one secret, GH_TOKEN: a fine-grained GitHub token with "Actions: read and write" on this repo only.
 * Deployed by .github/workflows/relay.yml, which also installs the secret.
 */
const REPO = 'brockwilson132/startsit';
const WORKFLOW = 'update.yml';

function etParts(ms) {
  const f = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', weekday: 'short', hour: 'numeric', minute: 'numeric', hourCycle: 'h23' });
  const p = Object.fromEntries(f.formatToParts(new Date(ms)).map((x) => [x.type, x.value]));
  return { day: p.weekday, hour: +p.hour, minute: +p.minute };
}

export function gameWindow({ day, hour }) {
  if (day === 'Sun') return hour >= 9;                                  // London games through Sunday night
  if (day === 'Mon') return hour < 1 || hour >= 19;                     // end of SNF; Monday night
  if (day === 'Tue' || day === 'Fri') return hour < 1;                  // end of MNF / TNF
  if (day === 'Thu') return hour >= 19;                                 // Thursday night
  if (day === 'Sat') return hour >= 12;                                 // December / January Saturday games
  return false;
}

export function shouldRun(ms) {
  const t = etParts(ms);
  return gameWindow(t) || t.minute < 10;                                // every tick in game windows, else once an hour
}

export default {
  async scheduled(event, env, ctx) {
    if (!shouldRun(event.scheduledTime)) return;
    const r = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${env.GH_TOKEN}`,
        Accept: 'application/vnd.github+json',
        'User-Agent': 'startsit-cron',
        'X-GitHub-Api-Version': '2022-11-28',
      },
      body: JSON.stringify({ ref: 'main' }),
    });
    if (!r.ok) console.log('dispatch failed', r.status, await r.text());
  },
  async fetch() {
    return new Response('Not found', { status: 404 });
  },
};
