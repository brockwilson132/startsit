# Start/Sit data pipeline

Rebuilds the fantasy projections on a schedule and publishes them, so the shared
Start/Sit app is always current. Nothing to run by hand once it's set up.

## Where things live
- **App:** https://brockwilson132.github.io/startsit/ (rebuilt on every run, with fresh data baked in)
- **Data feed:** npm package `startsit-nfl-data`, used by the private claude.ai copy of the app.
  It publishes through npm trusted publishing (npmjs.com → package → Settings →
  Trusted Publisher → GitHub Actions: `brockwilson132` / `startsit` / `update.yml`).
  If that isn't set up, the npm step fails quietly and the app on Pages still updates.

## Schedule (Eastern time)
Tue 8am (after Monday night) · Wed/Thu/Fri 6pm (injury reports) · Sat noon ·
Sun 11:30am (before kickoff). Edit the `cron` lines in
`.github/workflows/update.yml` to change it. Run it manually any time from the
Actions tab.

## How it works
`pipeline.py` downloads nflverse stats, snap counts, play-by-play, injury
reports and Vegas lines, plus FantasyPros consensus rankings. It fits the model
on last season, tests it on the season before, projects the next unplayed week,
writes `dist/data.js`, then `build_site.py` bakes it into the app for GitHub Pages. Each run also saves a snapshot to `history/` so the
app can score the model against the experts after games are played.

If the package name `startsit-nfl-data` is ever taken, change it in
`package.json` and in the workflow's purge URL, and ask Claude to update the
app's `LIVE_URL` to match.
