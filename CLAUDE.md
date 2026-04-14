# Notes for Claude

## Deploying

**Do not run `./deploy.sh` unless the user explicitly asks.** Make changes
locally — including regenerating `web/data.json` and committing — but
stop short of deploying. The user batches deploys themselves.

`./deploy.sh` syncs `web/index.html`, `web/data.json`, and the favicons to
S3 bucket `ppsdata.info` and invalidates CloudFront distribution
`E1Y8ZR63ATQJ9S`. The script is gitignored since it contains infra IDs.

Typical flow after a data or UI change:

```bash
.venv/bin/python scripts/export_web.py   # regen web/data.json if pipeline ran
# stop here — wait for the user to ask before running ./deploy.sh
```
