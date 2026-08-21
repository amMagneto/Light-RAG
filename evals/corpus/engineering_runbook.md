# Engineering Runbook

## Deployment Process

All production deployments require two approvals from senior engineers. Deploys happen only inside the release window of Tuesday and Thursday, 10:00 to 16:00 UTC. Every deployment starts as a canary receiving 5 percent of traffic for 30 minutes before full rollout. If a release misbehaves, run the rollback script scripts/rollback.sh to return to the previous version.

## On-Call Rotation

The on-call rotation hands over every Monday at 09:00. If the primary engineer does not acknowledge a page within 15 minutes, alerting escalates automatically to the secondary on-call.

## Incident Severity Levels

A SEV1 incident means a full outage of customer-facing services; responders must acknowledge and respond within 5 minutes and update the public status page within 15 minutes. A SEV2 incident is a partial degradation with a target response of 30 minutes. A SEV3 issue is minor and may be handled the next business day.

## Database Operations

Take a backup snapshot before every production migration. When the primary database grows beyond 500GB, open a sharding review ticket.
