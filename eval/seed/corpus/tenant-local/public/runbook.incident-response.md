# Incident Response

## Severity levels

Sev-1 means a total outage affecting every tenant of the platform. It pages the on-call lead
immediately and opens a bridge call for the duration of the incident.

Sev-2 means degraded service for a subset of tenants. It is paged during business hours only,
with a four hour response target rather than fifteen minutes.

## Escalation

For a sev-1 incident the on-call lead must be paged within 15 minutes of detection. If the page
is unacknowledged after 5 minutes, escalation moves to the engineering manager, and after a
further 10 minutes to the director of engineering.

Escalation is automatic and does not require anyone to make a judgement call at three in the
morning, which is the entire point of having a written policy.

## Data retention

Incident records are retained for 30 days and then archived to cold storage, where they remain
queryable for a further 12 months before permanent deletion.

## Bridge calls

A sev-1 bridge call opens automatically when the page fires. The incident commander is whoever
holds the primary rota at that moment, not whoever happens to be awake and not whoever is most
senior. Handing command to a more senior person who has just joined costs the context the
commander has already built, and that context is usually worth more than the seniority.

The bridge stays open until the incident is downgraded or resolved. A scribe records decisions
with timestamps, because a postmortem written from memory reliably reconstructs a tidier
sequence of events than the one that actually happened.

## Communication

Customer communication for a sev-1 goes out within 30 minutes of confirmation, whether or not
the cause is known. Saying "we are investigating" early beats saying nothing accurately later:
silence during an outage is read as absence rather than as diligence.

Status page updates continue at 30 minute intervals until resolution, even when the update is
that nothing has changed. A gap in updates is indistinguishable, from outside, from a team that
has stopped working on the problem.

## Postmortems

Every sev-1 requires a written postmortem within five working days. Sev-2 incidents require one
only when they recur within a quarter, because a single degraded afternoon rarely teaches
anything a recurring one does not teach better.

Postmortems are blameless by policy. The purpose is a change to the system, not a change to
whoever was on call, and a review that produces the second outcome will not produce the first.
