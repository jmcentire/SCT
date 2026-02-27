# Directive: Four-Part Actionable Finding Structure

## Problem

The correlator currently produces findings with a single `summary` field — a description of the pattern found. In practice, these summaries get acknowledged and not acted on. A busy tech lead reads "systemic availability sync issue" and files it under "known problems." The output format causes the exact decay path the system's own theory formalizes: converting second-order ignorance to awareness without producing the conditions for action.

## Requirement

Every correlator finding must produce four parts:

1. **summary** (exists): What was found. This is the current output. No change needed to this field.

2. **unknowns** (new): Specific unanswered questions that require investigation. These complete the conversion from second-order ignorance to first-order ignorance — the paper's entire value proposition. Not "more research needed" but "Do INT-332 and INT-543 share a common sync code path?" or "How many other SAAS orgs have listed auto-onboarded properties with the same payout misconfiguration?" 1-3 concrete questions that a specific person could answer.

3. **recommended_action** (new): An opinionated, specific, assignable next step. "Create a parent ticket linking these 8 sync bugs and assign one person to triage root cause" or "Disable price matching immediately — the tax calculation is known broken." If the correlator is confident enough to report a pattern, it's confident enough to recommend what to do about it.

4. **inaction_risk** (new): Concrete stakes of doing nothing, using evidence from the signals. Not "this could cause problems" but "$13K in owner payouts already failed to reach Stripe" or "A double-booking went undetected for 2 months; 8 open sync bugs across 5 vendors suggest more are accumulating." Numbers, ticket references, and dollar amounts when they appear in the signals.

The quality test changes from "would a tech lead say 'I didn't know that'?" to "would a tech lead know exactly what to do Monday morning after reading this?"

## What to change

- Add the three new fields to the correlator's output schema. All three are required, not optional.
- Update the correlator's system prompt to instruct it to produce all four parts, with the guidance above on what good output looks like for each field.
- Ensure the new fields flow through to the annotation store and the persisted annotations JSON so they appear in the output humans read.
- Update tests to account for the new fields.

## Validation

All tests pass. Then `stigmergy run --once` and review `.stigmergy/annotations.json`. Each finding should have all four parts. Read one and determine whether you know what to do about it without inferring anything. If you have to think about the action, the prompt needs tightening.
