# Fernwick Relay release handbook

This handbook describes how Fernwick Relay releases are delivered and what changed in version 2. It is fictional evaluation material.

## Release channels

Fernwick ships two channels. The stable channel receives a release every four weeks. The preview channel receives a release every week and is supported for testing only. Switch channels with `fernwick channel set preview` or `fernwick channel set stable`.

## Version 2 changes

Version 2 replaces the legacy `/v1/hooks` endpoint with `/v2/deliveries`. The legacy endpoint stops accepting requests on 1 March 2027. Version 2 also renames the `retry_limit` setting to `max_attempts`; the old name is still read until version 3.

## Upgrade steps

1. Upgrade the agent to 2.0 or later.
2. Replace calls to `/v1/hooks` with `/v2/deliveries`.
3. Rename `retry_limit` to `max_attempts` in `agent.toml`.
4. Run `fernwick doctor` to confirm that every destination is reachable.
