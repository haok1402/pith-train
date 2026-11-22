# e2efp8-gate handoff

The live failure was `StopIteration` in `wait_for_slot()` because a launcher removed another live launcher's ticket. Both launchers report the same hostname, while their PID namespaces can differ. One queue record showed PID 4 on the login host as seen from another launcher. Looking up that PID in a different namespace is therefore insufficient to judge whether the ticket is live. [Example failure](../logs/ppdense-submit-full-adamw-wikitext-pp2-mbs1-20261122T083342.log).

Proposed change for `bench/gpu_gate.py`:

1. While holding the existing queue lock, allocate the FIFO ticket, open a unique `ticket-<number>.lock` file, and acquire an exclusive `flock` on that file. Retain the open file descriptor in the launcher until admission or exit.
2. Record leases with a sentinel owner such as `"flock"` in the second element of the existing four-element queue record. Previously launched wrappers compare that element to their hostname, so they preserve newer records during migration.
3. While holding the queue lock, prune leased records by trying a nonblocking exclusive `flock` from a separate file descriptor. A busy lease is active. An acquired or vanished lease is stale. Leave old-format records for their original launchers to clean up during migration. Avoid deleting a live foreign-namespace record based on `/proc`.
4. Preserve the existing ticket on a cap rejection. Remove the row and lease on confirmed Slurm acceptance or on launcher exit. The current shared `next_poll` remains the global 15-second `squeue` throttle.
5. Test mixed old/new queue records, concurrent launchers with distinct PID namespaces, an abandoned lease, and two accepted commands that overlap after FIFO admission.

No implementation change was applied here. This file is the handoff.
