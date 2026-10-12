#!/usr/bin/env bash
# Usage: annotate_tail.sh <logfile> [lines]
# Publishes the end of a log as a workflow error annotation, so a failure can be read through the
# GitHub API (check-run annotations) even where raw log downloads are not reachable.
f="$1"; n="${2:-40}"
[ -f "$f" ] || { echo "::error::no log file $f"; exit 0; }
body=$(tail -n "$n" "$f" | cut -c1-300 | sed -e 's/%/%25/g' | awk 'BEGIN{ORS="%0A"} {print}')
echo "::error title=tail of $(basename "$f")::${body}"
