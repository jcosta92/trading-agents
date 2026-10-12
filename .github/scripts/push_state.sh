#!/usr/bin/env bash
# Usage: push_state.sh "<message>" mirror:<src>:<dst> copy:<src>:<dst> ...
#
# Publishes the bots' state and the dashboard pages to the branch `bot-state`, which always holds
# exactly ONE commit (amended on every update). ~900 ledgers change every 15 minutes; keeping them
# on main would grow the repository by gigabytes over the test. GitHub Pages serves bot-state:/docs.
#   mirror:  the destination becomes an exact copy of the source folder (deletions included)
#   copy:    files are added/overwritten, others already on the branch are kept (e.g. preview.html)
# Concurrent writers (the trading loop and the preview job) are reconciled by fetching the branch
# before every attempt and pushing with --force-with-lease.
set -u
msg="$1"; shift
root="$(pwd)"
wt="${RUNNER_TEMP:-/tmp}/bot-state-wt"
url="$(git config --get remote.origin.url)"
hdr="$(git config --get http.https://github.com/.extraheader || true)"
if [ ! -d "$wt/.git" ]; then
  rm -rf "$wt"; mkdir -p "$wt"
  git -C "$wt" init -q
  git -C "$wt" remote add origin "$url"
  [ -n "$hdr" ] && git -C "$wt" config http.https://github.com/.extraheader "$hdr"
fi
git -C "$wt" config user.name "github-actions[bot]"
git -C "$wt" config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in 1 2 3 4 5 6; do
  lease=""
  if git -C "$wt" fetch -q --depth 1 origin bot-state 2>/dev/null; then
    lease="$(git -C "$wt" rev-parse FETCH_HEAD)"
    git -C "$wt" checkout -q -B bot-state FETCH_HEAD
    git -C "$wt" reset -q --hard FETCH_HEAD
  else
    git -C "$wt" checkout -q --orphan bot-state 2>/dev/null || true
  fi
  for spec in "$@"; do
    mode="${spec%%:*}"; rest="${spec#*:}"; src="${rest%%:*}"; dst="${rest#*:}"
    [ -e "$root/$src" ] || continue
    mkdir -p "$wt/$(dirname "$dst")"
    if [ -d "$root/$src" ]; then
      mkdir -p "$wt/$dst"
      if [ "$mode" = "mirror" ]; then rsync -a --delete "$root/$src/" "$wt/$dst/"; else rsync -a "$root/$src/" "$wt/$dst/"; fi
    else
      cp -f "$root/$src" "$wt/$dst"
    fi
  done
  find "$wt" -path "$wt/.git" -prune -o -type f -size +95M -print -delete
  git -C "$wt" add -A
  if git -C "$wt" rev-parse -q --verify HEAD >/dev/null; then
    git -C "$wt" commit -q --amend --allow-empty -m "$msg"
  else
    git -C "$wt" commit -q --allow-empty -m "$msg"
  fi
  if [ -n "$lease" ]; then
    git -C "$wt" push -q --force-with-lease="bot-state:$lease" origin HEAD:bot-state && { echo "state pushed"; exit 0; }
  else
    git -C "$wt" push -q origin HEAD:bot-state && { echo "state pushed (new branch)"; exit 0; }
  fi
  echo "push attempt $attempt rejected; retrying"
  sleep $((attempt * 4))
done
echo "::error::could not push bot-state after 6 attempts"
exit 1
