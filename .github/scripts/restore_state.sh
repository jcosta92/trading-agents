#!/usr/bin/env bash
# Usage: restore_state.sh path [path ...]
# Extracts the given folders from the `bot-state` branch into the working tree (without touching
# git's index on main). Missing branch or paths are not an error: the bots then start fresh.
set -u
if git fetch -q --depth 1 origin bot-state 2>/dev/null; then
  for p in "$@"; do
    if git cat-file -e "FETCH_HEAD:$p" 2>/dev/null; then
      rm -rf "$p"
      git archive FETCH_HEAD "$p" | tar -x
      echo "restored $p from bot-state"
    else
      echo "bot-state has no $p"
    fi
  done
else
  echo "no bot-state branch yet"
fi
