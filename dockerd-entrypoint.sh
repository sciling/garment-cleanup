#!/bin/bash
set -e

if [[ "$1" == "serve" ]]; then
    shift 1
    python server.py
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
