#!/usr/bin/env bash
# Headless paper IB Gateway via IBC. Version directory is auto-detected.
set -u
V=$(ls ~/Jts/ibgateway | sort -n | tail -1)
exec /opt/ibc/scripts/ibcstart.sh "$V" --gateway \
  --tws-path="$HOME/Jts" --tws-settings-path="$HOME/Jts" \
  --ibc-path=/opt/ibc --ibc-ini="$HOME/ibc/config.ini" \
  --mode=paper
