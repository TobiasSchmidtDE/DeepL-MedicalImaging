#!/usr/bin/env bash
# Author: Tobias
echo "URLs open for the following services:"
echo $(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(str([(tunnel['name'],tunnel['public_url']) for tunnel in json.load(sys.stdin)['tunnels']]))")
