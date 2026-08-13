#!/bin/bash
cd "$(dirname "$0")"

if [ -z "$(docker ps -q -f name=^grobid$)" ]; then
  if [ -n "$(docker ps -aq -f name=^grobid$)" ]; then
    echo "Starting existing GROBID container..."
    docker start grobid
  else
    echo "Creating GROBID container..."
    docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
  fi
  echo "Waiting for GROBID to become ready..."
  for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8070/api/isalive)
    if [ "$code" = "200" ]; then break; fi
    sleep 2
  done
fi

source .venv/bin/activate
streamlit run app.py
