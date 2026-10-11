#!/bin/sh
# Arranca el servidor RVC (si no está ya corriendo) y luego la app de clonación de voz.
cd "$(dirname "$0")"

RVC_DIR="${RVC_DIR:-/Users/franantartico/RVC-WebUI}"

if ! curl -s -o /dev/null -m 2 http://localhost:7865; then
  echo "Iniciando servidor RVC..."
  (
    cd "$RVC_DIR"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    nohup ./venv/bin/python infer-web.py > /tmp/rvc_server.log 2>&1 &
  )
  echo "Esperando a que RVC arranque..."
  for i in $(seq 1 60); do
    curl -s -o /dev/null -m 2 http://localhost:7865 && break
    sleep 2
  done
fi

echo "Iniciando la app de clonación de voz en http://127.0.0.1:5050 ..."
./venv/bin/python app.py
