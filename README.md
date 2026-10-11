# Clonador de Voz (RVC local)

App web local, en 3 pasos, para entrenar un modelo de voz y clonar audios con él usando [RVC-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) como motor de IA.

1. **Entrenar**: sube uno o varios `.wav`/`.mp3` con la voz a clonar.
2. **Clonar**: elige el modelo entrenado y el audio a convertir.
3. **Descargar**: guarda el resultado en `.wav` donde quieras.

Backend en Flask + frontend en HTML/CSS/JS vanilla. Todo corre en local, sin enviar nada a internet.

## Requisitos

- macOS (probado) con Python 3.9+.
- [RVC-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) instalado aparte, con su propio entorno virtual (Python 3.10) y los modelos preentrenados descargados (`assets/hubert`, `assets/pretrained_v2`, `assets/rmvpe`).
- `ffmpeg` instalado.

## Configuración

Las rutas a RVC-WebUI y a ffmpeg están al principio de [`app.py`](app.py) — ajústalas a tu instalación antes de arrancar:

```python
RVC_DIR = Path("/ruta/a/tu/RVC-WebUI")
FFMPEG = "/ruta/a/tu/ffmpeg"
```

> Nota: esta app invoca las funciones de entrenamiento/inferencia de RVC-WebUI directamente en un subproceso (no por su API HTTP), porque la API de la versión usada no es fiable para procesos largos tipo generador. Revisa `rvc_train_runner.py` y `rvc_infer_runner.py` (se instalan dentro de la carpeta de RVC-WebUI) si cambias de versión de RVC.

## Instalación

```bash
cd app
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Arrancar

```bash
./start.sh
```

Esto levanta el servidor de RVC-WebUI (si no está corriendo) y la app, y abre `http://127.0.0.1:5050` en el navegador. También puedes generar un icono de app de macOS con Automator/AppleScript que llame a `start.sh`.

## Estructura

```
app.py              Backend Flask (entrenamiento, conversión, gestión de modelos)
templates/           HTML
static/              CSS y JS del frontend
modelos/              Modelos .pth entrenados (no se suben a git)
data/                 Datasets, audios subidos y resultados (no se sube a git)
```

## Notas

- Los modelos entrenados (`.pth`) y los audios de usuario **no están en este repositorio** por tamaño — se generan localmente al usar la app.
- Borrar un modelo desde la interfaz lo mueve a `modelos/.papelera/` en vez de eliminarlo para siempre.
