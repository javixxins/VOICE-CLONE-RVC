const $ = (id) => document.getElementById(id);

if (location.protocol === "file:") {
  $("fileProtocolWarning").classList.remove("hidden");
}

function toast(msg, ok = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (ok ? " ok" : "");
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 5000);
}

// ---- Navegación entre pasos ----
let maxStepReached = 1;
document.querySelectorAll(".step").forEach((tab) => {
  tab.addEventListener("click", () => goToStep(tab.dataset.step));
});
function goToStep(n, preferredModel) {
  n = String(n);
  document.querySelectorAll(".step").forEach((t) => {
    const s = Number(t.dataset.step);
    t.classList.toggle("active", t.dataset.step === n);
    t.classList.toggle("done", s < Number(n) || (s < maxStepReached && s !== Number(n)));
  });
  document.querySelectorAll(".card").forEach((p) => p.classList.toggle("hidden", p.id !== `panel-${n}`));
  maxStepReached = Math.max(maxStepReached, Number(n));
  if (n === "2") loadModels(preferredModel);
}
goToStep(1);
loadModels(); // precarga la lista (p.ej. modelos ya entrenados en sesiones anteriores)

// ---- Estado servidor RVC ----
fetch("/api/health").then(r => r.json()).then(d => {
  const pill = $("rvcStatus");
  if (d.ok) {
    pill.textContent = "RVC conectado";
    pill.classList.add("ok");
  } else {
    pill.textContent = "RVC no responde";
    pill.classList.add("bad");
  }
}).catch(() => {
  const pill = $("rvcStatus");
  pill.textContent = "RVC no responde";
  pill.classList.add("bad");
});

// ---- Utilidad: dropzone genérica ----
function setupDropzone(zoneEl, inputEl, onFiles) {
  zoneEl.addEventListener("click", () => inputEl.click());
  inputEl.addEventListener("change", (e) => onFiles(Array.from(e.target.files || [])));

  ["dragenter", "dragover"].forEach(ev => zoneEl.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    zoneEl.classList.add("drag");
  }));
  ["dragleave", "dragend"].forEach(ev => zoneEl.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    zoneEl.classList.remove("drag");
  }));
  zoneEl.addEventListener("drop", (e) => {
    e.preventDefault(); e.stopPropagation();
    zoneEl.classList.remove("drag");
    const files = Array.from(e.dataTransfer.files || []);
    onFiles(files);
  });
}

function isAudioFile(f) {
  return /\.(wav|mp3)$/i.test(f.name);
}

// ---- Paso 1: entrenamiento ----
let trainFiles = [];

setupDropzone($("dropzoneTrain"), $("trainFilesInput"), (files) => {
  const audios = files.filter(isAudioFile);
  if (audios.length === 0) {
    toast("Ningún archivo .wav/.mp3 válido en esa selección.");
    return;
  }
  trainFiles = audios;
  renderTrainFiles();
});

function renderTrainFiles() {
  const ul = $("trainFileList");
  ul.innerHTML = "";
  trainFiles.forEach(f => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${f.name}</span><span class="size">${(f.size / 1024 / 1024).toFixed(1)} MB</span>`;
    ul.appendChild(li);
  });
  $("btnTrain").disabled = trainFiles.length === 0;
}

function formatElapsed(sec) {
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

let trainPollTimer = null;
function stopTrainPolling() {
  if (trainPollTimer) clearInterval(trainPollTimer);
  trainPollTimer = null;
}

$("btnTrain").addEventListener("click", async () => {
  const modelName = $("modelName").value.trim();
  if (!modelName) { toast("Ponle un nombre al modelo."); return; }
  if (trainFiles.length === 0) { toast("Sube al menos un audio."); return; }

  const fd = new FormData();
  fd.append("model_name", modelName);
  fd.append("epochs", $("epochs").value);
  trainFiles.forEach(f => fd.append("files", f, f.name));

  $("btnTrain").disabled = true;
  $("trainProgress").classList.remove("hidden");
  $("trainBar").style.width = "0%";
  $("trainPercent").textContent = "0%";
  $("trainEta").textContent = "⏳ estimando tiempo…";
  $("trainLog").classList.remove("hidden");
  $("trainLog").textContent = "Subiendo audios…";
  try {
    const r = await fetch("/api/train", { method: "POST", body: fd });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "Error desconocido");
    toast(`Entrenamiento iniciado con ${d.n_files} audio(s).`, true);
    pollTrainStatus(modelName);
  } catch (e) {
    $("trainLog").textContent = "Error: " + e.message;
    toast("Error al entrenar: " + e.message);
    $("btnTrain").disabled = false;
    $("trainProgress").classList.add("hidden");
  }
});

function pollTrainStatus(modelName) {
  stopTrainPolling();
  const check = async () => {
    try {
      const r = await fetch(`/api/train/status?model=${encodeURIComponent(modelName)}`);
      const d = await r.json();
      if (!d.ok) throw new Error(d.error || "Error consultando estado");

      const t = formatElapsed(d.elapsed_seconds);
      $("trainLog").textContent = `⏱ ${t} · ${d.step_label}\n\n${d.log || ""}`;
      $("trainBar").style.width = `${d.percent}%`;
      $("trainPercent").textContent = `${Math.round(d.percent)}%`;
      $("trainEta").textContent = d.eta_seconds != null
        ? `⏳ quedan ~${formatElapsed(d.eta_seconds)}`
        : "⏳ estimando tiempo…";

      if (d.status === "done") {
        stopTrainPolling();
        $("btnTrain").disabled = false;
        $("trainProgress").classList.add("hidden");
        toast(`Modelo '${modelName}' entrenado en ${t}.`, true);
        goToStep(2, modelName);
      } else if (d.status === "error") {
        stopTrainPolling();
        $("btnTrain").disabled = false;
        $("trainProgress").classList.add("hidden");
        $("trainLog").textContent += `\n\n❌ Error: ${d.error}`;
        toast("Error al entrenar: " + d.error);
      }
    } catch (e) {
      // red hiccup puntual: seguimos intentando en el próximo tick
    }
  };
  check();
  trainPollTimer = setInterval(check, 2000);
}

// ---- Paso 2: clonar ----
const LAST_MODEL_KEY = "clonadorVoz.lastModel";

async function loadModels(preferredModel) {
  const sel = $("modelSelect");
  sel.innerHTML = "<option>Cargando…</option>";
  try {
    const r = await fetch("/api/models");
    const d = await r.json();
    sel.innerHTML = "";
    if (!d.models || d.models.length === 0) {
      sel.innerHTML = "<option value=''>-- sin modelos entrenados --</option>";
      $("btnDeleteModel").disabled = true;
      return;
    }
    d.models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.replace(/\.(pth)$/i, "");
      opt.textContent = m;
      sel.appendChild(opt);
    });
    $("btnDeleteModel").disabled = false;

    // Selecciona el modelo pedido explícitamente, si no el último usado, si no el primero.
    const wanted = preferredModel || localStorage.getItem(LAST_MODEL_KEY);
    if (wanted && [...sel.options].some(o => o.value === wanted)) {
      sel.value = wanted;
    }
    localStorage.setItem(LAST_MODEL_KEY, sel.value);
  } catch (e) {
    sel.innerHTML = "<option value=''>Error cargando modelos</option>";
    $("btnDeleteModel").disabled = true;
    toast("No se pudo consultar la lista de modelos.");
  }
}
$("modelSelect").addEventListener("change", () => {
  localStorage.setItem(LAST_MODEL_KEY, $("modelSelect").value);
});
$("btnRefreshModels").addEventListener("click", () => loadModels());

// ---- Importar modelos ya entrenados desde una carpeta cualquiera ----
$("btnImportModels").addEventListener("click", () => $("importModelsInput").click());
$("importModelsInput").addEventListener("change", async (e) => {
  const pthFiles = Array.from(e.target.files || []).filter(f => /\.pth$/i.test(f.name));
  if (pthFiles.length === 0) {
    toast("No se encontró ningún archivo .pth en esa carpeta.");
    return;
  }
  const fd = new FormData();
  pthFiles.forEach(f => fd.append("files", f, f.name));

  $("btnImportModels").disabled = true;
  $("btnImportModels").textContent = "📂 Importando…";
  try {
    const r = await fetch("/api/models/import", { method: "POST", body: fd });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "Error desconocido");
    toast(`Importado(s) ${d.imported.length} modelo(s): ${d.imported.join(", ")}`, true);
    const firstImported = d.imported[0].replace(/\.(pth)$/i, "");
    goToStep(2, firstImported);
  } catch (e) {
    toast("Error al importar: " + e.message);
  } finally {
    $("btnImportModels").disabled = false;
    $("btnImportModels").textContent = "📂 Elegir modelo(s) .pth ya entrenados";
    $("importModelsInput").value = "";
  }
});

$("btnDeleteModel").addEventListener("click", async () => {
  const modelName = $("modelSelect").value;
  if (!modelName) { toast("No hay modelo seleccionado."); return; }
  if (!confirm(`¿Quitar el modelo '${modelName}' de la lista? Se moverá a la papelera (modelos/.papelera) por si lo necesitas recuperar.`)) return;

  $("btnDeleteModel").disabled = true;
  try {
    const r = await fetch(`/api/models/${encodeURIComponent(modelName)}`, { method: "DELETE" });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "Error desconocido");
    toast(`Modelo '${modelName}' borrado.`, true);
    await loadModels();
  } catch (e) {
    toast("Error al borrar: " + e.message);
    $("btnDeleteModel").disabled = false;
  }
});

let convertFile = null;
let convertAudioDuration = 8; // segundos, valor por defecto hasta que se detecte el real
setupDropzone($("dropzoneConvert"), $("convertFileInput"), (files) => {
  const audio = files.find(isAudioFile);
  if (!audio) {
    toast("Ese archivo no es .wav/.mp3.");
    return;
  }
  convertFile = audio;
  $("convertFileName").textContent = "📄 " + audio.name;
  $("btnConvert").disabled = false;

  const probe = new Audio();
  probe.preload = "metadata";
  probe.onloadedmetadata = () => {
    if (isFinite(probe.duration) && probe.duration > 0) convertAudioDuration = probe.duration;
    URL.revokeObjectURL(probe.src);
  };
  probe.src = URL.createObjectURL(audio);
});

$("btnConvert").addEventListener("click", async () => {
  const modelName = $("modelSelect").value;
  if (!modelName) { toast("No hay modelo seleccionado."); return; }
  if (!convertFile) { toast("Elige un audio a clonar."); return; }

  const fd = new FormData();
  fd.append("model_name", modelName);
  fd.append("pitch", $("pitch").value);
  fd.append("index_rate", $("indexRate").value);
  fd.append("file", convertFile, convertFile.name);

  // Estimación: carga del modelo (~5s) + procesado a ~2.5x la duración del audio.
  const estimatedSeconds = Math.max(6, 5 + convertAudioDuration * 2.5);

  $("btnConvert").disabled = true;
  $("convertProgress").classList.remove("hidden");
  $("convertBar").style.width = "0%";
  $("convertPercent").textContent = "0%";
  $("convertLog").classList.remove("hidden");
  const startedAt = Date.now();
  $("convertLog").textContent = "⏱ 0:00 · Clonando audio…";
  const tick = setInterval(() => {
    const elapsed = (Date.now() - startedAt) / 1000;
    const pct = Math.min(95, Math.round((elapsed / estimatedSeconds) * 100));
    $("convertBar").style.width = `${pct}%`;
    $("convertPercent").textContent = `${pct}%`;
    const remaining = Math.max(0, Math.round(estimatedSeconds - elapsed));
    $("convertEta").textContent = remaining > 0 ? `⏳ quedan ~${formatElapsed(remaining)}` : "⏳ casi listo…";
    $("convertLog").textContent = `⏱ ${formatElapsed(Math.floor(elapsed))} · Clonando audio…`;
  }, 500);
  try {
    const r = await fetch("/api/convert", { method: "POST", body: fd });
    const d = await r.json();
    clearInterval(tick);
    $("convertBar").style.width = "100%";
    $("convertPercent").textContent = "100%";
    $("convertEta").textContent = "✅ listo";
    const t = formatElapsed(Math.floor((Date.now() - startedAt) / 1000));
    if (!d.ok) throw new Error(d.error || "Error desconocido");
    $("convertLog").textContent = `⏱ ${t}\n${d.info || "Listo."}`;
    showResult(d.download_url);
    toast(`Audio clonado en ${t}.`, true);
    goToStep(3);
  } catch (e) {
    clearInterval(tick);
    $("convertLog").textContent = "Error: " + e.message;
    toast("Error al clonar: " + e.message);
  } finally {
    $("btnConvert").disabled = false;
    $("convertProgress").classList.add("hidden");
  }
});

// ---- Paso 3: descargar ----
let lastDownloadUrl = null;
function showResult(url) {
  lastDownloadUrl = url;
  $("resultBox").classList.add("hidden");
  const player = $("resultPlayer");
  player.src = url;
  player.classList.remove("hidden");
  $("btnSaveAs").disabled = false;
  const a = $("fallbackDownload");
  a.href = url;
  a.classList.remove("hidden");
}

$("btnSaveAs").addEventListener("click", async () => {
  if (!lastDownloadUrl) return;
  const resp = await fetch(lastDownloadUrl);
  const blob = await resp.blob();
  const suggestedName = lastDownloadUrl.split("/").pop();

  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName,
        types: [{ description: "Audio WAV", accept: { "audio/wav": [".wav"] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      toast("Guardado correctamente.", true);
      return;
    } catch (e) {
      if (e.name === "AbortError") return;
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = suggestedName;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
});
