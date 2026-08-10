const APP_VERSION = "0.2.3";
const OCR_SDK_VERSION = "0.4.2";
const MODEL_NAME = "PP-OCRv5_mobile";
const MAX_FILES = 12;
const ORT_WASM_PATHS =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
const MODEL_BASE =
  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/";
const RUN_MARKER_KEY = "dekiru-cards-ocr-running";
const IS_IOS =
  /iPhone|iPad|iPod/i.test(navigator.userAgent) ||
  new URLSearchParams(location.search).get("ios") === "1";

function writeRunMarker(phase, detail = "") {
  try {
    sessionStorage.setItem(
      RUN_MARKER_KEY,
      JSON.stringify({ phase, detail, at: new Date().toISOString() }),
    );
  } catch (_storageError) {
    // OCR can continue even if session storage is unavailable.
  }
}

function readRunMarker() {
  try {
    const raw = sessionStorage.getItem(RUN_MARKER_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(RUN_MARKER_KEY);
    try {
      return JSON.parse(raw);
    } catch (_parseError) {
      return { phase: "unknown", detail: "" };
    }
  } catch (_storageError) {
    return null;
  }
}

function clearRunMarker() {
  try {
    sessionStorage.removeItem(RUN_MARKER_KEY);
  } catch (_storageError) {
    // Result display is unaffected.
  }
}

function createPipelineConfig() {
  const detName = `${MODEL_NAME}_det`;
  const recName = `${MODEL_NAME}_rec`;
  return {
    pipelineName: "OCR",
    raw: {
      pipeline_name: "OCR",
      use_doc_preprocessor: false,
      use_textline_orientation: false,
      SubModules: {
        TextDetection: {
          model_name: detName,
          limit_side_len: 64,
          limit_type: "min",
          max_side_limit: 4000,
          thresh: 0.3,
          box_thresh: 0.6,
          unclip_ratio: 1.5,
        },
        TextRecognition: {
          model_name: recName,
          batch_size: 1,
          score_thresh: 0,
        },
      },
    },
    warnings: [],
    unsupportedFeatures: [],
    modelSelection: {
      textDetectionModelName: detName,
      textRecognitionModelName: recName,
    },
    assets: {
      det: { url: `${MODEL_BASE}${detName}_onnx_infer.tar` },
      rec: { url: `${MODEL_BASE}${recName}_onnx_infer.tar` },
    },
    runtimeDefaults: {
      text_det_limit_side_len: 64,
      text_det_limit_type: "min",
      text_det_max_side_limit: 4000,
      text_det_thresh: 0.3,
      text_det_box_thresh: 0.6,
      text_det_unclip_ratio: 1.5,
      text_rec_score_thresh: 0,
    },
    pipelineBatchSize: 1,
    textDetectionBatchSize: 1,
    textRecognitionBatchSize: 1,
  };
}

class OcrWorkerClient {
  constructor() {
    this.worker = null;
    this.pending = new Map();
    this.nextRequestId = 1;
  }

  ensureWorker() {
    if (this.worker) return this.worker;
    const worker = new Worker(
      new URL("./worker-entry-C9UNuyOJ.js", import.meta.url),
      { type: "module" },
    );
    worker.addEventListener("message", (event) => {
      const message = event.data;
      if (!message || message.kind !== "worker-transport-response") return;
      const pending = this.pending.get(message.requestId);
      if (!pending) return;
      this.pending.delete(message.requestId);
      if (message.status === "success") pending.resolve(message.payload);
      else {
        const error = new Error(message.error?.message || "OCR処理に失敗しました。");
        error.name = message.error?.name || "Error";
        if (message.error?.stack) error.stack = message.error.stack;
        pending.reject(error);
      }
    });
    const rejectAll = (event) => {
      const error = new Error(event?.message || "OCRのバックグラウンド処理が停止しました。");
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
      worker.terminate();
      if (this.worker === worker) this.worker = null;
    };
    worker.addEventListener("error", rejectAll);
    worker.addEventListener("messageerror", rejectAll);
    this.worker = worker;
    return worker;
  }

  request(type, payload, transferables = []) {
    const worker = this.ensureWorker();
    const requestId = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      try {
        worker.postMessage(
          { kind: "worker-transport-request", type, payload, requestId },
          transferables,
        );
      } catch (error) {
        this.pending.delete(requestId);
        reject(error);
      }
    });
  }

  async initialize() {
    const response = await this.request("init", {
      options: {
        pipelineConfig: createPipelineConfig(),
        ortOptions: {
          backend: "wasm",
          wasmPaths: ORT_WASM_PATHS,
          numThreads: 1,
          simd: true,
          proxy: false,
        },
      },
    });
    return response.summary;
  }

  async predict(blob, params) {
    if (typeof createImageBitmap !== "function") {
      throw new Error("このブラウザはバックグラウンド画像処理に対応していません。");
    }
    const bitmap = await createImageBitmap(blob);
    try {
      return await this.request(
        "predict",
        {
          sources: [{ kind: "imageBitmap", imageBitmap: bitmap }],
          params,
        },
        [bitmap],
      );
    } catch (error) {
      try {
        bitmap.close();
      } catch (_closeError) {
        // 転送後は既に閉じられている場合があります。
      }
      throw error;
    }
  }

  async dispose() {
    if (!this.worker) return;
    try {
      await this.request("dispose", {});
    } finally {
      this.worker?.terminate();
      this.worker = null;
    }
  }
}

const state = {
  pages: [],
  engine: null,
  engineReady: false,
  engineSummary: null,
  engineMode: "worker",
  running: false,
};

const $ = (id) => document.getElementById(id);

const ui = {
  libraryInput: $("libraryInput"),
  cameraInput: $("cameraInput"),
  libraryButton: $("libraryButton"),
  cameraButton: $("cameraButton"),
  startButton: $("startButton"),
  clearButton: $("clearButton"),
  selectedSection: $("selectedSection"),
  selectedCount: $("selectedCount"),
  selectedList: $("selectedList"),
  statusPanel: $("statusPanel"),
  statusTitle: $("statusTitle"),
  statusDetail: $("statusDetail"),
  progressBar: $("progressBar"),
  progressText: $("progressText"),
  httpWarning: $("httpWarning"),
  resultSection: $("resultSection"),
  summaryGrid: $("summaryGrid"),
  results: $("results"),
  copyJsonButton: $("copyJsonButton"),
  downloadJsonButton: $("downloadJsonButton"),
  toggleBoxes: $("toggleBoxes"),
  maxSide: $("maxSide"),
  detThresh: $("detThresh"),
  boxThresh: $("boxThresh"),
  unclipRatio: $("unclipRatio"),
  recScoreThresh: $("recScoreThresh"),
  retryEngineButton: $("retryEngineButton"),
};

function makeId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  return `page-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatMs(ms) {
  if (!Number.isFinite(ms)) return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}秒`;
  return `${Math.round(ms)}ms`;
}

function formatSize(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
  return `${Math.max(1, Math.round(bytes / 1024))}KB`;
}

function setStatus(title, detail, progress = 0, kind = "normal") {
  ui.statusPanel.hidden = false;
  ui.statusPanel.dataset.kind = kind;
  ui.statusTitle.textContent = title;
  ui.statusDetail.textContent = detail || "";
  const safeProgress = Math.max(0, Math.min(100, Number(progress) || 0));
  ui.progressBar.style.width = `${safeProgress}%`;
  ui.progressText.textContent = `${Math.round(safeProgress)}%`;
}

function setControlsDisabled(disabled) {
  state.running = disabled;
  ui.libraryButton.disabled = disabled;
  ui.cameraButton.disabled = disabled;
  ui.startButton.disabled = disabled || state.pages.length === 0;
  ui.clearButton.disabled = disabled || state.pages.length === 0;
  ui.maxSide.disabled = disabled;
  ui.detThresh.disabled = disabled;
  ui.boxThresh.disabled = disabled;
  ui.unclipRatio.disabled = disabled;
  ui.recScoreThresh.disabled = disabled;
}

function isHeic(file) {
  const name = (file?.name || "").toLowerCase();
  const type = (file?.type || "").toLowerCase();
  return (
    type.includes("heic") ||
    type.includes("heif") ||
    name.endsWith(".heic") ||
    name.endsWith(".heif")
  );
}

async function imageElementFromBlob(blob) {
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.decoding = "async";
  image.src = url;
  try {
    if (image.decode) {
      await image.decode();
    } else {
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
      });
    }
    if (!image.naturalWidth || !image.naturalHeight) {
      throw new Error("画像の大きさを取得できませんでした。");
    }
    return {
      drawable: image,
      width: image.naturalWidth,
      height: image.naturalHeight,
      close: () => URL.revokeObjectURL(url),
    };
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
}

async function decodeBlob(blob) {
  if (typeof createImageBitmap === "function") {
    try {
      const bitmap = await createImageBitmap(blob, {
        imageOrientation: "from-image",
      });
      return {
        drawable: bitmap,
        width: bitmap.width,
        height: bitmap.height,
        close: () => bitmap.close(),
      };
    } catch (_error) {
      // Safariでは形式によってImage要素の方が安定するため、次を試します。
    }
  }
  return imageElementFromBlob(blob);
}

function canvasToBlob(canvas, type = "image/jpeg", quality = 0.94) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("画像の変換に失敗しました。"));
      },
      type,
      quality,
    );
  });
}

async function convertHeic(blob) {
  const helper = await import("./heic-helper.js");
  return helper.convertHeicBlob(blob);
}

async function makeOcrTiles(canvas) {
  if (!IS_IOS) {
    return [
      {
        blob: await canvasToBlob(canvas),
        x: 0,
        y: 0,
        width: canvas.width,
        height: canvas.height,
      },
    ];
  }

  const tileCount = 3;
  const overlap = Math.max(64, Math.round(Math.min(canvas.width, canvas.height) * 0.07));
  const splitVertically = canvas.height >= canvas.width;
  const longSide = splitVertically ? canvas.height : canvas.width;
  const baseSize = Math.ceil(longSide / tileCount);
  const tiles = [];

  for (let index = 0; index < tileCount; index += 1) {
    const coreStart = index * baseSize;
    if (coreStart >= longSide) break;
    const coreEnd = Math.min(longSide, (index + 1) * baseSize);
    const start = Math.max(0, coreStart - (index > 0 ? overlap : 0));
    const end = Math.min(longSide, coreEnd + (index < tileCount - 1 ? overlap : 0));
    const x = splitVertically ? 0 : start;
    const y = splitVertically ? start : 0;
    const width = splitVertically ? canvas.width : end - start;
    const height = splitVertically ? end - start : canvas.height;
    const tileCanvas = document.createElement("canvas");
    tileCanvas.width = width;
    tileCanvas.height = height;
    const context = tileCanvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("画像を分割する領域を作成できませんでした。");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    context.drawImage(canvas, x, y, width, height, 0, 0, width, height);
    const blob = await canvasToBlob(tileCanvas);
    tileCanvas.width = 1;
    tileCanvas.height = 1;
    tiles.push({ blob, x, y, width, height });
  }
  return tiles;
}

async function normalizeImage(file) {
  let sourceBlob = file;
  let convertedFromHeic = false;
  let decoded;

  try {
    decoded = await decodeBlob(sourceBlob);
  } catch (nativeError) {
    if (!isHeic(file)) throw nativeError;
    sourceBlob = await convertHeic(file);
    convertedFromHeic = true;
    decoded = await decodeBlob(sourceBlob);
  }

  const maxSide = Number(ui.maxSide.value) || 2600;
  const scale = Math.min(1, maxSide / Math.max(decoded.width, decoded.height));
  const width = Math.max(1, Math.round(decoded.width * scale));
  const height = Math.max(1, Math.round(decoded.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) {
    decoded.close();
    throw new Error("画像を処理する領域を作成できませんでした。");
  }
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(decoded.drawable, 0, 0, width, height);
  decoded.close();

  const normalizedBlob = await canvasToBlob(canvas);
  const tiles = await makeOcrTiles(canvas);
  canvas.width = 1;
  canvas.height = 1;
  return {
    blob: normalizedBlob,
    tiles,
    width,
    height,
    originalWidth: decoded.width,
    originalHeight: decoded.height,
    convertedFromHeic,
  };
}

function nextPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

function pageLabel(index) {
  return `${index + 1}ページ目`;
}

function updateSelectedList() {
  ui.selectedCount.textContent = `${state.pages.length}枚`;
  ui.selectedSection.hidden = state.pages.length === 0;
  ui.selectedList.replaceChildren();

  state.pages.forEach((page, index) => {
    const item = document.createElement("article");
    item.className = "selected-item";

    const thumb = document.createElement("div");
    thumb.className = "selected-thumb";
    if (IS_IOS) {
      thumb.classList.add("is-unavailable");
      const fallback = document.createElement("span");
      fallback.textContent = isHeic(page.file) ? "HEIC" : "写真";
      thumb.append(fallback);
    } else {
      const image = document.createElement("img");
      image.src = page.previewUrl;
      image.alt = `${pageLabel(index)}のプレビュー`;
      image.addEventListener("error", () => {
        thumb.classList.add("is-unavailable");
        image.remove();
        const fallback = document.createElement("span");
        fallback.textContent = isHeic(page.file) ? "HEIC" : "画像";
        thumb.append(fallback);
      });
      thumb.append(image);
    }

    const info = document.createElement("div");
    info.className = "selected-info";
    const name = document.createElement("strong");
    name.textContent = page.file.name || pageLabel(index);
    const meta = document.createElement("span");
    meta.textContent = `${pageLabel(index)}・${formatSize(page.file.size)}`;
    info.append(name, meta);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button";
    remove.setAttribute("aria-label", `${page.file.name}を外す`);
    remove.textContent = "×";
    remove.addEventListener("click", () => removePage(page.id));

    item.append(thumb, info, remove);
    ui.selectedList.append(item);
  });
  setControlsDisabled(state.running);
}

function removePage(id) {
  if (state.running) return;
  const index = state.pages.findIndex((page) => page.id === id);
  if (index < 0) return;
  const [page] = state.pages.splice(index, 1);
  URL.revokeObjectURL(page.previewUrl);
  if (page.normalizedUrl) URL.revokeObjectURL(page.normalizedUrl);
  updateSelectedList();
  if (!state.pages.length) {
    ui.resultSection.hidden = true;
    ui.statusPanel.hidden = true;
  }
}

function addFiles(fileList) {
  const incoming = Array.from(fileList || []).filter((file) =>
    file.type.startsWith("image/") || isHeic(file),
  );
  const remaining = Math.max(0, MAX_FILES - state.pages.length);
  incoming.slice(0, remaining).forEach((file) => {
    state.pages.push({
      id: makeId(),
      file,
      previewUrl: URL.createObjectURL(file),
      normalizedUrl: null,
      normalizedBlob: null,
      result: null,
      error: null,
      wallMs: null,
      imageInfo: null,
    });
  });
  updateSelectedList();
  ui.resultSection.hidden = true;
  if (incoming.length > remaining) {
    setStatus(
      "追加できるのは12枚までです",
      `${remaining}枚を追加しました。残りは次の検証に分けてください。`,
      0,
      "warning",
    );
  }
}

async function ensureEngine() {
  if (state.engineReady && state.engine) return state.engine;
  setStatus(
    "文字認識の準備をしています",
    "初回は認識モデルを読み込むため、通信環境によって少し時間がかかります。",
    6,
  );

  if (state.engine) {
    try {
      await state.engine.dispose();
    } catch (_error) {
      // 破棄失敗は再初期化を妨げないため無視します。
    }
  }

  state.engine = new OcrWorkerClient();
  state.engineSummary = await state.engine.initialize();
  state.engineMode = "worker-lite";
  state.engineReady = true;
  return state.engine;
}

function getRecognitionParams() {
  return {
    textDetThresh: Number(ui.detThresh.value),
    textDetBoxThresh: Number(ui.boxThresh.value),
    textDetUnclipRatio: Number(ui.unclipRatio.value),
    textRecScoreThresh: Number(ui.recScoreThresh.value),
  };
}

function boundsFromPoly(poly) {
  const xs = poly.map((point) => point[0]);
  const ys = poly.map((point) => point[1]);
  const left = Math.min(...xs);
  const top = Math.min(...ys);
  const right = Math.max(...xs);
  const bottom = Math.max(...ys);
  return {
    x: Math.round(left),
    y: Math.round(top),
    width: Math.round(right - left),
    height: Math.round(bottom - top),
  };
}

function overlapRatio(a, b) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  if (!intersection) return 0;
  const smallerArea = Math.max(
    1,
    Math.min(a.width * a.height, b.width * b.height),
  );
  return intersection / smallerArea;
}

function mergeTileResults(tileResults, width, height) {
  const candidates = [];
  tileResults.forEach(({ tile, result }, tileIndex) => {
    for (const item of result.items || []) {
      candidates.push({
        ...item,
        poly: item.poly.map(([x, y]) => [x + tile.x, y + tile.y]),
        _tileIndex: tileIndex,
      });
    }
  });

  candidates.sort((a, b) => {
    const boxA = boundsFromPoly(a.poly);
    const boxB = boundsFromPoly(b.poly);
    return boxA.y - boxB.y || boxA.x - boxB.x;
  });

  const merged = [];
  for (const candidate of candidates) {
    const candidateBounds = boundsFromPoly(candidate.poly);
    const duplicateIndex = merged.findIndex((existing) => {
      if (existing._tileIndex === candidate._tileIndex) return false;
      if ((existing.text || "") !== (candidate.text || "")) return false;
      return overlapRatio(boundsFromPoly(existing.poly), candidateBounds) >= 0.55;
    });
    if (duplicateIndex < 0) {
      merged.push(candidate);
    } else if ((candidate.score || 0) > (merged[duplicateIndex].score || 0)) {
      merged[duplicateIndex] = candidate;
    }
  }

  const items = merged
    .map(({ _tileIndex, ...item }) => item)
    .sort((a, b) => {
      const boxA = boundsFromPoly(a.poly);
      const boxB = boundsFromPoly(b.poly);
      return boxA.y - boxB.y || boxA.x - boxB.x;
    });
  const metricTotals = tileResults.reduce(
    (totals, { result }) => {
      totals.detMs += result.metrics?.detMs || 0;
      totals.recMs += result.metrics?.recMs || 0;
      totals.totalMs += result.metrics?.totalMs || 0;
      totals.detectedBoxes += result.metrics?.detectedBoxes || 0;
      return totals;
    },
    { detMs: 0, recMs: 0, totalMs: 0, detectedBoxes: 0 },
  );
  const first = tileResults[0]?.result;
  return {
    image: { width, height },
    items,
    metrics: {
      ...metricTotals,
      recognizedCount: items.length,
    },
    runtime: first?.runtime || null,
  };
}

function makeExportData() {
  return {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    app: {
      name: "DEKIRU Cards OCR検証",
      version: APP_VERSION,
    },
    ocr: {
      sdk: "@paddleocr/paddleocr-js",
      sdkVersion: OCR_SDK_VERSION,
      model: MODEL_NAME,
      language: "japan (multilingual PP-OCRv5)",
      mode: state.engineMode,
      settings: {
        maxImageSide: Number(ui.maxSide.value),
        ...getRecognitionParams(),
      },
      initialization: state.engineSummary
        ? {
            elapsedMs: state.engineSummary.elapsedMs,
            backend: state.engineSummary.backend,
            detProvider: state.engineSummary.detProvider,
            recProvider: state.engineSummary.recProvider,
            webgpuAvailable: state.engineSummary.webgpuAvailable,
          }
        : null,
    },
    pages: state.pages.map((page, pageIndex) => ({
      page: pageIndex + 1,
      fileName: page.file.name,
      originalBytes: page.file.size,
      image: page.imageInfo,
      elapsedMs: page.wallMs,
      error: page.error,
      metrics: page.result?.metrics || null,
      runtime: page.result?.runtime || null,
      items:
        page.result?.items.map((item, itemIndex) => ({
          index: itemIndex + 1,
          text: item.text,
          confidence: item.score,
          poly: item.poly,
          bounds: boundsFromPoly(item.poly),
        })) || [],
    })),
  };
}

function makeSummaryCard(label, value, note) {
  const card = document.createElement("div");
  card.className = "summary-card";
  const labelElement = document.createElement("span");
  labelElement.textContent = label;
  const valueElement = document.createElement("strong");
  valueElement.textContent = value;
  card.append(labelElement, valueElement);
  if (note) {
    const noteElement = document.createElement("small");
    noteElement.textContent = note;
    card.append(noteElement);
  }
  return card;
}

function resultColor(index) {
  return `hsl(${(index * 53 + 203) % 360} 76% 46%)`;
}

async function drawOverlay(canvas, page) {
  if (!page.normalizedBlob || !page.result) return;
  const decoded = await decodeBlob(page.normalizedBlob);
  const maxCanvasWidth = 1180;
  const scale = Math.min(1, maxCanvasWidth / decoded.width);
  canvas.width = Math.max(1, Math.round(decoded.width * scale));
  canvas.height = Math.max(1, Math.round(decoded.height * scale));
  const context = canvas.getContext("2d", { alpha: false });
  context.drawImage(decoded.drawable, 0, 0, canvas.width, canvas.height);
  decoded.close();

  if (!ui.toggleBoxes.checked) return;
  context.lineJoin = "round";
  context.font = "600 13px -apple-system, BlinkMacSystemFont, sans-serif";
  page.result.items.forEach((item, index) => {
    const color = resultColor(index);
    context.beginPath();
    item.poly.forEach(([x, y], pointIndex) => {
      const px = x * scale;
      const py = y * scale;
      if (pointIndex === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    });
    context.closePath();
    context.fillStyle = color.replace("hsl(", "hsl(").replace(")", " / 0.10)");
    context.fill();
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.stroke();

    const [firstX, firstY] = item.poly[0];
    const label = String(index + 1);
    const labelX = Math.max(2, firstX * scale);
    const labelY = Math.max(16, firstY * scale);
    const labelWidth = Math.max(22, context.measureText(label).width + 10);
    context.fillStyle = color;
    context.fillRect(labelX, labelY - 16, labelWidth, 18);
    context.fillStyle = "#fff";
    context.fillText(label, labelX + 5, labelY - 3);
  });
}

function makeItemsTable(page) {
  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  const table = document.createElement("table");
  table.className = "result-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["No.", "認識した文字", "自信度", "位置 x / y / 幅 / 高さ"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.append(th);
  });
  head.append(headRow);
  const body = document.createElement("tbody");

  page.result.items.forEach((item, index) => {
    const row = document.createElement("tr");
    const number = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "number-badge";
    badge.style.background = resultColor(index);
    badge.textContent = String(index + 1);
    number.append(badge);

    const textCell = document.createElement("td");
    textCell.className = "recognized-text";
    textCell.textContent = item.text || "（空欄）";

    const score = document.createElement("td");
    score.textContent = `${(item.score * 100).toFixed(1)}%`;

    const bounds = boundsFromPoly(item.poly);
    const position = document.createElement("td");
    position.className = "position-cell";
    position.textContent = `${bounds.x} / ${bounds.y} / ${bounds.width} / ${bounds.height}`;
    row.append(number, textCell, score, position);
    body.append(row);
  });
  table.append(head, body);
  scroll.append(table);
  return scroll;
}

async function renderResults() {
  ui.results.replaceChildren();
  const successfulPages = state.pages.filter((page) => page.result);
  const totalLines = successfulPages.reduce(
    (sum, page) => sum + page.result.items.length,
    0,
  );
  const totalMs = state.pages.reduce((sum, page) => sum + (page.wallMs || 0), 0);
  ui.summaryGrid.replaceChildren(
    makeSummaryCard("処理した画像", `${successfulPages.length} / ${state.pages.length}枚`),
    makeSummaryCard("認識した文字列", `${totalLines}件`),
    makeSummaryCard("合計処理時間", formatMs(totalMs), "モデル準備時間は含みません"),
    makeSummaryCard(
      "実行方式",
      state.engineMode.startsWith("worker") ? "バックグラウンド" : "通常",
      "写真は端末内で処理",
    ),
  );

  for (let pageIndex = 0; pageIndex < state.pages.length; pageIndex += 1) {
    const page = state.pages[pageIndex];
    const card = document.createElement("article");
    card.className = "result-card";
    const heading = document.createElement("div");
    heading.className = "result-heading";
    const titleBlock = document.createElement("div");
    const title = document.createElement("h3");
    title.textContent = `${pageLabel(pageIndex)}　${page.file.name}`;
    const meta = document.createElement("p");
    if (page.result) {
      meta.textContent = `${page.result.items.length}件・${formatMs(page.wallMs)}・${page.imageInfo.width}×${page.imageInfo.height}px`;
    } else {
      meta.textContent = "処理できませんでした";
    }
    titleBlock.append(title, meta);
    heading.append(titleBlock);
    card.append(heading);

    if (page.error) {
      const error = document.createElement("div");
      error.className = "page-error";
      error.textContent = page.error;
      card.append(error);
    } else {
      const canvasWrap = document.createElement("div");
      canvasWrap.className = "ocr-canvas-wrap";
      const canvas = document.createElement("canvas");
      canvas.className = "ocr-canvas";
      canvas.dataset.pageId = page.id;
      canvasWrap.append(canvas);
      card.append(canvasWrap);

      const confidenceNote = document.createElement("p");
      confidenceNote.className = "confidence-note";
      confidenceNote.textContent =
        "自信度はOCRエンジンの推定値です。文字が正しいことを保証する点数ではありません。";
      card.append(confidenceNote, makeItemsTable(page));
      await drawOverlay(canvas, page);
    }
    ui.results.append(card);
  }
  ui.resultSection.hidden = false;
}

async function redrawOverlays() {
  const canvases = ui.results.querySelectorAll("canvas[data-page-id]");
  for (const canvas of canvases) {
    const page = state.pages.find((item) => item.id === canvas.dataset.pageId);
    if (page) await drawOverlay(canvas, page);
  }
}

async function runOcr() {
  if (state.running || state.pages.length === 0) return;
  if (location.protocol === "file:") {
    setStatus(
      "この開き方では文字認識を開始できません",
      "PaddleOCR.jsはHTTP(S)での表示が必要です。GitHub Pagesに置いてから開いてください。",
      0,
      "error",
    );
    return;
  }

  setControlsDisabled(true);
  writeRunMarker("starting");
  ui.resultSection.hidden = true;
  state.pages.forEach((page) => {
    page.result = null;
    page.error = null;
    page.wallMs = null;
  });

  try {
    let engine = state.engineReady ? state.engine : null;
    const pageCount = state.pages.length;
    for (let index = 0; index < pageCount; index += 1) {
      const page = state.pages[index];
      const baseProgress = 14 + (index / pageCount) * 80;
      setStatus(
        `${pageLabel(index)}を読み取っています`,
        `${page.file.name}（${index + 1} / ${pageCount}枚）`,
        baseProgress,
      );
      await nextPaint();
      const startedAt = performance.now();
      try {
        writeRunMarker("preparing-image", page.file.name);
        const normalized = await normalizeImage(page.file);
        if (page.normalizedUrl) URL.revokeObjectURL(page.normalizedUrl);
        page.normalizedBlob = normalized.blob;
        page.normalizedUrl = URL.createObjectURL(normalized.blob);
        page.imageInfo = {
          originalWidth: normalized.originalWidth,
          originalHeight: normalized.originalHeight,
          width: normalized.width,
          height: normalized.height,
          normalizedBytes: normalized.blob.size,
          convertedFromHeic: normalized.convertedFromHeic,
        };
        // On memory-constrained phones, finish decoding and reducing the
        // original camera image before loading the OCR models.
        if (!engine) {
          writeRunMarker("loading-model", page.file.name);
          await nextPaint();
          engine = await ensureEngine();
        }
        const tileResults = [];
        for (let tileIndex = 0; tileIndex < normalized.tiles.length; tileIndex += 1) {
          const tile = normalized.tiles[tileIndex];
          writeRunMarker(
            "recognizing",
            `${page.file.name}:${tileIndex + 1}/${normalized.tiles.length}`,
          );
          setStatus(
            `${pageLabel(index)}を分けて読み取っています`,
            `区画 ${tileIndex + 1} / ${normalized.tiles.length}`,
            baseProgress + ((tileIndex + 1) / normalized.tiles.length) * (70 / pageCount),
          );
          await nextPaint();
          const [result] = await engine.predict(tile.blob, getRecognitionParams());
          tileResults.push({ tile, result });
          await nextPaint();
        }
        page.result = mergeTileResults(
          tileResults,
          normalized.width,
          normalized.height,
        );
      } catch (error) {
        console.error(`OCR failed for ${page.file.name}`, error);
        page.error = error instanceof Error ? error.message : String(error);
      }
      page.wallMs = performance.now() - startedAt;
      setStatus(
        `${pageLabel(index)}の処理が終わりました`,
        page.error
          ? "この画像は処理できませんでした。次の画像へ進みます。"
          : `${page.result.items.length}件の文字列を検出しました。`,
        14 + ((index + 1) / pageCount) * 80,
        page.error ? "warning" : "normal",
      );
      await nextPaint();
    }

    setStatus(
      "結果をまとめています",
      "画像上の認識位置と一覧を作っています。",
      96,
    );
    await renderResults();
    setStatus(
      "検証が完了しました",
      "画像上の番号と、下の認識結果一覧を見比べてください。",
      100,
      state.pages.some((page) => page.error) ? "warning" : "success",
    );
    clearRunMarker();
    ui.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error(error);
    state.engineReady = false;
    const message = error instanceof Error ? error.message : String(error);
    setStatus(
      "文字認識を開始できませんでした",
      message,
      0,
      "error",
    );
    ui.retryEngineButton.hidden = false;
    clearRunMarker();
  } finally {
    setControlsDisabled(false);
  }
}

async function copyJson() {
  const text = JSON.stringify(makeExportData(), null, 2);
  try {
    await navigator.clipboard.writeText(text);
    const previous = ui.copyJsonButton.textContent;
    ui.copyJsonButton.textContent = "コピーしました";
    setTimeout(() => {
      ui.copyJsonButton.textContent = previous;
    }, 1600);
  } catch (_error) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
}

function downloadJson() {
  const text = JSON.stringify(makeExportData(), null, 2);
  const blob = new Blob([text], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  anchor.href = url;
  anchor.download = `DEKIRU_Cards_OCR_result_${stamp}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function clearAll() {
  if (state.running) return;
  state.pages.forEach((page) => {
    URL.revokeObjectURL(page.previewUrl);
    if (page.normalizedUrl) URL.revokeObjectURL(page.normalizedUrl);
  });
  state.pages = [];
  ui.libraryInput.value = "";
  ui.cameraInput.value = "";
  ui.results.replaceChildren();
  ui.resultSection.hidden = true;
  ui.statusPanel.hidden = true;
  updateSelectedList();
}

function bindEvents() {
  ui.libraryButton.addEventListener("click", () => ui.libraryInput.click());
  ui.cameraButton.addEventListener("click", () => ui.cameraInput.click());
  ui.libraryInput.addEventListener("change", (event) => {
    addFiles(event.target.files);
    event.target.value = "";
  });
  ui.cameraInput.addEventListener("change", (event) => {
    addFiles(event.target.files);
    event.target.value = "";
  });
  ui.startButton.addEventListener("click", runOcr);
  ui.clearButton.addEventListener("click", clearAll);
  ui.copyJsonButton.addEventListener("click", copyJson);
  ui.downloadJsonButton.addEventListener("click", downloadJson);
  ui.toggleBoxes.addEventListener("change", redrawOverlays);
  ui.retryEngineButton.addEventListener("click", () => {
    state.engineReady = false;
    ui.retryEngineButton.hidden = true;
    runOcr();
  });
}

function initialize() {
  $("versionLabel").textContent = `検証版 v${APP_VERSION}`;
  if (IS_IOS) {
    ui.maxSide.value = "1600";
  }
  if (location.protocol === "file:") {
    ui.httpWarning.hidden = false;
  }
  const interrupted = readRunMarker();
  if (interrupted) {
    const phaseText = {
      starting: "処理開始直後",
      "preparing-image": "写真の準備中",
      "loading-model": "OCRモデルの準備中",
      recognizing: "文字認識中",
      unknown: "処理中",
    }[interrupted.phase] || "処理中";
    setStatus(
      "前回の処理中にページが再読み込みされました",
      `${phaseText}に端末の負荷が高くなりました。今回の表示をお知らせください。`,
      0,
      "warning",
    );
  }
  bindEvents();
  updateSelectedList();
}

initialize();
