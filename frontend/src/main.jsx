import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WEBCAM_INTERVAL_MS = Number(import.meta.env.VITE_WEBCAM_INTERVAL_MS || 300);
const CLASS_COLOR = {
  person: '#60a5fa',
  helmet: '#34d399',
  vest: '#fbbf24',
  no_helmet: '#fb7185',
};

function drawDetections(canvas, source, payload) {
  if (!canvas || !source || !payload?.image) return;
  const ctx = canvas.getContext('2d');
  const width = payload.image.width;
  const height = payload.image.height;
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);

  ctx.lineWidth = Math.max(2, width / 320);
  ctx.font = `${Math.max(15, width / 45)}px Inter, system-ui, sans-serif`;
  for (const det of payload.detections || []) {
    const [x1, y1, x2, y2] = det.bbox;
    const color = CLASS_COLOR[det.class_name] || '#ffffff';
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width + 12;
    const th = Math.max(22, width / 30);
    const ty = Math.max(0, y1 - th);
    ctx.fillRect(x1, ty, tw, th);
    ctx.fillStyle = '#07111f';
    ctx.fillText(label, x1 + 6, ty + th - 6);
  }
}

async function predictBlob(blob, filename = 'frame.jpg', endpoint = '/predict/image') {
  const form = new FormData();
  form.append('file', blob, filename);
  const response = await fetch(`${API_URL}${endpoint}`, { method: 'POST', body: form });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function Metric({ label, value, danger = false }) {
  return (
    <div className={`metric ${danger ? 'danger' : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Summary({ result }) {
  if (!result) return <p className="muted">Run a detection to see metrics.</p>;
  const counts = result.summary?.counts || {};
  const violation = (result.summary?.violations?.no_helmet || 0) > 0;
  return (
    <>
      <div className="summary-grid">
        <Metric label="People" value={counts.person || 0} />
        <Metric label="Helmets" value={counts.helmet || 0} />
        <Metric label="Vests" value={counts.vest || 0} />
        <Metric label="No helmet" value={counts.no_helmet || 0} danger={violation} />
        <Metric label="Latency" value={`${result.latency_ms ?? '-'} ms`} />
        <Metric label="Inference FPS" value={result.inference_fps ?? '-'} />
        <Metric label="Status" value={result.summary?.status || '-'} danger={violation} />
      </div>
      {(result.summary?.messages || []).map((message) => (
        <p className="compliance-note" key={message}>{message}</p>
      ))}
    </>
  );
}

function ImageMode() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const canvasRef = useRef(null);

  async function onFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError('');
    setBusy(true);
    const img = new Image();
    img.onload = async () => {
      try {
        const payload = await predictBlob(file, file.name);
        setResult(payload);
        drawDetections(canvasRef.current, img, payload);
      } catch (err) {
        setError(err.message);
        setResult(null);
      } finally {
        setBusy(false);
        URL.revokeObjectURL(img.src);
      }
    };
    img.src = URL.createObjectURL(file);
  }

  return (
    <div className="mode-layout">
      <div className="viewer-card">
        <div className="toolbar">
          <label className="button primary">
            {busy ? 'Analyzing...' : 'Choose image'}
            <input type="file" accept="image/*" onChange={onFile} disabled={busy} hidden />
          </label>
          <span className="muted">PNG, JPG or WEBP</span>
        </div>
        <div className="canvas-wrap">
          <canvas ref={canvasRef} />
          {!result && <div className="empty-state">Upload a construction-site image</div>}
        </div>
        {error && <p className="error">{error}</p>}
      </div>
      <aside className="side-card">
        <h3>Detection summary</h3>
        <Summary result={result} />
      </aside>
    </div>
  );
}

function WebcamMode() {
  const videoRef = useRef(null);
  const captureRef = useRef(null);
  const overlayRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const inFlightRef = useRef(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [session, setSession] = useState({ frames: 0, violationFrames: 0, totalLatency: 0 });

  const avgLatency = useMemo(
    () => (session.frames ? (session.totalLatency / session.frames).toFixed(1) : '-'),
    [session],
  );

  async function start() {
    setError('');
    setSession({ frames: 0, violationFrames: 0, totalLatency: 0 });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setRunning(true);
      timerRef.current = window.setInterval(captureAndPredict, WEBCAM_INTERVAL_MS);
    } catch (err) {
      setError(`Camera error: ${err.message}`);
    }
  }

  function stop() {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
  }

  useEffect(() => () => stop(), []);

  function captureAndPredict() {
    if (inFlightRef.current || !videoRef.current?.videoWidth) return;
    const video = videoRef.current;
    const canvas = captureRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      inFlightRef.current = true;
      try {
        const payload = await predictBlob(blob, 'webcam.jpg', '/predict/frame');
        setResult(payload);
        drawDetections(overlayRef.current, canvas, payload);
        setSession((current) => ({
          frames: current.frames + 1,
          violationFrames: current.violationFrames + (payload.summary?.status === 'VIOLATION' ? 1 : 0),
          totalLatency: current.totalLatency + Number(payload.latency_ms || 0),
        }));
        setError('');
      } catch (err) {
        setError(err.message);
      } finally {
        inFlightRef.current = false;
      }
    }, 'image/jpeg', 0.82);
  }

  return (
    <div className="mode-layout">
      <div className="viewer-card">
        <div className="toolbar">
          {!running ? (
            <button className="button primary" onClick={start}>Start webcam</button>
          ) : (
            <button className="button danger-button" onClick={stop}>Stop webcam</button>
          )}
          <span className={`live-pill ${running ? 'on' : ''}`}>{running ? 'LIVE' : 'OFFLINE'}</span>
        </div>
        <div className="camera-stage">
          <video ref={videoRef} muted playsInline />
          <canvas ref={overlayRef} className={result ? 'visible' : ''} />
          {!running && <div className="empty-state">Camera preview</div>}
        </div>
        <canvas ref={captureRef} hidden />
        {error && <p className="error">{error}</p>}
      </div>
      <aside className="side-card">
        <h3>Live compliance</h3>
        <Summary result={result} />
        <div className="session-block">
          <h4>Session analytics</h4>
          <div className="summary-grid">
            <Metric label="Frames analyzed" value={session.frames} />
            <Metric label="Violation frames" value={session.violationFrames} danger={session.violationFrames > 0} />
            <Metric label="Avg latency" value={`${avgLatency} ms`} />
            <Metric label="Interval" value={`${WEBCAM_INTERVAL_MS} ms`} />
          </div>
        </div>
        {result?.summary?.status === 'VIOLATION' && (
          <div className="alert">Explicit no-helmet violation detected</div>
        )}
      </aside>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState('image');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    const check = () => {
      fetch(`${API_URL}/health`)
        .then((r) => r.json())
        .then((payload) => active && setHealth(payload))
        .catch(() => active && setHealth(null));
    };
    check();
    const id = window.setInterval(check, 5000);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, []);

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">SafeGuard</div>
        <div className={`api-status ${health?.model_ready ? 'ready' : ''}`}>
          <span /> {health ? (health.model_ready ? 'Model ready' : 'API ready / model pending') : 'API offline'}
        </div>
      </header>

      <section className="hero">
        <p className="eyebrow">REAL-TIME CONSTRUCTION SAFETY</p>
        <h1>See PPE risk as it happens.</h1>
        <p className="subtitle">
          Real-time YOLO detection trained on the semantic intersection of Roboflow Construction Site Safety and Ultralytics Construction-PPE.
        </p>
      </section>

      <section className="facts">
        <Metric label="Canonical classes" value="4" />
        <Metric label="Deployment" value="ONNX Runtime" />
        <Metric label="Modes" value="Image + webcam" />
      </section>

      <nav className="tabs">
        <button className={mode === 'image' ? 'active' : ''} onClick={() => setMode('image')}>Image analysis</button>
        <button className={mode === 'webcam' ? 'active' : ''} onClick={() => setMode('webcam')}>Live webcam</button>
      </nav>

      {mode === 'image' ? <ImageMode /> : <WebcamMode />}

      <footer className="footer">
        SafeGuard reports explicit labels learned from the datasets. It does not infer missing vest or other PPE unless explicit missing-PPE classes are part of the canonical ontology.
      </footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
