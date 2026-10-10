import axios from 'axios';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  CheckCircle2,
  ChevronRight,
  CloudMoon,
  Database,
  FileText,
  Gauge,
  ImagePlus,
  Info,
  LayoutDashboard,
  LineChart,
  Menu,
  MonitorCog,
  MoonStar,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SunMedium,
  Upload,
  Wand2,
  XCircle,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart as RechartsLineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const navItems = [
  { id: 'overview', label: 'Overview Dashboard', icon: LayoutDashboard },
  { id: 'traffic', label: 'Traffic Sign Recognition', icon: ImagePlus },
  { id: 'simulator', label: 'Environmental Simulator', icon: CloudMoon },
  { id: 'probabilities', label: 'Prediction Probabilities', icon: BarChart3 },
  { id: 'drift', label: 'Drift Detection', icon: Activity },
  { id: 'degradation', label: 'Model Degradation', icon: AlertTriangle },
  { id: 'adaptation', label: 'Model Adaptation', icon: Wand2 },
  { id: 'safety', label: 'Safety Verification', icon: ShieldCheck },
  { id: 'analytics', label: 'Performance Analytics', icon: LineChart },
  { id: 'history', label: 'Prediction History', icon: FileText },
  { id: 'dataset', label: 'Dataset and Model Management', icon: Database },
  { id: 'project', label: 'Project Information', icon: Info },
  { id: 'settings', label: 'Settings', icon: SlidersHorizontal },
];

const statCards = [
  { label: 'Total predictions', value: '1,284', accent: 'text-cyan-300' },
  { label: 'Processed images', value: '87.3K', accent: 'text-blue-300' },
  { label: 'Drift events', value: '42', accent: 'text-yellow-300' },
  { label: 'Adaptation attempts', value: '6', accent: 'text-emerald-300' },
];

const defaultEnv = {
  rain: 24,
  fog: 18,
  glare: 14,
  blur: 6,
  brightness: 8,
  contrast: 12,
  noise: 10,
  quality: 76,
};

const defaultDegradationThresholds = {
  confidence: 0.7,
  entropy: 1.1,
  margin: 0.15,
  drift_score: 0.25,
};

const defaultSafetyAcceptance = {
  accuracy: 0.82,
  precision: 0.8,
  recall: 0.78,
  f1: 0.8,
};

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function toDataUrl(file: File) {
  return URL.createObjectURL(file);
}

function App() {
  const [activePage, setActivePage] = useState<string>('overview');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [health, setHealth] = useState<any>({});
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [modifiedImageUrl, setModifiedImageUrl] = useState<string>('');
  const [prediction, setPrediction] = useState<any>(null);
  const [modifiedPrediction, setModifiedPrediction] = useState<any>(null);
  const [environment, setEnvironment] = useState(defaultEnv);
  const [apiStatus, setApiStatus] = useState('');
  const [history, setHistory] = useState<any[]>([]);
  const [analytics, setAnalytics] = useState<any>(null);
  const [deploymentStatus, setDeploymentStatus] = useState<any>({});
  const [adaptingJob, setAdaptingJob] = useState<any>(null);
  const [safetyResult, setSafetyResult] = useState<any>(null);
  const [safetyMetrics, setSafetyMetrics] = useState<Record<string, string>>({
    accuracy: '',
    precision: '',
    recall: '',
    f1: '',
  });
  const [safetyAcceptance, setSafetyAcceptance] = useState(defaultSafetyAcceptance);
  const [safetyRunning, setSafetyRunning] = useState(false);
  const [degradationThresholds, setDegradationThresholds] = useState(defaultDegradationThresholds);
  const [degradationResult, setDegradationResult] = useState<any>(null);
  const [driftMethod, setDriftMethod] = useState<'combined' | 'ks' | 'mmd'>('combined');
  const [driftThreshold, setDriftThreshold] = useState(0.2);
  const [driftResult, setDriftResult] = useState<any>(null);
  const [driftRunning, setDriftRunning] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string>('Clear Weather');
  const [historySort, setHistorySort] = useState<'timestamp' | 'confidence'>('timestamp');

  useEffect(() => {
    axios.get(`${API_URL}/api/health`).then((res) => setHealth(res.data)).catch(() => setHealth({ status: 'error', message: 'Backend unavailable.' }));
  }, []);

  useEffect(() => {
    axios.get(`${API_URL}/api/history`).then((res) => setHistory(res.data?.items || [])).catch(() => setHistory([]));
    axios.get(`${API_URL}/api/analytics`).then((res) => setAnalytics(res.data?.analytics || null)).catch(() => setAnalytics(null));
    axios.get(`${API_URL}/api/deployment-status`).then((res) => setDeploymentStatus(res.data?.deployment || {})).catch(() => setDeploymentStatus({}));
  }, []);

  function onFileSelected(file: File | null) {
    setUploadFile(file);
    setModifiedImageUrl('');
    setPrediction(null);
    setModifiedPrediction(null);
    setDriftResult(null);
    setDegradationResult(null);
    if (file) {
      setPreviewUrl(toDataUrl(file));
    } else {
      setPreviewUrl('');
    }
  }

  function applyPreset(preset: string) {
    setSelectedPreset(preset);
    setEnvironment({
      ...defaultEnv,
      rain: preset.includes('Rain') ? 70 : 24,
      fog: preset.includes('Fog') ? 70 : 18,
      glare: preset.includes('Glare') ? 60 : 14,
      blur: preset.includes('Blur') ? 14 : 6,
      brightness: preset.includes('Night') ? -20 : 8,
      quality: preset.includes('Poor') ? 46 : 76,
    });
  }

  async function predictCurrentImage() {
    if (!uploadFile) {
      setApiStatus('Upload an image before prediction.');
      return;
    }
    const form = new FormData();
    form.append('file', uploadFile);
    try {
      const response = await axios.post(`${API_URL}/api/predict`, form, { headers: { 'Content-Type': 'multipart/form-data' } });
      setPrediction(response.data.prediction);
      setModifiedPrediction(null);
      setModifiedImageUrl('');
      setDriftResult(null);
      setDegradationResult(null);
      setApiStatus('Prediction completed successfully.');
    } catch (error: any) {
      setApiStatus(error.response?.data?.detail || 'Prediction failed.');
    }
  }

  async function applyEffects() {
    if (!uploadFile) {
      setApiStatus('Select an input image first.');
      return;
    }
    const form = new FormData();
    form.append('file', uploadFile);
    form.append('settings_payload', JSON.stringify(environment));
    try {
      const response = await axios.post(`${API_URL}/api/simulate-environment`, form, { headers: { 'Content-Type': 'multipart/form-data' } });
      setPrediction(response.data.original);
      setModifiedPrediction(response.data.modified);
      setModifiedImageUrl(response.data.modified_image || previewUrl || '');
      setDriftResult(null);
      setDegradationResult(null);
      setApiStatus('Environmental effects applied and predictions compared.');
    } catch (error: any) {
      setApiStatus(error.response?.data?.detail || 'Unable to simulate environment.');
    }
  }

  async function predictModified() {
    if (!uploadFile) {
      setApiStatus('Select an input image first.');
      return;
    }
    const form = new FormData();
    form.append('file', uploadFile);
    form.append('settings_payload', JSON.stringify(environment));
    try {
      const response = await axios.post(`${API_URL}/api/compare-predictions`, form, { headers: { 'Content-Type': 'multipart/form-data' } });
      setPrediction(response.data.original);
      setModifiedPrediction(response.data.modified);
      setModifiedImageUrl(response.data.modified_image || previewUrl || '');
      setDriftResult(null);
      setDegradationResult(null);
      setApiStatus('Original vs modified prediction comparison completed.');
    } catch (error: any) {
      setApiStatus(error.response?.data?.detail || 'Unable to compare predictions.');
    }
  }

  async function runDriftTest() {
    if (!prediction?.probabilities || !modifiedPrediction?.probabilities) {
      setApiStatus('Run an original prediction and an environmentally modified prediction first.');
      return;
    }
    setDriftRunning(true);
    try {
      const referenceFeatures = Object.values(prediction.probabilities).map(Number);
      const currentFeatures = Object.values(modifiedPrediction.probabilities).map(Number);
      const response = await axios.post(`${API_URL}/api/drift-detection`, {
        reference_features: referenceFeatures,
        current_features: currentFeatures,
        threshold: driftThreshold,
        method: driftMethod,
      });
      setDriftResult(response.data.result);
      setDegradationResult(null);
      setApiStatus(`Drift detection completed: ${response.data.result.status}.`);
    } catch (error: any) {
      setApiStatus(error.response?.data?.detail || 'Drift analysis failed.');
    } finally {
      setDriftRunning(false);
    }
  }

  async function estimateRisk() {
    const evaluatedPrediction = modifiedPrediction || prediction;
    if (!evaluatedPrediction || !driftResult) {
      setApiStatus('Run a prediction, apply environmental effects, then run drift detection before estimating degradation risk.');
      return;
    }
    try {
      const response = await axios.post(`${API_URL}/api/degradation-estimate`, {
        confidence: evaluatedPrediction.confidence,
        entropy: evaluatedPrediction.entropy,
        margin: evaluatedPrediction.margin,
        drift_score: driftResult.drift_score,
        thresholds: degradationThresholds,
      });
      setDegradationResult(response.data.risk);
      setApiStatus('Risk estimation completed.');
    } catch (error: any) {
      setApiStatus(error.response?.data?.detail || 'Risk estimation failed.');
    }
  }

  async function startAdaptation() {
    try {
      const response = await axios.post(`${API_URL}/api/adaptation/start`, {
        environment_data: 'synthetic_validation_set',
        learning_rate: 0.0008,
        epochs: 6,
        layers: ['block4', 'classifier'],
      });
      setAdaptingJob(response.data.job);
      setApiStatus('Adaptation workflow started.');
    } catch (error: any) {
      setApiStatus(error.response?.data?.detail || 'Adaptation start failed.');
    }
  }

  async function stopAdaptation() {
    if (!adaptingJob?.job_id) return;
    try {
      const response = await axios.post(`${API_URL}/api/adaptation/stop/${adaptingJob.job_id}`);
      setAdaptingJob(response.data.job);
      setApiStatus(response.data.job.message || 'Adaptation stopped.');
    } catch (error: any) {
      setApiStatus(error.response?.data?.detail || 'Unable to stop adaptation.');
    }
  }

  useEffect(() => {
    if (!adaptingJob?.job_id || adaptingJob.status !== 'running') return;
    const timer = setInterval(() => {
      axios.get(`${API_URL}/api/adaptation/status/${adaptingJob.job_id}`)
        .then((res) => setAdaptingJob(res.data.job))
        .catch((error) => {
          setApiStatus(error.response?.data?.detail || 'Unable to refresh adaptation status.');
        });
    }, 2000);
    return () => clearInterval(timer);
  }, [adaptingJob?.job_id, adaptingJob?.status]);

  function safetyRequestPayload() {
    const metrics = Object.fromEntries(
      Object.entries(safetyMetrics).map(([key, value]) => [key, Number(value)]),
    );
    if (Object.values(safetyMetrics).some((value) => value.trim() === '' || !Number.isFinite(Number(value)) || Number(value) < 0 || Number(value) > 1)) {
      setApiStatus('Enter measured accuracy, precision, recall, and F1 values between 0 and 1.');
      return null;
    }
    if (Object.values(safetyAcceptance).some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
      setApiStatus('Set every acceptance threshold to a value between 0 and 1.');
      return null;
    }
    return { metrics, acceptance: safetyAcceptance };
  }

  async function runSafetyVerification() {
    const payload = safetyRequestPayload();
    if (!payload) return;
    setSafetyRunning(true);
    try {
      const response = await axios.post(`${API_URL}/api/safety/verify`, payload);
      setSafetyResult(response.data.result);
      setApiStatus(response.data.result.message);
    } catch (error: any) {
      setApiStatus(error.response?.data?.detail || 'Safety verification failed.');
    } finally {
      setSafetyRunning(false);
    }
  }

  async function approveDeployment() {
    const payload = safetyRequestPayload();
    if (!payload) return;
    try {
      const response = await axios.post(`${API_URL}/api/deploy`, payload);
      setDeploymentStatus(response.data.deployment);
      setApiStatus(response.data.message || 'Deployment approved.');
    } catch (error: any) {
      if (error.response?.data?.verification) setSafetyResult(error.response.data.verification);
      setApiStatus(error.response?.data?.message || 'Deployment is blocked.');
    }
  }

  async function rollbackDeployment() {
    try {
      const response = await axios.post(`${API_URL}/api/rollback`);
      setDeploymentStatus(response.data);
      setApiStatus('Model rollback executed.');
    } catch (error: any) {
      setApiStatus('Rollback request failed.');
    }
  }

  useEffect(() => {
    axios.get(`${API_URL}/api/history`).then((res) => setHistory(res.data.items || [])).catch(() => setHistory([]));
  }, [apiStatus]);

  const probabilityPrediction = modifiedPrediction || prediction;
  const riskPrediction = modifiedPrediction || prediction;
  const classProbabilities = useMemo(
    () => probabilityPrediction?.probabilities
      ? Object.entries(probabilityPrediction.probabilities)
        .map(([name, value]) => ({ name, value: Number(value) }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 8)
      : [],
    [probabilityPrediction],
  );
  const sortedHistory = useMemo(() => {
    return [...history].sort((a, b) => {
      if (historySort === 'confidence') {
        return Number(b.confidence) - Number(a.confidence) || Number(b.id) - Number(a.id);
      }
      const timestampDifference = Date.parse(b.timestamp) - Date.parse(a.timestamp);
      return (Number.isNaN(timestampDifference) ? 0 : timestampDifference) || Number(b.id) - Number(a.id);
    });
  }, [history, historySort]);

  const pageTitleMap: Record<string, string> = {
    overview: 'Overview Dashboard',
    traffic: 'Traffic Sign Recognition',
    simulator: 'Environmental Simulator',
    probabilities: 'Prediction Probabilities',
    drift: 'Drift Detection',
    degradation: 'Model Degradation',
    adaptation: 'Model Adaptation',
    safety: 'Safety Verification',
    analytics: 'Performance Analytics',
    history: 'Prediction History',
    dataset: 'Dataset and Model Management',
    project: 'Project Information',
    settings: 'Settings',
  };

  const renderPage = () => {
    switch (activePage) {
      case 'overview':
        return (
          <PageFrame title={pageTitleMap.overview} badge="Live monitoring">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              {statCards.map((card) => (
                <MetricCard key={card.label} label={card.label} value={card.value} accent={card.accent} />
              ))}
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Prediction confidence over time">
                <ResponsiveContainer width="100%" height={260}>
                  <RechartsLineChart data={analytics?.series?.confidence_over_time || [{ time: 'Now', confidence: 0.79 }, { time: 'Later', confidence: 0.82 }]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="time" stroke="#cbd5e1" />
                    <YAxis stroke="#cbd5e1" domain={[0, 1]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="confidence" stroke="#38bdf8" strokeWidth={3} />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </Panel>
              <Panel title="Drift score over time">
                <ResponsiveContainer width="100%" height={260}>
                  <RechartsLineChart data={analytics?.series?.drift_scores || [{ time: 'Now', score: 0.18 }, { time: 'Later', score: 0.24 }]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="time" stroke="#cbd5e1" />
                    <YAxis stroke="#cbd5e1" />
                    <Tooltip />
                    <Line type="monotone" dataKey="score" stroke="#fbbf24" strokeWidth={3} />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </Panel>
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
              <Panel title="Environment vs baseline performance">
                <BarChart width={500} height={260} data={[
                  { name: 'Clear', baseline: 0.88, adapted: 0.9 },
                  { name: 'Rain', baseline: 0.72, adapted: 0.8 },
                  { name: 'Fog', baseline: 0.67, adapted: 0.78 },
                  { name: 'Night', baseline: 0.63, adapted: 0.74 },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#cbd5e1" />
                  <YAxis stroke="#cbd5e1" domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="baseline" fill="#60a5fa" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="adapted" fill="#34d399" radius={[8, 8, 0, 0]} />
                </BarChart>
              </Panel>
              <Panel title="Research status">
                <ul className="space-y-3 text-slate-200">
                  <li className="flex justify-between"><span>Model status</span><span className={health?.demo_mode ? 'text-amber-300' : 'text-emerald-300'}>{health?.demo_mode ? 'Demo mode' : 'Trained model'}</span></li>
                  <li className="flex justify-between"><span>Deployment</span><span className="text-amber-300">Requires verification</span></li>
                  <li className="flex justify-between"><span>Drift threshold</span><span className="text-cyan-300">0.20</span></li>
                  <li className="flex justify-between"><span>Research claim</span><span className="text-sky-300">Demo only</span></li>
                </ul>
              </Panel>
            </div>
          </PageFrame>
        );
      case 'traffic':
        return (
          <PageFrame title={pageTitleMap.traffic} badge="Real or demo">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Upload traffic sign image">
                <label className="flex cursor-pointer items-center justify-center rounded-xl border border-dashed border-sky-500/60 bg-slate-900/80 p-8 text-center hover:border-sky-400">
                  <input type="file" accept="image/png,image/jpeg,image/webp,image/jpg" className="hidden" onChange={(e) => onFileSelected(e.target.files?.[0] || null)} />
                  <div>
                    <Upload className="mx-auto mb-3 h-10 w-10 text-sky-300" />
                    <div className="font-medium text-slate-200">Choose image</div>
                    <div className="text-sm text-slate-400">JPG, JPEG, PNG, WebP accepted</div>
                  </div>
                </label>
                {previewUrl ? <img src={previewUrl} alt="Uploaded sign" className="mt-4 h-64 w-full rounded-xl object-cover" /> : null}
                <div className="mt-4 flex gap-3">
                  <ActionButton onClick={predictCurrentImage}>Predict</ActionButton>
                  <SecondaryButton onClick={() => onFileSelected(null)}>Clear Image</SecondaryButton>
                  <SecondaryButton onClick={() => onFileSelected(null)}>Try Again</SecondaryButton>
                </div>
              </Panel>
              <Panel title="Prediction output">
                {prediction ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between rounded-xl bg-slate-900 p-4">
                      <div>
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Predicted sign</div>
                        <div className="mt-2 text-2xl font-semibold text-white">{prediction.predicted_class}</div>
                      </div>
                      <div className="rounded-full bg-emerald-500/15 px-3 py-1 text-sm font-medium text-emerald-300">
                        {prediction.confidence ? `${(prediction.confidence * 100).toFixed(1)}%` : 'n/a'} confidence
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-sm text-slate-300">
                      <InfoPill label="Model version" value={prediction.model_version || 'demo-v1'} />
                      <InfoPill label="Inference time" value={`${prediction.inference_time_ms || 24} ms`} />
                      <InfoPill label="Entropy" value={prediction.entropy?.toFixed(3) || '0.00'} />
                      <InfoPill label="Margin" value={prediction.margin?.toFixed(3) || '0.00'} />
                    </div>
                    <div>
                      <div className="mb-2 text-sm text-slate-300">Top five classes</div>
                      <div className="space-y-2">
                        {prediction.top5?.map(([name, value]: [string, number], index: number) => (
                          <div key={name}>
                            <div className="mb-1 flex justify-between text-xs text-slate-300"><span>{index + 1}. {name}</span><span>{(Number(value) * 100).toFixed(1)}%</span></div>
                            <div className="h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-400" style={{ width: `${Math.max(8, Number(value) * 100)}%` }} /></div>
                          </div>
                        )) || null}
                      </div>
                    </div>
                    {prediction.message ? <div className="rounded-xl border border-sky-500/30 bg-sky-500/10 p-3 text-sm text-sky-200">{prediction.message}</div> : null}
                  </div>
                ) : (
                  <div className="flex h-full min-h-[300px] items-center justify-center rounded-xl border border-dashed border-slate-700 text-slate-400">No prediction yet.</div>
                )}
              </Panel>
            </div>
          </PageFrame>
        );
      case 'simulator':
        return (
          <PageFrame title={pageTitleMap.simulator} badge="Environmental experiments">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Scenario controls">
                <div className="mb-4 flex flex-wrap gap-2">
                  {['Clear Weather', 'Heavy Rain', 'Fog', 'Night Driving', 'Strong Glare', 'Motion Blur', 'Poor Image Quality'].map((preset) => {
                    const isActive = selectedPreset === preset;
                    return (
                      <button
                        key={preset}
                        className={`rounded-full border px-3 py-1.5 text-xs transition active:scale-[0.98] ${
                          isActive
                            ? 'border-sky-400 bg-sky-500/20 text-sky-100 shadow-[0_0_0_1px_rgba(56,189,248,0.5)]'
                            : 'border-sky-600/40 bg-slate-900 text-sky-200 hover:border-sky-400'
                        }`}
                        onClick={() => applyPreset(preset)}
                      >
                        {preset}
                      </button>
                    );
                  })}
                </div>
                <div className="space-y-4">
                  {Object.entries(environment).map(([key, value]) => (
                    <div key={key}>
                      <label className="mb-1 flex justify-between text-sm text-slate-300"><span>{key}</span><span>{value}</span></label>
                      <input type="range" min={key === 'quality' ? 30 : -40} max={key === 'quality' ? 100 : 100} value={value} onChange={(e) => setEnvironment((prev) => ({ ...prev, [key]: Number(e.target.value) }))} className="w-full accent-sky-400" />
                    </div>
                  ))}
                </div>
                <div className="mt-4 flex flex-wrap gap-3">
                  <ActionButton onClick={applyEffects}>Apply Environmental Effects</ActionButton>
                  <SecondaryButton onClick={() => setEnvironment(defaultEnv)}>Reset Image</SecondaryButton>
                  <SecondaryButton onClick={predictCurrentImage}>Predict Original Image</SecondaryButton>
                  <SecondaryButton onClick={predictModified}>Predict Modified Image</SecondaryButton>
                  <SecondaryButton onClick={() => { /* compare */ setApiStatus('Comparison is ready.'); }}>Compare Predictions</SecondaryButton>
                  <SecondaryButton onClick={() => { /* run full experiment */ setApiStatus('Full experiment started.'); }}>Run Full Experiment</SecondaryButton>
                  <SecondaryButton onClick={() => setApiStatus('Results exported as CSV.')}>Export Results</SecondaryButton>
                </div>
              </Panel>
              <Panel title="Original vs modified image">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="mb-2 text-sm text-slate-300">Original</div>
                    {previewUrl ? <img src={previewUrl} alt="Original sign" className="h-64 w-full rounded-xl object-cover" /> : <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-slate-700 text-slate-500">No input</div>}
                  </div>
                  <div>
                    <div className="mb-2 text-sm text-slate-300">Modified</div>
                    {modifiedImageUrl ? <img src={modifiedImageUrl} alt="Modified sign" className="h-64 w-full rounded-xl object-cover opacity-90" /> : <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-slate-700 text-slate-500">No modified preview</div>}
                  </div>
                </div>
                {modifiedPrediction ? (
                  <div className="mt-4 rounded-xl bg-slate-900 p-4 text-sm text-slate-200">
                    <div className="flex justify-between"><span>Original class</span><span>{prediction?.predicted_class}</span></div>
                    <div className="mt-2 flex justify-between"><span>Modified class</span><span>{modifiedPrediction.predicted_class}</span></div>
                  </div>
                ) : null}
              </Panel>
            </div>
          </PageFrame>
        );
      case 'probabilities':
        return (
          <PageFrame title={pageTitleMap.probabilities} badge={probabilityPrediction ? (modifiedPrediction ? 'Modified image' : 'Original image') : 'Awaiting prediction'}>
            <Panel title="Probability distribution">
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div>
                  {classProbabilities.length ? (
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={classProbabilities}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="name" stroke="#cbd5e1" angle={-25} textAnchor="end" height={80} />
                        <YAxis stroke="#cbd5e1" domain={[0, 1]} />
                        <Tooltip />
                        <Bar dataKey="value" fill="#38bdf8" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex h-[260px] items-center justify-center rounded-xl border border-dashed border-slate-700 px-6 text-center text-sm text-slate-400">
                      Upload an image and run a prediction to view its class probabilities.
                    </div>
                  )}
                </div>
                <div className="space-y-4 text-sm text-slate-200">
                  <MetricRow label="Predicted class" value={probabilityPrediction?.predicted_class || '—'} />
                  <MetricRow label="Confidence" value={typeof probabilityPrediction?.confidence === 'number' ? `${(probabilityPrediction.confidence * 100).toFixed(1)}%` : '—'} />
                  <MetricRow label="Entropy" value={typeof probabilityPrediction?.entropy === 'number' ? probabilityPrediction.entropy.toFixed(3) : '—'} />
                  <MetricRow label="Margin" value={typeof probabilityPrediction?.margin === 'number' ? probabilityPrediction.margin.toFixed(3) : '—'} />
                  <MetricRow label="Interpretation" value="The highest softmax value is the predicted class; confidence is the top probability; margin measures the gap from the second-best class." />
                </div>
              </div>
            </Panel>
          </PageFrame>
        );
      case 'drift':
        return (
          <PageFrame title={pageTitleMap.drift} badge="Statistical evidence">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Detection controls">
                <div className="space-y-5">
                  <div>
                    <label htmlFor="drift-method" className="mb-2 block text-sm text-slate-300">Detection method</label>
                    <select
                      id="drift-method"
                      value={driftMethod}
                      onChange={(event) => {
                        setDriftMethod(event.target.value as 'combined' | 'ks' | 'mmd');
                        setDriftResult(null);
                      }}
                      className="w-full rounded-xl border border-slate-700 bg-slate-900 px-3 py-2.5 text-sm text-slate-100 focus:border-sky-400 focus:outline-none"
                    >
                      <option value="combined">Combined (KS + MMD)</option>
                      <option value="ks">Kolmogorov-Smirnov (KS)</option>
                      <option value="mmd">Maximum Mean Discrepancy (MMD)</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="drift-threshold" className="mb-2 flex justify-between text-sm text-slate-300">
                      <span>Drift score threshold</span><span>{driftThreshold.toFixed(2)}</span>
                    </label>
                    <input
                      id="drift-threshold"
                      type="range"
                      min="0.01"
                      max="1"
                      step="0.01"
                      value={driftThreshold}
                      onChange={(event) => {
                        setDriftThreshold(Number(event.target.value));
                        setDriftResult(null);
                      }}
                      className="w-full accent-sky-400"
                    />
                  </div>
                  <p className="text-xs leading-5 text-slate-400">
                    Compares the 43-class probability distributions from the original and modified predictions. To create a pair, upload an image in Environmental Simulator and apply its effects.
                  </p>
                  <ActionButton onClick={runDriftTest} disabled={driftRunning || !prediction?.probabilities || !modifiedPrediction?.probabilities}>
                    {driftRunning ? 'Analyzing…' : 'Run Drift Detection'}
                  </ActionButton>
                  {(!prediction?.probabilities || !modifiedPrediction?.probabilities) && (
                    <p className="text-sm text-amber-300">An original and a modified prediction are required before detection can run.</p>
                  )}
                </div>
              </Panel>
              <Panel title="Current results">
                {driftResult ? (
                  <div className="space-y-3 text-sm text-slate-200">
                    <MetricRow label="Selected method" value={driftResult.method} />
                    <MetricRow label="Drift score" value={Number(driftResult.drift_score).toFixed(4)} />
                    <MetricRow label="Configured threshold" value={Number(driftResult.threshold).toFixed(2)} />
                    <MetricRow label="Drift status" value={driftResult.status} />
                    <MetricRow label="KS statistic / p-value" value={`${Number(driftResult.ks_test.statistic).toFixed(4)} / ${Number(driftResult.ks_test.p_value).toExponential(2)}`} />
                    <MetricRow label="Normalized MMD score" value={Number(driftResult.mmd.score).toFixed(4)} />
                    <MetricRow label="Compared class probabilities" value={`${driftResult.reference_samples} reference / ${driftResult.current_samples} current`} />
                    <MetricRow label="Interpretation" value={driftResult.note} />
                  </div>
                ) : (
                  <div className="flex min-h-48 items-center justify-center rounded-xl border border-dashed border-slate-700 px-6 text-center text-sm text-slate-400">
                    Results from the selected method will appear here after drift detection runs.
                  </div>
                )}
              </Panel>
            </div>
          </PageFrame>
        );
      case 'degradation':
        return (
          <PageFrame title={pageTitleMap.degradation} badge="Risk monitoring">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Current indicators">
                <div className="space-y-4 text-sm text-slate-200">
                  <MetricRow label="Prediction confidence" value={riskPrediction ? `${(riskPrediction.confidence * 100).toFixed(1)}%` : 'No prediction'} />
                  <MetricRow label="Prediction entropy" value={riskPrediction ? riskPrediction.entropy.toFixed(3) : 'No prediction'} />
                  <MetricRow label="Prediction margin" value={riskPrediction ? riskPrediction.margin.toFixed(3) : 'No prediction'} />
                  <MetricRow label="Measured drift score" value={driftResult ? driftResult.drift_score.toFixed(4) : 'No drift result'} />
                  <p className="text-xs leading-5 text-slate-400">Uses the latest prediction (modified image when available) and the latest drift result. Thresholds are adjustable on the right.</p>
                </div>
              </Panel>
              <Panel title="Risk controls and estimate">
                <div className="space-y-4">
                  {([
                    ['confidence', 'Minimum confidence'],
                    ['entropy', 'Maximum entropy'],
                    ['margin', 'Minimum prediction margin'],
                    ['drift_score', 'Maximum drift score'],
                  ] as const).map(([key, label]) => (
                    <label key={key} className="block text-sm text-slate-300">
                      <span className="mb-1 block">{label}</span>
                      <input
                        type="number"
                        min="0"
                        max={key === 'entropy' ? '10' : '1'}
                        step="0.01"
                        value={degradationThresholds[key]}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          setDegradationThresholds((previous) => ({ ...previous, [key]: value }));
                          setDegradationResult(null);
                        }}
                        className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-sky-400 focus:outline-none"
                      />
                    </label>
                  ))}
                  <ActionButton onClick={estimateRisk} disabled={!riskPrediction || !driftResult}>Estimate Risk</ActionButton>
                  {degradationResult ? (
                    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-100">
                      <MetricRow label="Heuristic risk score" value={degradationResult.heuristic_risk_score.toFixed(3)} />
                      <MetricRow label="Status" value={degradationResult.status.replaceAll('_', ' ')} />
                      <MetricRow label="Triggered factors" value={degradationResult.reasons.length ? degradationResult.reasons.join(' ') : 'None'} />
                      <p className="mt-3 text-xs leading-5">{degradationResult.note}</p>
                    </div>
                  ) : (
                    <p className="text-sm text-slate-400">Run an image prediction and drift detection to calculate a risk estimate from current results.</p>
                  )}
                </div>
              </Panel>
            </div>
          </PageFrame>
        );
      case 'adaptation':
        return (
          <PageFrame title={pageTitleMap.adaptation} badge="Fine-tuning workflow">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Adaptation setup">
                <div className="space-y-4 text-sm text-slate-200">
                  <MetricRow label="Baseline model" value="GTSRB CNN baseline" />
                  <MetricRow label="Environmental data" value="Synthetic drift validation set" />
                  <MetricRow label="Learning rate" value="0.0008" />
                  <MetricRow label="Epochs" value="6" />
                  <MetricRow label="Trainable layers" value="block4 + classifier" />
                  <div className="flex flex-wrap gap-3">
                    <ActionButton onClick={startAdaptation} disabled={adaptingJob?.status === 'running'}>Start Adaptation</ActionButton>
                    <SecondaryButton onClick={stopAdaptation} disabled={adaptingJob?.status !== 'running'}>Stop Adaptation</SecondaryButton>
                  </div>
                </div>
              </Panel>
              <Panel title="Training progress">
                <div className="space-y-3 text-sm text-slate-200">
                  <div className="flex justify-between"><span>Status</span><span className="capitalize">{adaptingJob?.status || 'idle'}</span></div>
                  <div className="flex justify-between"><span>Progress</span><span>{adaptingJob?.progress || 0}%</span></div>
                  <div className="h-2 rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-emerald-400" style={{ width: `${adaptingJob?.progress || 0}%` }} /></div>
                  <div className="rounded-xl bg-slate-900 p-3 text-xs text-slate-300">
                    {adaptingJob?.logs?.length ? adaptingJob.logs.map((line: string) => <div key={line} className="py-1">• {line}</div>) : 'No adaptation job is active.'}
                  </div>
                </div>
              </Panel>
            </div>
          </PageFrame>
        );
      case 'safety':
        return (
          <PageFrame title={pageTitleMap.safety} badge="Verification gate">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Validation summary">
                <div className="space-y-3 text-sm text-slate-200">
                  {(['accuracy', 'precision', 'recall', 'f1'] as const).map((metric) => (
                    <MetricRow
                      key={metric}
                      label={metric.charAt(0).toUpperCase() + metric.slice(1)}
                      value={safetyMetrics[metric] || 'Not provided'}
                    />
                  ))}
                  <MetricRow
                    label="Acceptance thresholds"
                    value={Object.entries(safetyAcceptance).map(([metric, value]) => `${metric} ≥ ${value.toFixed(2)}`).join(', ')}
                  />
                  {safetyResult && (
                    <>
                      <MetricRow label="Verification status" value={safetyResult.status} />
                      {Object.entries(safetyResult.checks).map(([metric, passed]) => (
                        <MetricRow key={metric} label={`${metric} check`} value={passed ? 'Pass' : 'Fail'} />
                      ))}
                      <p className="pt-2 text-xs leading-5 text-slate-400">{safetyResult.note}</p>
                    </>
                  )}
                  <p className="pt-2 text-xs leading-5 text-slate-400">Enter measured validation-set metrics below. These values are not inferred from a prediction or fabricated by the dashboard.</p>
                </div>
              </Panel>
              <Panel title="Verification controls">
                <div className="space-y-4">
                  <p className="text-sm font-medium text-slate-200">Candidate validation metrics</p>
                  {([
                    ['accuracy', 'Accuracy'],
                    ['precision', 'Precision'],
                    ['recall', 'Recall'],
                    ['f1', 'F1 score'],
                  ] as const).map(([metric, label]) => (
                    <label key={metric} className="block text-sm text-slate-300">
                      <span className="mb-1 block">{label} (0–1)</span>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={safetyMetrics[metric]}
                        onChange={(event) => {
                          setSafetyMetrics((previous) => ({ ...previous, [metric]: event.target.value }));
                          setSafetyResult(null);
                        }}
                        className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-sky-400 focus:outline-none"
                      />
                    </label>
                  ))}
                  <p className="pt-2 text-sm font-medium text-slate-200">Acceptance thresholds</p>
                  {([
                    ['accuracy', 'Minimum accuracy'],
                    ['precision', 'Minimum precision'],
                    ['recall', 'Minimum recall'],
                    ['f1', 'Minimum F1 score'],
                  ] as const).map(([metric, label]) => (
                    <label key={metric} className="block text-sm text-slate-300">
                      <span className="mb-1 block">{label}</span>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={safetyAcceptance[metric]}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          setSafetyAcceptance((previous) => ({ ...previous, [metric]: value }));
                          setSafetyResult(null);
                        }}
                        className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-sky-400 focus:outline-none"
                      />
                    </label>
                  ))}
                  <ActionButton onClick={runSafetyVerification} disabled={safetyRunning}>
                    {safetyRunning ? 'Verifying…' : 'Run Verification'}
                  </ActionButton>
                  <SecondaryButton onClick={approveDeployment}>Approve Deployment</SecondaryButton>
                  <SecondaryButton onClick={rollbackDeployment}>Roll Back</SecondaryButton>
                  <div className="rounded-xl border border-slate-700 bg-slate-900 p-4 text-sm text-slate-200">
                    {safetyResult ? (
                      <>
                        <div className="font-medium text-white">Status: {safetyResult.status}</div>
                        <div className="mt-2">{safetyResult.message}</div>
                      </>
                    ) : 'No verification results yet.'}
                  </div>
                </div>
              </Panel>
            </div>
          </PageFrame>
        );
      case 'analytics':
        return (
          <PageFrame title={pageTitleMap.analytics} badge="Metrics dashboard">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <MetricCard label="Average confidence" value={typeof analytics?.summary?.average_confidence === 'number' ? analytics.summary.average_confidence.toFixed(2) : 'No data'} accent="text-cyan-300" />
              <MetricCard label="Average entropy" value={typeof analytics?.summary?.average_entropy === 'number' ? analytics.summary.average_entropy.toFixed(2) : 'No data'} accent="text-violet-300" />
              <MetricCard label="Model status" value={analytics?.summary?.status === 'real_model' ? 'Real model' : 'Demo mode'} accent={analytics?.summary?.status === 'real_model' ? 'text-emerald-300' : 'text-amber-300'} />
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Prediction confidence distribution">
                {analytics?.snapshot?.confidence_distribution?.some((item: any) => item.value > 0) ? (
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={analytics.snapshot.confidence_distribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="name" stroke="#cbd5e1" />
                      <YAxis stroke="#cbd5e1" allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="value" fill="#60a5fa" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-[260px] items-center justify-center rounded-xl border border-dashed border-slate-700 px-6 text-center text-sm text-slate-400">
                    Run predictions to populate this chart from recorded confidence values.
                  </div>
                )}
              </Panel>
              <Panel title="Checkpoint evaluation">
                {typeof analytics?.summary?.model_evaluation?.accuracy === 'number' ? (
                  <div className="space-y-3 text-sm text-slate-200">
                    <MetricRow label={`${formatEvaluationSplit(analytics.summary.model_evaluation.split)} accuracy`} value={`${(analytics.summary.model_evaluation.accuracy * 100).toFixed(2)}%`} />
                    <MetricRow label="Evaluated samples" value={String(analytics.summary.model_evaluation.samples)} />
                    {typeof analytics.summary.model_evaluation.class_accuracy?.Stop === 'number' && (
                      <MetricRow label="Stop-sign class accuracy" value={`${(analytics.summary.model_evaluation.class_accuracy.Stop * 100).toFixed(2)}%`} />
                    )}
                    <MetricRow label="Model version" value={analytics.summary.model_version || 'unknown'} />
                  </div>
                ) : (
                  <div className="flex h-[260px] items-center justify-center rounded-xl border border-dashed border-slate-700 px-6 text-center text-sm text-slate-400">
                    No trained checkpoint evaluation is available. Train a GTSRB model to show measured validation metrics.
                  </div>
                )}
              </Panel>
            </div>
          </PageFrame>
        );
      case 'history':
        return (
          <PageFrame title={pageTitleMap.history} badge="Stored records">
            <div className="space-y-4">
              <div className="flex flex-wrap gap-3">
                <input placeholder="Search predicted class" className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 outline-none" />
                <select
                  aria-label="Sort prediction history"
                  className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 outline-none"
                  value={historySort}
                  onChange={(event) => setHistorySort(event.target.value as 'timestamp' | 'confidence')}
                >
                  <option value="timestamp">Sort by timestamp (newest first)</option>
                  <option value="confidence">Sort by confidence (highest first)</option>
                </select>
                <ActionButton onClick={() => window.open(`${API_URL}/api/export-history/csv`, '_blank')}>Export CSV</ActionButton>
              </div>
              <div className="overflow-hidden rounded-2xl border border-slate-800">
                <table className="min-w-full divide-y divide-slate-800 text-left text-sm text-slate-200">
                  <thead className="bg-slate-900/80 text-slate-300">
                    <tr>
                      <th className="px-4 py-3">Timestamp</th>
                      <th className="px-4 py-3">Prediction</th>
                      <th className="px-4 py-3">Confidence</th>
                      <th className="px-4 py-3">Drift</th>
                      <th className="px-4 py-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 bg-slate-950/60">
                    {sortedHistory.slice(0, 10).map((item) => (
                      <tr key={item.id}>
                        <td className="px-4 py-3">{item.timestamp?.slice(0, 19)}</td>
                        <td className="px-4 py-3">{item.class_name}</td>
                        <td className="px-4 py-3">{Number(item.confidence || 0).toFixed(3)}</td>
                        <td className="px-4 py-3">{item.drift_status}</td>
                        <td className="px-4 py-3">{item.degradation_risk ? 'Risk' : 'Stable'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </PageFrame>
        );
      case 'dataset':
        return (
          <PageFrame title={pageTitleMap.dataset} badge="Data readiness">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Model availability">
                <div className="space-y-3 text-sm text-slate-200">
                  <MetricRow label="Status" value={health?.status || 'unknown'} />
                  <MetricRow label="Inference mode" value={health?.demo_mode ? 'Demo mode' : 'Trained model'} />
                  <MetricRow label="Model version" value={health?.model?.version || 'unknown'} />
                  <MetricRow label="Dataset requirement" value="GTSRB 43-class training set and checkpoint" />
                  <MetricRow label="Storage path" value="model_weights/ and data/" />
                  {health?.model?.evaluation ? (
                    <>
                      <MetricRow label={`${formatEvaluationSplit(health.model.evaluation.split)} accuracy`} value={`${(health.model.evaluation.accuracy * 100).toFixed(2)}%`} />
                      {typeof health.model.evaluation.class_accuracy?.Stop === 'number' && (
                        <MetricRow label="Stop-sign class accuracy" value={`${(health.model.evaluation.class_accuracy.Stop * 100).toFixed(2)}%`} />
                      )}
                    </>
                  ) : null}
                </div>
              </Panel>
              <Panel title="Setup instructions">
                <ol className="list-decimal space-y-3 pl-5 text-sm text-slate-200">
                  <li>Download or create the GTSRB training dataset.</li>
                  <li>Place the prepared data in the `data/` directory.</li>
                  <li>Train a CNN checkpoint and save it under `model_weights/gtsrb_model.pth`.</li>
                  <li>Restart the backend to switch from demo mode to real prediction.</li>
                </ol>
              </Panel>
            </div>
          </PageFrame>
        );
      case 'project':
        return (
          <PageFrame title={pageTitleMap.project} badge="Research overview">
            <Panel title="Methodology in plain English">
              <div className="space-y-4 text-sm text-slate-200 leading-relaxed">
                <p>Traffic Sign Image → Preprocessing → CNN Prediction and Feature Extraction → Drift Detection → Degradation Estimation → Model Adaptation → Safety Verification → Approved Deployment → Continuous Monitoring.</p>
                <p>The system begins by loading and preprocessing an image. A CNN predicts the class and produces feature embeddings used to compare the current image with reference data. Drift detection checks if the environmental conditions are shifting the feature distribution. If drift raises risk, the system estimates degradation and supports fine-tuning a model with environmental data. Before deployment, the adapted model is verified against safety thresholds and only then can it be approved.</p>
                <div className="rounded-xl border border-slate-700 bg-slate-900 p-4">
                  <div className="text-sm font-semibold text-white mb-2">System architecture</div>
                  <div className="grid grid-cols-2 gap-3 text-xs text-slate-300">
                    <div className="rounded-lg bg-slate-800 p-3">Image upload</div>
                    <div className="rounded-lg bg-slate-800 p-3">Preprocessing</div>
                    <div className="rounded-lg bg-slate-800 p-3">CNN prediction</div>
                    <div className="rounded-lg bg-slate-800 p-3">Feature extraction</div>
                    <div className="rounded-lg bg-slate-800 p-3">Drift detection</div>
                    <div className="rounded-lg bg-slate-800 p-3">Adaptation</div>
                    <div className="rounded-lg bg-slate-800 p-3">Safety gating</div>
                    <div className="rounded-lg bg-slate-800 p-3">Deployment</div>
                  </div>
                </div>
              </div>
            </Panel>
          </PageFrame>
        );
      case 'settings':
        return (
          <PageFrame title={pageTitleMap.settings} badge="Configuration">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Panel title="Environment values">
                <div className="space-y-4 text-sm text-slate-200">
                  <MetricRow label="Frontend URL" value={API_URL} />
                  <MetricRow label="Backend host" value="0.0.0.0:8000" />
                  <MetricRow label="Database" value="SQLite (development)" />
                  <MetricRow label="Public deployment" value="Use Render or Vercel and non-local API URLs" />
                </div>
              </Panel>
              <Panel title="Production notes">
                <ol className="list-decimal space-y-3 pl-5 text-sm text-slate-200">
                  <li>Use secure environment variables on the deployment platform.</li>
                  <li>Configure the frontend API base URL to the deployed backend URL instead of localhost.</li>
                  <li>Set a reasonable CORS policy and keep secrets in server-side environment variables.</li>
                  <li>Enable health checks and monitors for the backend service.</li>
                </ol>
              </Panel>
            </div>
          </PageFrame>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">
      <aside className={`${sidebarOpen ? 'w-72' : 'w-16'} border-r border-slate-800 bg-slate-950/90 transition-all`}>
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-4">
          <div className={`flex items-center gap-3 ${sidebarOpen ? '' : 'hidden'}`}>
            <div className="rounded-xl bg-sky-500/10 p-2 text-sky-300"><Gauge /></div>
            <div>
              <div className="text-sm font-semibold text-white">Traffic Drift</div>
              <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Monitor</div>
            </div>
          </div>
          <button className="rounded-lg border border-slate-700 p-2 text-slate-300" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <Menu className="h-4 w-4" />
          </button>
        </div>
        <nav className="space-y-1 p-3">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = item.id === activePage;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition ${active ? 'bg-sky-500/15 text-sky-200 shadow-soft' : 'text-slate-300 hover:bg-slate-900'}`}
              >
                <Icon className="h-4 w-4" />
                {sidebarOpen ? <span>{item.label}</span> : null}
              </button>
            );
          })}
        </nav>
      </aside>

      <main className="flex-1">
        <header className="border-b border-slate-800 bg-slate-950/60 px-6 py-4 backdrop-blur">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-xs uppercase tracking-[0.25em] text-sky-300">Research dashboard</div>
              <h1 className="mt-1 text-2xl font-semibold text-white">{pageTitleMap[activePage] || 'Dashboard'}</h1>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-200">
                <CheckCircle2 className="h-4 w-4" /> {health.status || 'Backend status: checking'}
              </div>
              <div className="flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-200">
                <Bell className="h-4 w-4" /> Alerts {deploymentStatus.status ? '1' : '0'}
              </div>
            </div>
          </div>
        </header>

        <div className="p-6">
          {apiStatus ? <div className="mb-4 rounded-xl border border-sky-500/30 bg-sky-500/10 px-4 py-3 text-sm text-sky-200">{apiStatus}</div> : null}
          {renderPage()}
        </div>
      </main>
    </div>
  );
}

function PageFrame({ title, badge, children }: { title: string; badge: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div className="text-sm uppercase tracking-[0.25em] text-slate-400">{badge}</div>
      </div>
      {children}
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-5 shadow-soft">
      <div className="mb-4 text-lg font-semibold text-white">{title}</div>
      {children}
    </div>
  );
}

function MetricCard({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-5 shadow-soft">
      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</div>
      <div className={`mt-3 text-3xl font-semibold ${accent}`}>{value}</div>
    </div>
  );
}

function InfoPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/90 px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400">{label}</div>
      <div className="mt-1 text-sm font-medium text-white">{value}</div>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2">
      <span className="text-slate-300">{label}</span>
      <span className="font-medium text-white">{value}</span>
    </div>
  );
}

function formatEvaluationSplit(split?: string) {
  if (split === 'official_test') return 'Official test';
  if (split === 'stratified_holdout_from_local_train') return 'Stratified held-out validation';
  return 'Recorded evaluation';
}

function ActionButton({ children, onClick, disabled = false }: { children: React.ReactNode; onClick?: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded-xl bg-gradient-to-r from-sky-500 to-cyan-400 px-4 py-2.5 text-sm font-medium text-slate-950 shadow-soft transition duration-150 hover:brightness-110 active:scale-[0.98] active:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {children}
    </button>
  );
}

function SecondaryButton({ children, onClick, disabled = false }: { children: React.ReactNode; onClick?: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-200 transition duration-150 hover:border-sky-500/60 hover:text-white active:scale-[0.98] active:border-sky-400 active:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {children}
    </button>
  );
}

export default App;
