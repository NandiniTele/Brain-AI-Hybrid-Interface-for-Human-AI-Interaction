/**
 * Neural-Link BCI — Centralized API Service
 * All REST calls go through /api/ which Vite proxies to http://localhost:8000
 * WebSocket connects through /ws which Vite proxies to ws://localhost:8000/ws
 */

const BASE = '/api';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface SimulationSettings {
  dataset_name: string;
  artifacts: string[];
}

export interface DatasetMeta {
  name?: string;
  dataset_name?: string;
  subjects: number | string;
  subjects_label?: string;
  channels: number;
  original_channels?: number;
  selected_channels?: number;
  runs_per_subject?: number;
  total_clusters?: number;
  fs: number;
  sampling_rate?: number;
  unique_windows?: number;
  window_duration?: string;
  duration: string;
  labels: string[];
  local: boolean;
  source?: string;
}

export interface DatasetsResponse {
  available: string[];
  current: string;
  meta: Record<string, DatasetMeta>;
}

export interface HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  db_status: string;
  log_count: number;
  current_dataset: string;
  uptime: number;
}

export interface TrainStatusResponse {
  is_training: boolean;
  epoch: number;
  loss: number;
  accuracy: number;
  progress: number;
  model_stats: Record<string, any>;
}

export interface TelemetryLog {
  id: number;
  session_id: string;
  timestamp: number;
  dataset: string;
  focus: number;
  attention: number;
  stress: number;
  fatigue: number;
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  inference_latency: number;
}

// ─── Token management ────────────────────────────────────────────────────────

const TOKEN_KEY = 'neural_link_token';

/** Store a JWT after successful login */
export const storeToken = (token: string): void =>
  localStorage.setItem(TOKEN_KEY, token);

/** Retrieve the stored JWT (null if not logged in) */
export const getToken = (): string | null =>
  localStorage.getItem(TOKEN_KEY);

/** Remove the stored JWT (logout) */
export const clearToken = (): void =>
  localStorage.removeItem(TOKEN_KEY);

// ─── API helpers ──────────────────────────────────────────────────────────────

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const res = await fetch(`${BASE}${path}`, {
    headers,
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed [${res.status}]: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

/** Check backend health and connection status */
export const getHealth = (): Promise<HealthResponse> =>
  fetchJSON<HealthResponse>('/health');

/** Fetch all available datasets with metadata */
export const getDatasets = (): Promise<DatasetsResponse> =>
  fetchJSON<DatasetsResponse>('/datasets');

/** Update simulation settings (dataset + artifacts) */
export const postSimulate = (settings: SimulationSettings): Promise<{ status: string; dataset_meta: DatasetMeta }> =>
  fetchJSON('/simulate', {
    method: 'POST',
    body: JSON.stringify(settings),
  });

/** Kick off model training asynchronously */
export const postTrain = (): Promise<{ status: string }> =>
  fetchJSON('/train', { method: 'POST' });

/** Poll training progress */
export const getTrainStatus = (): Promise<TrainStatusResponse> =>
  fetchJSON<TrainStatusResponse>('/train-status');

/** Fetch telemetry logs from SQLite */
export const getLogs = (limit = 500, offset = 0): Promise<TelemetryLog[]> =>
  fetchJSON<TelemetryLog[]>(`/logs?limit=${limit}&offset=${offset}`);

/** Fetch server-side filtered log count */
export const getLogCount = (params?: {
  emotion?: string;
  min_confidence?: number;
  max_confidence?: number;
  search?: string;
}): Promise<{ count: number }> => {
  const qs = new URLSearchParams();
  if (params?.emotion)         qs.set('emotion', params.emotion);
  if (params?.min_confidence != null) qs.set('min_confidence', String(params.min_confidence));
  if (params?.max_confidence != null) qs.set('max_confidence', String(params.max_confidence));
  if (params?.search)          qs.set('search', params.search);
  const query = qs.toString();
  return fetchJSON<{ count: number }>(`/logs/count${query ? '?' + query : ''}`);
};

/** Fetch current model statistics (architecture, accuracy, etc.) */
export const getModelStats = (): Promise<Record<string, any>> =>
  fetchJSON<Record<string, any>>('/model-stats');

/** Fetch model comparison benchmarks */
export const getModelComparison = (): Promise<Record<string, any>[]> =>
  fetchJSON<Record<string, any>[]>('/model-comparison');

/** Fetch raw training metrics JSON */
export const getMetrics = (): Promise<Record<string, any>> =>
  fetchJSON<Record<string, any>>('/metrics');

// ─── Auth ─────────────────────────────────────────────────────────────────────

export interface LoginResponse {
  access_token: string;
  token_type: string;
  role: string;
}

/** Authenticate and receive a JWT token — token is stored automatically */
export const login = async (username: string, password: string): Promise<LoginResponse> => {
  const res = await fetchJSON<LoginResponse>('/auth/token', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
  storeToken(res.access_token);
  return res;
};

export const WS_URL =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' ||
   window.location.hostname === '127.0.0.1')
    ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8000/ws`
    : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

