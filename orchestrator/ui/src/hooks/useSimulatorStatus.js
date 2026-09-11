import { useCallback, useEffect, useRef, useState } from 'react';

const API_BASE = '/api/simulator';

export const STOPPED_SIMULATOR_STATUS = {
  available: false,
  state: 'checking',
  environment: null,
  reset_count: 0,
  observation_sequence: 0,
  camera_sequence: 0,
  message: 'Checking Cyclo Lab…',
  training_active: false,
};

async function readResponse(response) {
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }
  if (!response.ok) {
    throw new Error(data.detail || data.message || `Simulator request failed (${response.status})`);
  }
  return data;
}

export async function fetchSimulatorStatus() {
  return readResponse(await fetch(`${API_BASE}/status`));
}

export async function requestSimulator(command, body) {
  return readResponse(await fetch(`${API_BASE}/${command}`, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  }));
}

export async function waitForSimulatorReady({ timeoutMs = 300000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let status = null;
  while (Date.now() < deadline) {
    status = await fetchSimulatorStatus();
    if (status.state === 'error') {
      throw new Error(status.error || status.message || 'Cyclo Lab simulation failed');
    }
    if (status.state === 'stopped') {
      throw new Error(status.error || status.message || 'Cyclo Lab simulation exited during startup');
    }
    if (
      ['ready', 'running'].includes(status.state) &&
      Number(status.observation_sequence) > 0 &&
      Number(status.camera_sequence) > 0
    ) {
      return status;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error('Timed out waiting for Cyclo Lab and fresh state/camera samples');
}

export default function useSimulatorStatus({ intervalMs = 1000, enabled = true } = {}) {
  const [status, setStatus] = useState(STOPPED_SIMULATOR_STATUS);
  const [environments, setEnvironments] = useState([]);
  const [error, setError] = useState('');
  const mountedRef = useRef(true);

  const refreshStatus = useCallback(async ({ quiet = false } = {}) => {
    try {
      const next = await fetchSimulatorStatus();
      if (mountedRef.current) {
        setStatus(next);
        setError('');
      }
      return next;
    } catch (requestError) {
      if (mountedRef.current && !quiet) setError(requestError.message);
      return null;
    }
  }, []);

  const runCommand = useCallback(async (command, body) => {
    const next = await requestSimulator(command, body);
    if (mountedRef.current) {
      setStatus(next);
      setError('');
    }
    return next;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    if (!enabled) return () => { mountedRef.current = false; };

    refreshStatus();
    const timer = setInterval(() => refreshStatus({ quiet: true }), intervalMs);
    return () => {
      mountedRef.current = false;
      clearInterval(timer);
    };
  }, [enabled, intervalMs, refreshStatus]);

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    fetch(`${API_BASE}/environments`)
      .then(readResponse)
      .then((result) => {
        if (!cancelled) setEnvironments(result.environments || []);
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError.message);
      });
    return () => { cancelled = true; };
  }, [enabled]);

  return { status, environments, error, refreshStatus, runCommand };
}
