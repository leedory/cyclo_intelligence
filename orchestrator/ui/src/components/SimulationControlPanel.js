import React, { useCallback, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import toast from 'react-hot-toast';
import { MdRefresh, MdStop } from 'react-icons/md';
import { InferencePhase } from '../constants/taskPhases';
import { selectInferenceTaskInfo, setInferenceTaskInfo } from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import useSimulatorStatus, {
  fetchSimulatorStatus,
  requestSimulator,
} from '../hooks/useSimulatorStatus';

const ACTIVE_STATES = new Set(['starting', 'ready', 'resetting', 'running', 'stopping', 'error']);
const READY_STATES = new Set(['ready', 'running']);
const RESET_TIMEOUT_MS = 45000;

const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

const resetHasFreshSamples = (baseline, current) => (
  current &&
  Number(current.reset_count) > Number(baseline.reset_count || 0) &&
  Number(current.observation_sequence) > Number(baseline.observation_sequence || 0) &&
  Number(current.camera_sequence) > Number(baseline.camera_sequence || 0) &&
  READY_STATES.has(current.state)
);

export default function SimulationControlPanel() {
  const dispatch = useDispatch();
  const inferencePhase = useSelector((state) => state.tasks.inferenceStatus.inferencePhase);
  const taskInfo = useSelector(selectInferenceTaskInfo);
  const { sendRecordCommand } = useRosServiceCaller();
  const { status, environments, error, refreshStatus, runCommand } = useSimulatorStatus();
  const [busy, setBusy] = useState(false);
  const [resolvedPolicy, setResolvedPolicy] = useState(null);
  const [resolveError, setResolveError] = useState('');

  useEffect(() => {
    const policyPath = String(taskInfo.policyPath || '').trim();
    let cancelled = false;
    if (!policyPath) {
      setResolvedPolicy(null);
      setResolveError('');
      return () => { cancelled = true; };
    }
    requestSimulator('resolve', { policy_path: policyPath })
      .then((result) => {
        if (!cancelled) {
          setResolvedPolicy(result);
          setResolveError('');
          dispatch(setInferenceTaskInfo({ simulationProfile: result.profile }));
        }
      })
      .catch((requestError) => {
        if (!cancelled) {
          setResolvedPolicy(null);
          setResolveError(requestError.message);
        }
      });
    return () => { cancelled = true; };
  }, [dispatch, taskInfo.policyPath]);

  const activeEnvironment = environments.find(
    (environment) => environment.key === status.environment
  );
  const displayedEnvironment = status.environment || resolvedPolicy?.environment;
  const displayedProfile = status.profile || resolvedPolicy?.profile;
  const displayedEnvironmentLabel = status.environment
    ? activeEnvironment?.label
    : resolvedPolicy?.environment_label;
  const profileOptions = environments.find(
    (environment) => environment.key === displayedEnvironment
  )?.profiles || [];
  const selectedProfile = taskInfo.simulationProfile || displayedProfile || '';
  const selectedProfileInfo = profileOptions.find(
    (profile) => profile.key === selectedProfile
  );
  const displayedGymId = status.environment
    ? status.gym_id
    : selectedProfileInfo?.gym_id || resolvedPolicy?.gym_id;

  const active = ACTIVE_STATES.has(status.state);
  const ready = READY_STATES.has(status.state);
  const policyActive = inferencePhase !== InferencePhase.READY;

  const reset = useCallback(async () => {
    const baseline = await refreshStatus();
    if (!baseline || !READY_STATES.has(baseline.state)) return;
    const shouldResume = inferencePhase === InferencePhase.INFERENCING;
    setBusy(true);
    try {
      if (shouldResume) {
        const paused = await sendRecordCommand('stop_inference');
        if (!paused?.success) throw new Error(paused?.message || 'Could not pause inference');
      }
      await runCommand('reset');

      const deadline = Date.now() + RESET_TIMEOUT_MS;
      let current = null;
      while (Date.now() < deadline) {
        await wait(250);
        current = await fetchSimulatorStatus();
        if (resetHasFreshSamples(baseline, current)) break;
      }
      if (!resetHasFreshSamples(baseline, current)) {
        throw new Error('Reset timed out before fresh state and camera samples arrived');
      }
      await refreshStatus({ quiet: true });
      if (shouldResume) {
        const resumed = await sendRecordCommand('resume_inference');
        if (!resumed?.success) throw new Error(resumed?.message || 'Reset completed, but inference did not resume');
      }
      toast.success('Simulation reset complete');
    } catch (requestError) {
      toast.error(requestError.message);
    } finally {
      setBusy(false);
    }
  }, [inferencePhase, refreshStatus, runCommand, sendRecordCommand]);

  const stop = useCallback(async () => {
    setBusy(true);
    try {
      if (policyActive) {
        const finished = await sendRecordCommand('finish');
        if (!finished?.success) throw new Error(finished?.message || 'Could not unload policy');
      }
      await runCommand('stop');
      toast.success('UI-launched simulation session stopped');
    } catch (requestError) {
      toast.error(requestError.message);
    } finally {
      setBusy(false);
    }
  }, [policyActive, runCommand, sendRecordCommand]);

  const statusColor = status.state === 'error'
    ? 'bg-red-100 text-red-700'
    : ready
      ? 'bg-emerald-100 text-emerald-700'
      : active
        ? 'bg-amber-100 text-amber-700'
        : 'bg-gray-100 text-gray-600';

  return (
    <div className="mb-3 rounded-lg border border-indigo-200 bg-indigo-50/50 p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-sm font-semibold text-indigo-900">UI-launched simulation session</span>
        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${statusColor}`}>
          {status.state}
        </span>
      </div>
      <div className="mb-2 rounded-md border border-gray-200 bg-white px-2 py-1.5 text-xs text-gray-700">
        {displayedEnvironment ? (
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-semibold">{displayedEnvironmentLabel || displayedEnvironment}</span>
              <select
                aria-label="Reset profile"
                value={selectedProfile}
                onChange={(event) => dispatch(setInferenceTaskInfo({
                  simulationProfile: event.target.value,
                }))}
                disabled={active || profileOptions.length === 0}
                className="ml-auto h-7 rounded border border-gray-300 bg-white px-1 text-xs disabled:bg-gray-100"
              >
                {profileOptions.map((profile) => (
                  <option key={profile.key} value={profile.key}>{profile.label}</option>
                ))}
              </select>
            </div>
            <div className="mt-1 break-all font-mono text-[10px] text-gray-500">{displayedGymId}</div>
          </div>
        ) : (
          'Task and reset profile resolve from the selected policy contract.'
        )}
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        <button
          type="button"
          onClick={reset}
          disabled={!ready || busy || inferencePhase === InferencePhase.LOADING}
          className="h-9 rounded-md bg-emerald-600 text-white text-xs font-semibold disabled:opacity-40 flex items-center justify-center gap-1"
        >
          <MdRefresh size={16} /> Reset
        </button>
        <button
          type="button"
          onClick={stop}
          disabled={!active || busy}
          className="h-9 rounded-md bg-gray-700 text-white text-xs font-semibold disabled:opacity-40 flex items-center justify-center gap-1"
        >
          <MdStop size={16} /> Stop
        </button>
      </div>
      <p className="mt-2 text-xs leading-snug text-gray-600">
        {status.training_active
          ? 'Simulation launch is disabled while training is active.'
          : error || resolveError || status.message || 'Choose a policy checkpoint, then use Isaac Sim Deploy.'}
      </p>
    </div>
  );
}
