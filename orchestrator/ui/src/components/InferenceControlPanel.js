// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Dongyun Kim

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { shallowEqual, useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import toast, { useToasterStore } from 'react-hot-toast';
import {
  MdPlayArrow,
  MdStop,
  MdDeleteOutline,
} from 'react-icons/md';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import Tooltip from './Tooltip';
import { InferencePhase } from '../constants/taskPhases';
import { selectInferenceTaskInfo, setInferenceStatus } from '../features/tasks/taskSlice';
import { requiresInstruction } from '../constants/policyCapabilities';
import usePolicyBackendStatus, {
  getPolicyBackendReadiness,
} from '../hooks/usePolicyBackendStatus';
import useSimulatorStatus, { waitForSimulatorReady } from '../hooks/useSimulatorStatus';

const phaseGuideMessages = {
  [InferencePhase.READY]: 'Ready to start',
  [InferencePhase.LOADING]: 'Loading model / downloading assets...',
  [InferencePhase.INFERENCING]: 'Inferencing',
  [InferencePhase.PAUSED]: 'Paused',
};

const buildRequiredFields = (serviceType, policyType) => {
  const fields = [{ key: 'policyPath', label: 'Policy Path' }];
  if (requiresInstruction(serviceType, policyType)) {
    fields.unshift({ key: 'taskInstruction', label: 'Task Instruction' });
  }
  return fields;
};

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
const API_BASE = '/api';

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function buildTrtStatusUrl(modelPath, enginePath) {
  const params = new URLSearchParams();
  params.set('model_path', modelPath);
  if (enginePath) params.set('engine_path', enginePath);
  return `${API_BASE}/backends/groot/trt/status?${params.toString()}`;
}

export default function InferenceControlPanel() {
  const dispatch = useDispatch();
  const taskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const inferenceStatus = useSelector((state) => state.tasks.inferenceStatus);
  const rosHost = useSelector((state) => state.ros.rosHost);

  const [hovered, setHovered] = useState(null);
  const [pressed, setPressed] = useState(null);
  const [lastPolicyPath, setLastPolicyPath] = useState('');
  const [spinnerIndex, setSpinnerIndex] = useState(0);
  const [isDeploying, setIsDeploying] = useState(false);

  const { sendRecordCommand } = useRosServiceCaller();

  const { toasts } = useToasterStore();
  const TOAST_LIMIT = 3;

  const phase = inferenceStatus.inferencePhase;
  const isIdle = phase === InferencePhase.READY;
  const isLoading = phase === InferencePhase.LOADING;
  const isInferencing = phase === InferencePhase.INFERENCING;
  const isPaused = phase === InferencePhase.PAUSED;
  const inferencePhaseRef = useRef(phase);
  const isModelLoaded = isInferencing || isPaused;
  const shouldCheckBackend = isIdle || isPaused;

  const {
    readiness: backendReadiness,
    refreshStatus: refreshBackendStatus,
  } = usePolicyBackendStatus(taskInfo.serviceType, {
    enabled: shouldCheckBackend,
    intervalMs: 2000,
  });

  const isBackendStartBlocked = shouldCheckBackend && !backendReadiness.ready;
  const isBackendWarming = isBackendStartBlocked &&
    (backendReadiness.state === 'checking' || backendReadiness.state === 'warming');
  const {
    status: simulatorStatus,
    refreshStatus: refreshSimulatorStatus,
    runCommand: runSimulatorCommand,
  } = useSimulatorStatus({ intervalMs: 1000 });
  const isSimulatorReady = ['ready', 'running'].includes(simulatorStatus.state);

  useEffect(() => {
    inferencePhaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    toasts
      .filter((t) => t.visible)
      .filter((_, i) => i >= TOAST_LIMIT)
      .forEach((t) => toast.dismiss(t.id));
  }, [toasts]);

  // Spin while loading / inferencing — independent of taskStatus update
  // cadence. The old (`}, [taskStatus]`) pattern only ticked when a new
  // TaskStatus message arrived, which happens once per phase transition
  // in cyclo_intelligence (LOADING → INFERENCING) — so the spinner
  // sat frozen between transitions. setInterval gives a steady visual
  // beat while either banner is on screen.
  useEffect(() => {
    if (!isLoading && !isInferencing && !isBackendWarming) return undefined;
    const id = setInterval(() => {
      setSpinnerIndex((prev) => (prev + 1) % spinnerFrames.length);
    }, 100);
    return () => clearInterval(id);
  }, [isLoading, isInferencing, isBackendWarming]);

  const validateTaskInfo = useCallback(() => {
    const missingFields = [];
    const fields = buildRequiredFields(taskInfo.serviceType, taskInfo.policyType);
    for (const field of fields) {
      const value = taskInfo[field.key];
      if (
        value === null ||
        value === undefined ||
        value === '' ||
        (typeof value === 'string' && value.trim() === '') ||
        (Array.isArray(value) && value.length === 0) ||
        (Array.isArray(value) && value.every((item) => item.trim() === ''))
      ) {
        missingFields.push(field.label);
      }
    }
    return { isValid: missingFields.length === 0, missingFields };
  }, [taskInfo]);

  const ensureTensorRtReady = useCallback(async () => {
    if (
      taskInfo.serviceType !== 'groot' ||
      taskInfo.accelerationMode !== 'tensorrt_dit'
    ) {
      return true;
    }
    const policyPath = String(taskInfo.policyPath || '').trim();
    if (!policyPath) return true;
    try {
      const response = await fetch(buildTrtStatusUrl(
        policyPath,
        String(taskInfo.accelerationEnginePath || '').trim()
      ));
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || data.message || `status failed (${response.status})`);
      }
      if (data.status === 'ready') return true;
      toast.error(
        data.status === 'building'
          ? 'TRT engine is still building'
          : 'Build TRT engine before starting inference'
      );
      return false;
    } catch (error) {
      toast.error(`TRT engine status failed: ${error.message}`);
      return false;
    }
  }, [
    taskInfo.serviceType,
    taskInfo.accelerationMode,
    taskInfo.policyPath,
    taskInfo.accelerationEnginePath,
  ]);

  const executeCommand = useCallback(
    async (commandName, commandString, options = {}) => {
      const isStartTimeoutDuringLoading = (message = '') => (
        commandString === 'start_inference' &&
        String(message).toLowerCase().includes('timeout') &&
        inferencePhaseRef.current === InferencePhase.LOADING
      );

      try {
        const result = await sendRecordCommand(commandString, options);
        if (result && result.success === false) {
          if (isStartTimeoutDuringLoading(result.message)) {
            toast('Model loading is still running. Large downloads can take several minutes.');
            return result;
          }
          toast.error(`Command failed: ${result.message || 'Unknown error'}`);
          // Backend may have left phase in LOADING/INFERENCING after a failed
          // setup; force the local phase back to READY so the panel becomes
          // editable and the user can retry.
          dispatch(setInferenceStatus({ inferencePhase: InferencePhase.READY }));
        } else if (result && result.success === true) {
          toast.success(`${commandName} executed successfully`);
        } else {
          toast.error(`${commandName} completed with uncertain status`);
          dispatch(setInferenceStatus({ inferencePhase: InferencePhase.READY }));
        }
        return result;
      } catch (error) {
        let errorMessage = error.message || error.toString();
        if (
          errorMessage.includes('ROS connection failed') ||
          errorMessage.includes('WebSocket')
        ) {
          toast.error(`ROS connection failed: rosbridge server is not running (${rosHost})`);
        } else if (isStartTimeoutDuringLoading(errorMessage)) {
          toast('Model loading is still running. Large downloads can take several minutes.');
          return {
            success: true,
            message: 'Model loading is still running',
          };
        } else if (errorMessage.includes('timeout')) {
          toast.error(`Command timeout [${commandName}]: Server did not respond`);
        } else {
          toast.error(`Command failed [${commandName}]: ${errorMessage}`);
        }
        // Same reasoning as the success===false branch above.
        dispatch(setInferenceStatus({ inferencePhase: InferencePhase.READY }));
        return null;
      }
    },
    [sendRecordCommand, rosHost, dispatch]
  );

  const executeStartIntent = useCallback(async (intent, inferenceMode) => {
    if (!intent) return;
    if (intent.policyPath) {
      setLastPolicyPath(intent.policyPath);
    }
    await executeCommand(intent.commandName, intent.commandString, {
      inferenceMode,
    });
  }, [executeCommand]);

  const handleStart = useCallback(async () => {
    if (isDeploying) return;
    setIsDeploying(true);
    try {
      let readiness = backendReadiness;
      if (!readiness.ready) {
        const refreshedStatus = await refreshBackendStatus({ quiet: true });
        readiness = getPolicyBackendReadiness(refreshedStatus);
      }
      if (!readiness.ready) {
        const message = readiness.message || 'Policy backend is not ready yet';
        if (readiness.state === 'warming' || readiness.state === 'checking') {
          toast(message);
        } else {
          toast.error(message);
        }
        return;
      }

      let startIntent;
      if (isPaused && taskInfo.policyPath === lastPolicyPath) {
        startIntent = {
          commandName: 'Resume',
          commandString: 'resume_inference',
          policyPath: '',
        };
      } else {
        const validation = validateTaskInfo();
        if (!validation.isValid) {
          toast.error(`Missing required fields: ${validation.missingFields.join(', ')}`);
          return;
        }
        startIntent = {
          commandName: 'Start Inference',
          commandString: 'start_inference',
          policyPath: taskInfo.policyPath,
        };
      }

      if (
        startIntent.commandString === 'start_inference' &&
        !(await ensureTensorRtReady())
      ) {
        return;
      }

      try {
        const launchStatus = await runSimulatorCommand('start', {
          policy_path: taskInfo.policyPath,
          ...(taskInfo.simulationProfile ? { profile: taskInfo.simulationProfile } : {}),
        });
        if (!['ready', 'running'].includes(launchStatus.state)) {
          toast('Launching UI-launched simulation session…');
        }
        await waitForSimulatorReady();
        await refreshSimulatorStatus({ quiet: true });
      } catch (error) {
        toast.error(`Isaac Sim Deploy failed: ${error.message}`);
        return;
      }

      await executeStartIntent(startIntent, 'isaac');
    } finally {
      setIsDeploying(false);
    }
  }, [
    isDeploying,
    refreshSimulatorStatus,
    runSimulatorCommand,
    backendReadiness,
    refreshBackendStatus,
    isPaused,
    taskInfo.policyPath,
    taskInfo.simulationProfile,
    lastPolicyPath,
    executeStartIntent,
    ensureTensorRtReady,
    validateTaskInfo,
  ]);

  const handleStop = useCallback(async () => {
    await executeCommand('Stop', 'stop_inference');
  }, [executeCommand]);

  const handleClear = useCallback(async () => {
    await executeCommand('Clear', 'finish');
    setLastPolicyPath('');
  }, [executeCommand]);

  const startEnabled = shouldCheckBackend && backendReadiness.ready &&
    !simulatorStatus.training_active && !isDeploying;
  const stopEnabled = isInferencing;
  const clearEnabled = isModelLoaded;
  const startDescription = isDeploying
    ? 'Isaac Sim Deploy is starting'
    : simulatorStatus.training_active
    ? 'Isaac Sim Deploy is disabled while training is active'
    : !isSimulatorReady
    ? 'Start inference and launch Isaac Sim from the policy contract'
    : isBackendStartBlocked
    ? backendReadiness.message
    : isPaused
      ? 'Resume inference'
      : 'Start inference';
  const guideMessage = isDeploying
    ? 'Launching Isaac Sim…'
    : simulatorStatus.training_active
    ? 'Training active'
    : !isSimulatorReady
    ? 'Start launches Isaac Sim'
    : isBackendStartBlocked
    ? backendReadiness.message
    : phaseGuideMessages[phase] || '';
  const showGuideSpinner = isDeploying || isInferencing || isLoading || isBackendWarming;

  const handleKeyAction = useCallback(
    (e) => {
      if (e.key === ' ' || e.key === 'Spacebar' || e.code === 'Space') {
        if (startEnabled) return 'Isaac Sim Deploy';
      }
      if (
        (e.ctrlKey || e.metaKey) &&
        e.shiftKey &&
        (e.key === 's' || e.key === 'S')
      ) {
        if (stopEnabled) return 'Stop';
      }
      if (e.key === 'Escape') {
        if (clearEnabled) return 'Clear';
      }
      return null;
    },
    [startEnabled, stopEnabled, clearEnabled]
  );

  useEffect(() => {
    const isInputFocused = () => {
      const el = document.activeElement;
      if (!el) return false;
      const tag = el.tagName.toLowerCase();
      return tag === 'input' || tag === 'textarea' || tag === 'select' ||
        el.contentEditable === 'true';
    };

    const handleKeyDown = (e) => {
      if (e.repeat || isInputFocused()) return;
      const action = handleKeyAction(e);
      if (action) {
        e.preventDefault();
        setPressed(action);
      }
    };

    const handleKeyUp = (e) => {
      setPressed(null);
      if (isInputFocused()) return;
      const action = handleKeyAction(e);
      if (action === 'Isaac Sim Deploy') handleStart();
      else if (action === 'Stop') handleStop();
      else if (action === 'Clear') handleClear();
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleKeyAction, handleStart, handleStop, handleClear]);

  const classBody = clsx(
    'bg-white/90',
    'backdrop-blur-sm',
    'rounded-full',
    'px-3',
    'py-1',
    'flex',
    'flex-row',
    'items-center',
    'gap-1.5',
    'shadow-md',
    'border',
    'border-gray-100'
  );

  const classBtn = (label, isDisabled) =>
    clsx(
      'h-full',
      'rounded-lg',
      'border-none',
      'cursor-pointer',
      'px-2.5',
      'flex',
      'items-center',
      'justify-center',
      'gap-1',
      'bg-gray-100',
      'transition-all',
      'duration-150',
      'font-semibold',
      'text-lg',
      'shrink-0',
      {
        'bg-gray-400': pressed === label && !isDisabled,
        'bg-gray-200': hovered === label && pressed !== label && !isDisabled,
        'opacity-30 cursor-not-allowed bg-gray-50': isDisabled,
      }
    );

  const controlButtons = [
    {
      label: 'Isaac Sim Deploy',
      icon: MdPlayArrow,
      color: '#1976d2',
      enabled: startEnabled,
      handler: handleStart,
      description: startDescription,
      shortcut: 'Space',
    },
    {
      label: 'Stop',
      icon: MdStop,
      color: '#f57c00',
      enabled: stopEnabled,
      handler: handleStop,
      description: 'Pause inference (model stays loaded)',
      shortcut: 'Ctrl+Shift+S',
    },
    {
      label: 'Clear',
      icon: MdDeleteOutline,
      color: '#d32f2f',
      enabled: clearEnabled,
      handler: handleClear,
      description: 'Stop inference and unload model',
      shortcut: 'Escape',
    },
  ];

  return (
      <div className={classBody}>
        <span className="text-lg font-semibold text-gray-500 whitespace-nowrap px-1 shrink-0">Inference</span>
        <div className="w-px h-2/3 bg-gray-300 shrink-0"></div>
        {controlButtons.map(({ label, icon: Icon, color, enabled, handler, description, shortcut }) => {
          const isDisabled = !enabled;
          return (
            <Tooltip
              key={label}
              position="bottom"
              content={
                <div className="text-center">
                  <div className="font-semibold">{description}</div>
                  {!isDisabled && (
                    <div className="text-sm mt-1 text-gray-300">
                      <span className="font-mono bg-gray-700 px-1 rounded">{shortcut}</span>
                    </div>
                  )}
                </div>
              }
              disabled={false}
              className="relative h-full shrink-0"
            >
              <button
                className={classBtn(label, isDisabled)}
                onClick={() => !isDisabled && handler()}
                onMouseEnter={() => !isDisabled && setHovered(label)}
                onMouseLeave={() => { setHovered(null); setPressed(null); }}
                onMouseDown={() => !isDisabled && setPressed(label)}
                onMouseUp={() => setPressed(null)}
                disabled={isDisabled}
                aria-label={description}
              >
                <Icon
                  style={{ fontSize: '1.1rem' }}
                  color={isDisabled ? '#9ca3af' : color}
                />
                {label}
              </button>
            </Tooltip>
          );
        })}

        {(guideMessage || showGuideSpinner) && (
          <>
            <div className="w-px h-2/3 bg-gray-400 shrink-0"></div>
            <div className="flex items-center gap-1 shrink-0">
              <span className="text-gray-600 font-semibold text-lg whitespace-nowrap">
                {guideMessage}
              </span>
              {showGuideSpinner && (
                <span className="font-mono text-blue-500 text-sm">
                  {spinnerFrames[spinnerIndex]}
                </span>
              )}
            </div>
          </>
        )}
      </div>
  );
}
