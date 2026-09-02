import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import toast from 'react-hot-toast';
import InferenceControlPanel from './InferenceControlPanel';
import taskReducer, { setInferenceStatus } from '../features/tasks/taskSlice';
import rosReducer from '../features/ros/rosSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  toast.dismiss = jest.fn();
  return {
    __esModule: true,
    default: toast,
    useToasterStore: () => ({ toasts: [] }),
  };
});

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('../hooks/usePolicyBackendStatus', () => ({
  __esModule: true,
  default: () => ({
    readiness: {
      ready: true,
      state: 'ready',
      message: 'Backend ready',
    },
    refreshStatus: jest.fn(),
  }),
  getPolicyBackendReadiness: (status) => status,
}));

const mockRunSimulatorCommand = jest.fn();
const mockRefreshSimulatorStatus = jest.fn();
const mockWaitForSimulatorReady = jest.fn();

jest.mock('../hooks/useSimulatorStatus', () => ({
  __esModule: true,
  default: () => ({
    status: {
      state: 'ready',
      observation_sequence: 1,
      camera_sequence: 1,
    },
    refreshStatus: mockRefreshSimulatorStatus,
    runCommand: mockRunSimulatorCommand,
  }),
  waitForSimulatorReady: (...args) => mockWaitForSimulatorReady(...args),
}));

const renderPanel = ({
  inferenceMode = 'robot',
  inferencePhase = InferencePhase.READY,
  taskOverrides = {},
  sendRecordCommand: sendOverride = null,
} = {}) => {
  const sendRecordCommand = sendOverride || jest.fn().mockResolvedValue({
    success: true,
    message: 'ok',
  });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });

  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const initialRos = rosReducer(undefined, { type: '@@INIT' });
  const { taskInstruction, ...inferenceOverrides } = taskOverrides;
  const sharedTaskInstruction =
    taskInstruction ?? initialTasks.sharedTaskInfo.taskInstruction;
  const store = configureStore({
    reducer: {
      tasks: taskReducer,
      ros: rosReducer,
    },
    preloadedState: {
      tasks: {
        ...initialTasks,
        sharedTaskInfo: {
          ...initialTasks.sharedTaskInfo,
          taskInstruction: sharedTaskInstruction,
        },
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          policyPath: '/workspace/model/lerobot/model',
          inferenceMode,
          ...inferenceOverrides,
        },
        taskInfo: {
          ...initialTasks.taskInfo,
          policyPath: '/workspace/model/lerobot/model',
          inferenceMode,
          ...taskOverrides,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
      },
      ros: {
        ...initialRos,
        rosHost: 'localhost',
      },
    },
  });

  render(
    <Provider store={store}>
      <InferenceControlPanel />
    </Provider>
  );

  return { store, sendRecordCommand };
};

describe('InferenceControlPanel deploy safety', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRunSimulatorCommand.mockResolvedValue({
      state: 'ready',
      environment: 'task_000458',
      observation_sequence: 1,
      camera_sequence: 1,
    });
    mockRefreshSimulatorStatus.mockResolvedValue({ state: 'ready' });
    mockWaitForSimulatorReady.mockResolvedValue({ state: 'ready' });
  });

  test('one click launches the policy contract environment then starts Isaac inference', async () => {
    const { sendRecordCommand } = renderPanel({ inferenceMode: 'simulation' });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(mockRunSimulatorCommand).toHaveBeenCalledWith('start', {
        policy_path: '/workspace/model/lerobot/model',
      });
      expect(mockWaitForSimulatorReady).toHaveBeenCalled();
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'isaac',
      });
    });
  });

  test('does not expose real robot or browser-preview deployment controls', () => {
    renderPanel({ inferenceMode: 'robot' });

    expect(screen.queryByText(/Real Robot Deploy/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/3D Sim Deploy/i)).not.toBeInTheDocument();
  });

  test('keeps loading state when start command times out after LOADING begins', async () => {
    let rejectStart;
    const sendRecordCommand = jest.fn(() => new Promise((_, reject) => {
      rejectStart = reject;
    }));
    const { store } = renderPanel({
      inferenceMode: 'simulation',
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'isaac',
      });
    });

    act(() => {
      store.dispatch(setInferenceStatus({ inferencePhase: InferencePhase.LOADING }));
      rejectStart(new Error('Service call timeout for /task/command'));
    });

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        'Model loading is still running. Large downloads can take several minutes.'
      );
    });
    expect(toast.error).not.toHaveBeenCalledWith(
      expect.stringContaining('Command timeout')
    );
    expect(store.getState().tasks.inferenceStatus.inferencePhase)
      .toBe(InferencePhase.LOADING);
  });

  test('does not expose recording controls on the inference panel', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation',
      inferencePhase: InferencePhase.INFERENCING,
    });

    expect(screen.queryByRole('button', { name: /start recording/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /save recording/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /discard recording/i })).not.toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'r', code: 'KeyR' });
    fireEvent.keyUp(window, { key: 'r', code: 'KeyR' });

    await waitFor(() => {
      expect(sendRecordCommand).not.toHaveBeenCalled();
    });
  });

  test('requires shared task instruction for language-conditioned inference', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation',
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        taskInstruction: [],
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Missing required fields: Task Instruction');
      expect(sendRecordCommand).not.toHaveBeenCalled();
    });
  });

  test('allows language-conditioned inference when shared task instruction is set', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation',
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        taskInstruction: ['pick up the red cup'],
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'isaac',
      });
    });
  });
});
