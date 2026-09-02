import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import SimulationControlPanel from './SimulationControlPanel';
import taskReducer from '../features/tasks/taskSlice';
import rosReducer from '../features/ros/rosSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

const mockRunCommand = jest.fn();
const mockRefreshStatus = jest.fn();
const mockFetchSimulatorStatus = jest.fn();
const mockRequestSimulator = jest.fn();

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return { __esModule: true, default: toast };
});

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('../hooks/useSimulatorStatus', () => ({
  __esModule: true,
  default: () => ({
    status: {
      state: 'ready',
      environment: 'task_000458',
      profile: 'deterministic',
      reset_count: 3,
      observation_sequence: 20,
      camera_sequence: 30,
      training_active: false,
    },
    environments: [{
      key: 'task_000458',
      label: 'Task 000458',
      profiles: [{ key: 'deterministic', label: 'Deterministic' }],
    }],
    error: '',
    refreshStatus: mockRefreshStatus,
    runCommand: mockRunCommand,
  }),
  fetchSimulatorStatus: (...args) => mockFetchSimulatorStatus(...args),
  requestSimulator: (...args) => mockRequestSimulator(...args),
}));

function renderPanel(phase, sendRecordCommand) {
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const tasks = taskReducer(undefined, { type: '@@INIT' });
  const ros = rosReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer, ros: rosReducer },
    preloadedState: {
      tasks: {
        ...tasks,
        inferenceStatus: { ...tasks.inferenceStatus, inferencePhase: phase },
      },
      ros,
    },
  });
  render(<Provider store={store}><SimulationControlPanel /></Provider>);
}

describe('SimulationControlPanel reset lifecycle', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRefreshStatus.mockResolvedValue({
      state: 'ready',
      reset_count: 3,
      observation_sequence: 20,
      camera_sequence: 30,
    });
    mockRunCommand.mockResolvedValue({ state: 'resetting' });
    mockFetchSimulatorStatus.mockResolvedValue({
      state: 'ready',
      reset_count: 4,
      observation_sequence: 21,
      camera_sequence: 31,
    });
    mockRequestSimulator.mockResolvedValue({
      environment: 'task_000458',
      environment_label: 'Task 000458',
      profile: 'deterministic',
      gym_id: 'Cyclo-Real-Showroom-Task000458-FFW-SG2-v0',
    });
  });

  test('pauses an active policy, waits for fresh samples, then resumes', async () => {
    const events = [];
    const sendRecordCommand = jest.fn(async (command) => {
      events.push(command);
      return { success: true };
    });
    mockRunCommand.mockImplementation(async (command) => {
      events.push(`sim:${command}`);
      return { state: 'resetting' };
    });
    mockFetchSimulatorStatus.mockImplementation(async () => {
      events.push('sim:fresh');
      return {
        state: 'ready',
        reset_count: 4,
        observation_sequence: 21,
        camera_sequence: 31,
      };
    });
    renderPanel(InferencePhase.INFERENCING, sendRecordCommand);

    fireEvent.click(screen.getByRole('button', { name: /reset/i }));

    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith('resume_inference'));
    expect(events).toEqual([
      'stop_inference',
      'sim:reset',
      'sim:fresh',
      'resume_inference',
    ]);
  });

  test('does not reset or resume when pause is not confirmed', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({
      success: false,
      message: 'pause rejected',
    });
    renderPanel(InferencePhase.INFERENCING, sendRecordCommand);

    fireEvent.click(screen.getByRole('button', { name: /reset/i }));

    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith('stop_inference'));
    expect(mockRunCommand).not.toHaveBeenCalled();
    expect(sendRecordCommand).not.toHaveBeenCalledWith('resume_inference');
  });
});
