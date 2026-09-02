import { waitForSimulatorReady } from './useSimulatorStatus';

describe('waitForSimulatorReady', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  test('fails immediately when a launched Cyclo Lab session stops', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({ state: 'stopped', message: '' }),
    });

    await expect(waitForSimulatorReady({ timeoutMs: 300000 })).rejects.toThrow(
      'Cyclo Lab simulation exited during startup'
    );
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });
});
