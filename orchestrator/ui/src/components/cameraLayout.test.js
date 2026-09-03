import {
  DEFAULT_POLICY_CAMERA_LAYOUT,
  SG2_POLICY_CAMERA_LAYOUT,
  getPolicyCameraLayout,
  isSg2RobotType,
} from './cameraLayout';

describe('policy camera layout', () => {
  test('uses canonical SG2 ratios and gives the head the primary panel', () => {
    expect(getPolicyCameraLayout('ffw_sg2_rev1')).toBe(SG2_POLICY_CAMERA_LAYOUT);
    expect(SG2_POLICY_CAMERA_LAYOUT).toEqual([
      { aspect: '3/4', flexClass: 'flex-[1_1_0]' },
      { aspect: '672/376', flexClass: 'flex-[3_1_0]' },
      { aspect: '3/4', flexClass: 'flex-[1_1_0]' },
    ]);
  });

  test('recognizes SG2 revisions without changing other robot layouts', () => {
    expect(isSg2RobotType('ffw_sg2_rev2')).toBe(true);
    expect(getPolicyCameraLayout('ffw_sh5_rev1')).toBe(DEFAULT_POLICY_CAMERA_LAYOUT);
    expect(getPolicyCameraLayout('')).toBe(DEFAULT_POLICY_CAMERA_LAYOUT);
  });
});
