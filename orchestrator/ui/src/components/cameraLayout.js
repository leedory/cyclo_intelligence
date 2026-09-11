// [left wrist (idx 0), head (idx 1), right wrist (idx 2)]
export const DEFAULT_POLICY_CAMERA_LAYOUT = [
  { aspect: '3/4', flexClass: 'flex-[3_1_0]' },
  { aspect: '16/9', flexClass: 'flex-[7_1_0]' },
  { aspect: '3/4', flexClass: 'flex-[3_1_0]' },
];

// SG2 publishes a 672x376 head image and 480x640 portrait wrist images.
// The 3:1:1 allocation keeps the head camera primary while preserving every
// camera's native display ratio (no stretching or preprocessing changes).
export const SG2_POLICY_CAMERA_LAYOUT = [
  { aspect: '3/4', flexClass: 'flex-[1_1_0]' },
  { aspect: '672/376', flexClass: 'flex-[3_1_0]' },
  { aspect: '3/4', flexClass: 'flex-[1_1_0]' },
];

export const SG2_SIMULATION_CAMERA_LAYOUT = {
  columnsClass: 'grid-cols-[minmax(0,3fr)_minmax(420px,2fr)]',
  policyRowsClass: 'grid-rows-[minmax(0,3fr)_minmax(0,2fr)]',
};

export const isSg2RobotType = (robotType) => (
  /^ffw_sg2(?:_|$)/.test(String(robotType || '').trim())
);

export const getPolicyCameraLayout = (robotType) => (
  isSg2RobotType(robotType)
    ? SG2_POLICY_CAMERA_LAYOUT
    : DEFAULT_POLICY_CAMERA_LAYOUT
);
