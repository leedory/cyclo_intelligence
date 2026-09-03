import React from 'react';
import ImageGridCell from './ImageGridCell';
import {
  SG2_POLICY_CAMERA_LAYOUT,
  SG2_SIMULATION_CAMERA_LAYOUT,
} from './cameraLayout';

const CAMERAS = {
  overhead: {
    label: 'Overhead',
    topic: '/camera_external/color/image_rect_raw/compressed',
    aspect: '1/1',
  },
  head: {
    label: 'Head · policy input · 15 Hz',
    topic: '/zed/zed_node/left/image_rect_color/compressed',
    aspect: SG2_POLICY_CAMERA_LAYOUT[1].aspect,
  },
  left: {
    label: 'Left wrist · policy input · 15 Hz',
    topic: '/camera_left/camera_left/color/image_rect_raw/compressed',
    aspect: SG2_POLICY_CAMERA_LAYOUT[0].aspect,
  },
  right: {
    label: 'Right wrist · policy input · 15 Hz',
    topic: '/camera_right/camera_right/color/image_rect_raw/compressed',
    aspect: SG2_POLICY_CAMERA_LAYOUT[2].aspect,
  },
};

function Camera({ camera, index, isActive }) {
  return (
    <div className="relative min-w-0 min-h-0 h-full overflow-hidden">
      <ImageGridCell
        topic={camera.topic}
        aspect={camera.aspect}
        idx={index}
        isActive={isActive}
        showControls={false}
        objectFit="contain"
        letterboxColor="#111827"
        onClose={() => {}}
        onPlusClick={() => {}}
        style={{ height: '100%', aspectRatio: 'auto' }}
      />
      <div className="absolute bottom-2 left-2 z-10 rounded bg-black/60 px-2 py-1 text-xs font-medium text-white">
        {camera.label}
      </div>
    </div>
  );
}

export default function SimulationCameraGrid({ isActive = true }) {
  return (
    <div className={`grid h-full w-full min-h-0 ${SG2_SIMULATION_CAMERA_LAYOUT.columnsClass} gap-2 overflow-hidden`}>
      <Camera camera={CAMERAS.overhead} index={3} isActive={isActive} />
      <div className={`grid min-h-0 ${SG2_SIMULATION_CAMERA_LAYOUT.policyRowsClass} gap-2 overflow-hidden`}>
        <Camera camera={CAMERAS.head} index={1} isActive={isActive} />
        <div className="grid min-h-0 grid-cols-2 gap-2 overflow-hidden">
          <Camera camera={CAMERAS.left} index={0} isActive={isActive} />
          <Camera camera={CAMERAS.right} index={2} isActive={isActive} />
        </div>
      </div>
    </div>
  );
}
