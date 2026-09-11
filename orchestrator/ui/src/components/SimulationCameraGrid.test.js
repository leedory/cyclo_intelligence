import React from 'react';
import { render, screen } from '@testing-library/react';
import SimulationCameraGrid from './SimulationCameraGrid';

jest.mock('./ImageGridCell', () => ({ topic, aspect }) => (
  <div data-testid={topic} data-aspect={aspect} />
));

describe('SimulationCameraGrid', () => {
  test('prioritizes the head view and keeps canonical simulation camera ratios', () => {
    const { container } = render(<SimulationCameraGrid />);
    const root = container.firstChild;
    const policyGrid = root.children[1];

    expect(root).toHaveClass('grid-cols-[minmax(0,3fr)_minmax(420px,2fr)]');
    expect(policyGrid).toHaveClass('grid-rows-[minmax(0,3fr)_minmax(0,2fr)]');
    expect(screen.getByTestId('/zed/zed_node/left/image_rect_color/compressed'))
      .toHaveAttribute('data-aspect', '672/376');
    expect(screen.getByTestId('/camera_left/camera_left/color/image_rect_raw/compressed'))
      .toHaveAttribute('data-aspect', '3/4');
    expect(screen.getByTestId('/camera_right/camera_right/color/image_rect_raw/compressed'))
      .toHaveAttribute('data-aspect', '3/4');
    expect(screen.getByText('Head · simulation view · 15 Hz')).toBeInTheDocument();
    expect(screen.getByText('Left wrist · simulation view · 15 Hz')).toBeInTheDocument();
    expect(screen.getByText('Right wrist · simulation view · 15 Hz')).toBeInTheDocument();
    expect(screen.queryByText(/policy input/i)).not.toBeInTheDocument();
  });
});
