export interface JointRecord {
  name: string;
  rotation: [number, number, number, number]; // [w, x, y, z]
}

export interface RootPose {
  position: [number, number, number];
  rotation: [number, number, number, number]; // [w, x, y, z]
}

export interface PoseFrame {
  type: 'poseFrame';
  timestamp: number; // server ms since epoch
  skeleton: 'Mixamo' | string;
  frame: number;
  fps: number;
  root: RootPose;
  joints: JointRecord[];
  meta?: Record<string, unknown>;
}
