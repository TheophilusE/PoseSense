export interface JointRecord {
  name: string;
  rotation: [number, number, number, number]; // [w, x, y, z]
}

export type Vec3Tuple = [number, number, number];

export interface PoseFrameMeta {
  scale?: number;
  model_space?: string;
  landmark_schema?: string;
  pose_landmarks_2d?: [number, number, number][];
  intermediate_targets?: Record<string, Vec3Tuple>;
  intermediate_targets_raw?: Record<string, Vec3Tuple>;
  intermediate_targets_normalized?: Record<string, Vec3Tuple>;
  segment_directions?: Record<string, Vec3Tuple>;
  [key: string]: unknown;
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
  meta?: PoseFrameMeta;
}
