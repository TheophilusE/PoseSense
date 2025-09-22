import { Quaternion, Vector3 } from 'three';

export function quatFromArray(a: [number, number, number, number]): Quaternion {
  const q = new Quaternion(a[1], a[2], a[3], a[0]); // server [w,x,y,z] -> THREE (x,y,z,w)
  return q.normalize();
}

export function vec3FromArray(a: [number, number, number]): Vector3 {
  return new Vector3(a[0], a[1], a[2]);
}

export function slerpQuat(out: Quaternion, qa: Quaternion, qb: Quaternion, t: number): Quaternion {
  return out.copy(qa).slerp(qb, t);
}

export function lerpVec3(out: Vector3, va: Vector3, vb: Vector3, t: number): Vector3 {
  return out.copy(va).lerp(vb, t);
}

export function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x));
}

export class EMA {
  private alpha: number;
  private y: number;
  private initialized = false;

  constructor(alpha = 0.2, initial = 0) {
    this.alpha = alpha;
    this.y = initial;
  }

  update(x: number): number {
    this.y = this.initialized ? (this.alpha * x + (1 - this.alpha) * this.y) : x;
    this.initialized = true;
    return this.y;
  }

  value(): number { return this.y; }
}
