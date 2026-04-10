import * as THREE from 'three';

export class SimpleSkeletonHelper {
    mesh: THREE.LineSegments;
    bones: THREE.Object3D[];
    private _pos: Float32Array;

    constructor(skinned: THREE.SkinnedMesh) {
        this.bones = skinned.skeleton ? skinned.skeleton.bones.slice() : [];
        this._pos = new Float32Array(this.bones.length * 2 * 3);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(this._pos, 3));
        const mat = new THREE.LineBasicMaterial({ color: 0x1f73ff });
        this.mesh = new THREE.LineSegments(geom, mat);
        this.mesh.frustumCulled = false;
        this.update();
    }

    update() {
        const attr = this.mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
        const arr = attr.array as Float32Array;
        const a = new THREE.Vector3();
        const b = new THREE.Vector3();
        let i = 0;
        for (const bone of this.bones) {
            bone.getWorldPosition(a);
            const parent = bone.parent as THREE.Object3D | null;
            if (parent) parent.getWorldPosition(b); else b.copy(a);
            arr[i++] = a.x; arr[i++] = a.y; arr[i++] = a.z;
            arr[i++] = b.x; arr[i++] = b.y; arr[i++] = b.z;
        }
        attr.needsUpdate = true;
        if ((this.mesh.geometry as any).computeBoundingSphere) (this.mesh.geometry as any).computeBoundingSphere();
    }

    set visible(v: boolean) { this.mesh.visible = v; }
    get visible() { return this.mesh.visible; }

    dispose() {
        this.mesh.geometry.dispose();
        (this.mesh.material as any).dispose();
    }
}

export function setWireframeForObject(obj: THREE.Object3D, enabled: boolean) {
    obj.traverse((o) => {
        const m = (o as any).material as THREE.Material | THREE.Material[] | undefined;
        if (!m) return;
        if (Array.isArray(m)) {
            for (const mm of m) {
                if ((mm as any).wireframe !== undefined) (mm as any).wireframe = enabled;
            }
        } else {
            if ((m as any).wireframe !== undefined) (m as any).wireframe = enabled;
        }
    });
}

/**
 * ServerSkeletonHelper
 * Renders a simple line for each server joint showing the joint origin and
 * a short local forward/up axis derived from the joint quaternion. When a
 * model bone map is supplied (name -> Object3D), the helper will place the
 * joint origins at the model bone world positions so you can compare server
 * joint orientations to the loaded model.
 */
export class ServerSkeletonHelper {
    mesh: THREE.LineSegments;
    names: string[];
    private _pos: Float32Array;

    constructor(names: string[]) {
        this.names = names.slice();
        this._pos = new Float32Array(this.names.length * 2 * 3);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(this._pos, 3));
        const mat = new THREE.LineBasicMaterial({ color: 0xff6b6b });
        this.mesh = new THREE.LineSegments(geom, mat);
        this.mesh.frustumCulled = false;
        // Initialize positions to zero
        for (let i = 0; i < this._pos.length; ++i) this._pos[i] = 0;
        (this.mesh.geometry as any).computeBoundingSphere?.();
    }

    /**
     * Update helper from a PoseFrame-like object.
     * frame.joints: Array<{ name: string; rotation: [w,x,y,z] }>
     * frame.root.position: optional [x,y,z] used as fallback anchor
     * bonesByName: optional Map<string, Object3D> to position joints at model bones
     */
    updateFromPose(frame: any, bonesByName?: Map<string, THREE.Object3D>) {
        const attr = this.mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
        const arr = attr.array as Float32Array;
        const tmpPosA = new THREE.Vector3();
        const tmpPosB = new THREE.Vector3();
        const q = new THREE.Quaternion();
        const forward = new THREE.Vector3(0, 0.12, 0);

        const rootPos = (frame && frame.root && Array.isArray(frame.root.position))
            ? new THREE.Vector3(frame.root.position[0], frame.root.position[1], frame.root.position[2])
            : new THREE.Vector3(0, 1, 0);

        // build a fallback layout offset so joints don't overlap when no model
        const fallbackOffsets: Record<string, THREE.Vector3> = {};
        for (let i = 0; i < this.names.length; ++i) {
            const name = this.names[i] ?? `j${i}`;
            fallbackOffsets[name] = new THREE.Vector3(((i % 6) - 2.5) * 0.18, Math.floor(i / 6) * 0.14 + rootPos.y, ((i % 3) - 1) * 0.12);
        }

        // Map joint name -> rotation quaternion for quick lookup
        const jointMap = new Map<string, any>();
        if (frame && Array.isArray(frame.joints)) {
            for (const j of frame.joints) jointMap.set(j.name, j.rotation);
        }

        let i = 0;
        for (const name of this.names) {
            const rot = jointMap.get(name);
            // position origin: prefer model bone world position when available
            const bone = bonesByName?.get(name) ?? bonesByName?.get(`mixamorig:${name}`);
            if (bone) {
                bone.getWorldPosition(tmpPosA);
            } else {
                tmpPosA.copy(rootPos).add(fallbackOffsets[name] ?? new THREE.Vector3());
            }

            if (rot && Array.isArray(rot) && rot.length === 4) {
                q.set(rot[0], rot[1], rot[2], rot[3]);
            } else {
                q.identity();
            }

            tmpPosB.copy(forward).applyQuaternion(q).add(tmpPosA);

            arr[i++] = tmpPosA.x; arr[i++] = tmpPosA.y; arr[i++] = tmpPosA.z;
            arr[i++] = tmpPosB.x; arr[i++] = tmpPosB.y; arr[i++] = tmpPosB.z;
        }

        attr.needsUpdate = true;
        (this.mesh.geometry as any).computeBoundingSphere?.();
    }

    set visible(v: boolean) { this.mesh.visible = v; }
    get visible() { return this.mesh.visible; }

    dispose() {
        this.mesh.geometry.dispose();
        (this.mesh.material as any).dispose();
    }
}
