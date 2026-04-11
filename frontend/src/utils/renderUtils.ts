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
            if (parent && (parent as any).isBone) parent.getWorldPosition(b); else b.copy(a);
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
    private _segments: Array<[string, string]> = [
        ['Pelvis', 'Chest'],
        ['Pelvis', 'LeftHip'],
        ['LeftHip', 'LeftKnee'],
        ['LeftKnee', 'LeftAnkle'],
        ['Pelvis', 'RightHip'],
        ['RightHip', 'RightKnee'],
        ['RightKnee', 'RightAnkle'],
        ['Chest', 'LeftShoulder'],
        ['LeftShoulder', 'LeftElbow'],
        ['LeftElbow', 'LeftWrist'],
        ['Chest', 'RightShoulder'],
        ['RightShoulder', 'RightElbow'],
        ['RightElbow', 'RightWrist'],
    ];

    private _vA = new THREE.Vector3();
    private _vB = new THREE.Vector3();
    private _vC = new THREE.Vector3();
    private _vD = new THREE.Vector3();
    private _vE = new THREE.Vector3();
    private _vF = new THREE.Vector3();
    private _vG = new THREE.Vector3();
    private _qA = new THREE.Quaternion();
    private _qB = new THREE.Quaternion();

    constructor(names: string[]) {
        this.names = names.slice();
        this._pos = new Float32Array(this._segments.length * 2 * 3);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(this._pos, 3));
        const mat = new THREE.LineBasicMaterial({ color: 0xff6b6b });
        this.mesh = new THREE.LineSegments(geom, mat);
        this.mesh.frustumCulled = false;
        // Initialize positions to zero
        for (let i = 0; i < this._pos.length; ++i) this._pos[i] = 0;
        (this.mesh.geometry as any).computeBoundingSphere?.();
    }

    private _quatFromWxyz(arr: any, out: THREE.Quaternion): THREE.Quaternion {
        if (Array.isArray(arr) && arr.length === 4) {
            out.set(Number(arr[1]), Number(arr[2]), Number(arr[3]), Number(arr[0])).normalize();
            return out;
        }
        return out.identity();
    }

    // Draw an actual connected canonical skeleton from server pose data.
    updateFromPose(frame: any, bonesByName?: Map<string, THREE.Object3D>) {
        const attr = this.mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
        const arr = attr.array as Float32Array;

        const rootPos = this._vA.set(0, 1, 0);
        if (frame && frame.root && Array.isArray(frame.root.position) && frame.root.position.length >= 3) {
            rootPos.set(Number(frame.root.position[0]), Number(frame.root.position[1]), Number(frame.root.position[2]));
        }

        const rootQ = this._quatFromWxyz(frame?.root?.rotation, this._qA);

        const scaleRaw = Number(frame?.meta?.scale);
        const baseScale = Number.isFinite(scaleRaw) ? Math.max(0.08, scaleRaw) : 0.23;
        const spineLen = baseScale * 1.45;
        const hipOffset = baseScale * 0.52;
        const shoulderOffset = baseScale * 0.62;
        const upperLegLen = baseScale * 1.75;
        const lowerLegLen = baseScale * 1.70;
        const upperArmLen = baseScale * 1.40;
        const lowerArmLen = baseScale * 1.30;

        const jointMap = new Map<string, THREE.Quaternion>();
        if (frame && Array.isArray(frame.joints)) {
            for (const j of frame.joints) {
                const qq = new THREE.Quaternion();
                this._quatFromWxyz(j?.rotation, qq);
                jointMap.set(String(j?.name ?? ''), qq);
            }
        }

        // Auto-correct upside-down input coordinate frames for visualization.
        const frameFlip = this._qB.identity();
        let shouldFlip = false;
        const spineProbeQ = jointMap.get('Spine');
        if (spineProbeQ) {
            const spineDirProbe = this._vD.set(0, 1, 0).applyQuaternion(spineProbeQ);
            shouldFlip = spineDirProbe.y < 0;
        } else {
            const upProbe = this._vD.set(0, 1, 0).applyQuaternion(rootQ);
            shouldFlip = upProbe.y < 0;
        }
        if (shouldFlip) {
            frameFlip.setFromAxisAngle(this._vE.set(1, 0, 0), Math.PI);
            rootPos.y *= -1;
        }

        const right = this._vB.set(1, 0, 0).applyQuaternion(rootQ);
        if (shouldFlip) right.applyQuaternion(frameFlip);
        right.normalize();
        const left = this._vC.copy(right).multiplyScalar(-1);

        const dirFromJoint = (jointName: string, restDir: THREE.Vector3): THREE.Vector3 => {
            const q = jointMap.get(jointName);
            const dir = this._vF.copy(restDir);
            if (!q) {
                dir.applyQuaternion(rootQ);
            } else {
                dir.applyQuaternion(q);
            }
            if (shouldFlip) dir.applyQuaternion(frameFlip);
            return dir.normalize();
        };

        const pelvis = this._vA;
        const chest = this._vD.copy(pelvis).addScaledVector(dirFromJoint('Spine', new THREE.Vector3(0, 1, 0)), spineLen);

        const leftHip = this._vE.copy(pelvis).addScaledVector(left, hipOffset);
        const leftKnee = leftHip.clone().addScaledVector(dirFromJoint('LeftUpLeg', new THREE.Vector3(0, -1, 0)), upperLegLen);
        const leftAnkle = leftKnee.clone().addScaledVector(dirFromJoint('LeftLeg', new THREE.Vector3(0, -1, 0)), lowerLegLen);

        const rightHip = pelvis.clone().addScaledVector(right, hipOffset);
        const rightKnee = rightHip.clone().addScaledVector(dirFromJoint('RightUpLeg', new THREE.Vector3(0, -1, 0)), upperLegLen);
        const rightAnkle = rightKnee.clone().addScaledVector(dirFromJoint('RightLeg', new THREE.Vector3(0, -1, 0)), lowerLegLen);

        const leftShoulder = chest.clone().addScaledVector(left, shoulderOffset);
        const leftElbow = leftShoulder.clone().addScaledVector(dirFromJoint('LeftArm', new THREE.Vector3(-1, 0, 0)), upperArmLen);
        const leftWrist = leftElbow.clone().addScaledVector(dirFromJoint('LeftForeArm', new THREE.Vector3(-1, 0, 0)), lowerArmLen);

        const rightShoulder = chest.clone().addScaledVector(right, shoulderOffset);
        const rightElbow = rightShoulder.clone().addScaledVector(dirFromJoint('RightArm', new THREE.Vector3(1, 0, 0)), upperArmLen);
        const rightWrist = rightElbow.clone().addScaledVector(dirFromJoint('RightForeArm', new THREE.Vector3(1, 0, 0)), lowerArmLen);

        const points = new Map<string, THREE.Vector3>([
            ['Pelvis', pelvis],
            ['Chest', chest],
            ['LeftHip', leftHip],
            ['LeftKnee', leftKnee],
            ['LeftAnkle', leftAnkle],
            ['RightHip', rightHip],
            ['RightKnee', rightKnee],
            ['RightAnkle', rightAnkle],
            ['LeftShoulder', leftShoulder],
            ['LeftElbow', leftElbow],
            ['LeftWrist', leftWrist],
            ['RightShoulder', rightShoulder],
            ['RightElbow', rightElbow],
            ['RightWrist', rightWrist],
        ]);

        // Vertically align server overlay to the loaded mesh hips so the
        // helper sits on the character instead of around world-origin height.
        let yOffset = 0;
        if (bonesByName) {
            const hips = bonesByName.get('Hips')
                ?? bonesByName.get('mixamorig:Hips')
                ?? bonesByName.get('mixamorigHips')
                ?? bonesByName.get('mixamorighips');
            if (hips) {
                hips.getWorldPosition(this._vG);
                const pelvisPoint = points.get('Pelvis') ?? rootPos;
                yOffset = this._vG.y - pelvisPoint.y;
            }
        }

        let i = 0;
        for (const [aName, bName] of this._segments) {
            const pa = points.get(aName) ?? rootPos;
            const pb = points.get(bName) ?? rootPos;

            arr[i++] = pa.x; arr[i++] = pa.y + yOffset; arr[i++] = pa.z;
            arr[i++] = pb.x; arr[i++] = pb.y + yOffset; arr[i++] = pb.z;
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
