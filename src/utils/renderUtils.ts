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
