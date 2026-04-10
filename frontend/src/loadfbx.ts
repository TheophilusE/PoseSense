import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";

export async function loadYBotFbx(url = "/models/y-bot/y-bot.fbx"): Promise<THREE.Object3D> {
  const loader = new FBXLoader();
  const model = await new Promise<THREE.Group>((resolve, reject) => {
    loader.load(url, resolve, undefined, reject);
  });

  // Normalize units: Mixamo FBX is often centimeters. Convert to meters.
  const UNIT_SCALE = 0.01;
  model.scale.setScalar(UNIT_SCALE);

  // Optional: ensure Y-up (Mixamo is already Y-up, but if needed you could rotate here)
  // model.rotation.set(0, 0, 0);

  // Materials: ensure standard materials for PBR-ish look
  model.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) {
      const m = obj as THREE.Mesh;
      m.castShadow = true;
      m.frustumCulled = false;
      // If FBXLoader created MeshPhongMaterial, upgrade to MeshStandardMaterial
      // and enable skinning on the material instance (skinning is not a
      // constructor option for MeshStandardMaterial).
      if ((m.material as THREE.Material).type === "MeshPhongMaterial") {
        const old = m.material as THREE.MeshPhongMaterial;
        const mat = new THREE.MeshStandardMaterial({
          color: old.color,
          map: (old as any).map ?? null,
          roughness: 0.8,
          metalness: 0.0,
        });
        // enable skinning on the created material instance
        (mat as any).skinning = true;
        m.material = mat;
      } else {
        // Ensure skinning flag is set when needed. Some loaders return an
        // array of materials; handle both single material and arrays.
        const setSkinningFlag = (mat: any) => { if (mat && mat.skinning === undefined) mat.skinning = true; };
        if (Array.isArray(m.material)) {
          for (const mm of m.material) setSkinningFlag(mm);
        } else {
          setSkinningFlag(m.material as any);
        }
      }
    }
  });

  return model;
}
