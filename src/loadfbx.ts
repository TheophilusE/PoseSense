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
      // If FBXLoader created MeshPhongMaterial, it’s fine; you can also upgrade:
      if ((m.material as THREE.Material).type === "MeshPhongMaterial") {
        const old = m.material as THREE.MeshPhongMaterial;
        const mat = new THREE.MeshStandardMaterial({
          color: old.color,
          map: (old as any).map ?? null,
          skinning: true,
          roughness: 0.8,
          metalness: 0.0,
        });
        m.material = mat;
      } else if ((m.material as any).skinning === undefined) {
        // Ensure skinning flag is set when needed
        (m.material as any).skinning = true;
      }
    }
  });

  return model;
}
