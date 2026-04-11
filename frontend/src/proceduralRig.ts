import * as THREE from 'three';

export interface ProceduralHumanoidRig {
  root: THREE.Group;
  skinnedMesh: THREE.SkinnedMesh;
}

function makeBone(name: string, pos: [number, number, number], parent?: THREE.Bone): THREE.Bone {
  const bone = new THREE.Bone();
  bone.name = name;
  bone.position.set(pos[0], pos[1], pos[2]);
  if (parent) parent.add(bone);
  return bone;
}

function attachSegment(parent: THREE.Bone, child: THREE.Bone, radius: number, material: THREE.Material): void {
  const dir = child.position.clone();
  const length = dir.length();
  if (length < 1e-4) return;

  const shaftLength = Math.max(length - (radius * 2.0), 0.001);
  const geom = new THREE.CapsuleGeometry(radius, shaftLength, 4, 8);
  const mesh = new THREE.Mesh(geom, material);
  const up = new THREE.Vector3(0, 1, 0);
  mesh.position.copy(dir.multiplyScalar(0.5));
  mesh.quaternion.setFromUnitVectors(up, child.position.clone().normalize());
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = `${parent.name}_segment`;
  parent.add(mesh);
}

function attachCapsuleLocal(parent: THREE.Object3D, localVector: THREE.Vector3, radius: number, material: THREE.Material, name: string): void {
  const length = localVector.length();
  if (length < 1e-4) return;

  const shaftLength = Math.max(length - (radius * 2.0), 0.001);
  const geom = new THREE.CapsuleGeometry(radius, shaftLength, 4, 8);
  const mesh = new THREE.Mesh(geom, material);
  const up = new THREE.Vector3(0, 1, 0);
  mesh.position.copy(localVector).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(up, localVector.clone().normalize());
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = name;
  parent.add(mesh);
}

export function createProceduralHumanoid(): ProceduralHumanoidRig {
  const root = new THREE.Group();
  root.name = 'ProceduralHumanoidRoot';

  const hips = makeBone('Hips', [0, 1.0, 0]);

  const spine = makeBone('Spine', [0, 0.18, 0], hips);
  const spine1 = makeBone('Spine1', [0, 0.16, 0], spine);
  const spine2 = makeBone('Spine2', [0, 0.16, 0], spine1);
  const neck = makeBone('Neck', [0, 0.12, 0], spine2);
  const head = makeBone('Head', [0, 0.16, 0], neck);

  const leftShoulder = makeBone('LeftShoulder', [-0.11, 0.07, 0], spine2);
  const leftArm = makeBone('LeftArm', [-0.19, 0, 0], leftShoulder);
  const leftForeArm = makeBone('LeftForeArm', [-0.23, 0, 0], leftArm);
  const leftHand = makeBone('LeftHand', [-0.18, 0, 0], leftForeArm);

  const rightShoulder = makeBone('RightShoulder', [0.11, 0.07, 0], spine2);
  const rightArm = makeBone('RightArm', [0.19, 0, 0], rightShoulder);
  const rightForeArm = makeBone('RightForeArm', [0.23, 0, 0], rightArm);
  const rightHand = makeBone('RightHand', [0.18, 0, 0], rightForeArm);

  const leftUpLeg = makeBone('LeftUpLeg', [-0.09, -0.1, 0], hips);
  const leftLeg = makeBone('LeftLeg', [0, -0.42, 0], leftUpLeg);
  const leftFoot = makeBone('LeftFoot', [0, -0.43, 0.03], leftLeg);
  const leftToeBase = makeBone('LeftToeBase', [0, -0.06, 0.16], leftFoot);

  const rightUpLeg = makeBone('RightUpLeg', [0.09, -0.1, 0], hips);
  const rightLeg = makeBone('RightLeg', [0, -0.42, 0], rightUpLeg);
  const rightFoot = makeBone('RightFoot', [0, -0.43, 0.03], rightLeg);
  const rightToeBase = makeBone('RightToeBase', [0, -0.06, 0.16], rightFoot);

  const bones: THREE.Bone[] = [
    hips,
    spine,
    spine1,
    spine2,
    neck,
    head,
    leftShoulder,
    leftArm,
    leftForeArm,
    leftHand,
    rightShoulder,
    rightArm,
    rightForeArm,
    rightHand,
    leftUpLeg,
    leftLeg,
    leftFoot,
    leftToeBase,
    rightUpLeg,
    rightLeg,
    rightFoot,
    rightToeBase,
  ];

  const coreGeom = new THREE.BoxGeometry(0.1, 0.1, 0.1);
  const vertexCount = coreGeom.attributes.position.count;
  const skinIndices = new Uint16Array(vertexCount * 4);
  const skinWeights = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; ++i) {
    skinIndices[(i * 4) + 0] = 0;
    skinWeights[(i * 4) + 0] = 1.0;
  }
  coreGeom.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndices, 4));
  coreGeom.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeights, 4));

  const coreMaterial = new THREE.MeshStandardMaterial({
    color: 0x4c5c72,
    transparent: true,
    opacity: 0.0,
    roughness: 0.8,
    metalness: 0.0,
    depthWrite: false,
  });
  (coreMaterial as any).skinning = true;

  const skinnedMesh = new THREE.SkinnedMesh(coreGeom, coreMaterial);
  skinnedMesh.name = 'ProceduralRigCore';
  skinnedMesh.castShadow = false;
  skinnedMesh.receiveShadow = false;
  skinnedMesh.frustumCulled = false;

  skinnedMesh.add(hips);
  const skeleton = new THREE.Skeleton(bones);
  skinnedMesh.bind(skeleton);
  root.add(skinnedMesh);

  const torsoMat = new THREE.MeshStandardMaterial({ color: 0x5b8def, roughness: 0.45, metalness: 0.05 });
  const limbMat = new THREE.MeshStandardMaterial({ color: 0x58b27d, roughness: 0.5, metalness: 0.04 });
  const headMat = new THREE.MeshStandardMaterial({ color: 0xe8d6c1, roughness: 0.55, metalness: 0.0 });
  const footMat = new THREE.MeshStandardMaterial({ color: 0x4f5d75, roughness: 0.58, metalness: 0.02 });

  // Keep the procedural mesh intentionally simple: one torso, one head,
  // and two segments per major limb.
  attachCapsuleLocal(hips, new THREE.Vector3(0, 0.5, 0), 0.085, torsoMat, 'Torso');

  attachSegment(leftArm, leftForeArm, 0.043, limbMat);
  attachSegment(leftForeArm, leftHand, 0.037, limbMat);

  attachSegment(rightArm, rightForeArm, 0.043, limbMat);
  attachSegment(rightForeArm, rightHand, 0.037, limbMat);

  attachSegment(leftUpLeg, leftLeg, 0.05, limbMat);
  attachSegment(leftLeg, leftFoot, 0.046, limbMat);
  attachSegment(leftFoot, leftToeBase, 0.03, footMat);

  attachSegment(rightUpLeg, rightLeg, 0.05, limbMat);
  attachSegment(rightLeg, rightFoot, 0.046, limbMat);
  attachSegment(rightFoot, rightToeBase, 0.03, footMat);

  const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12), headMat);
  headMesh.position.set(0, 0.11, 0);
  headMesh.castShadow = true;
  headMesh.receiveShadow = true;
  head.add(headMesh);

  return { root, skinnedMesh };
}
