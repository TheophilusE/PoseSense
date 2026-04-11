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

function attachJointSphere(parent: THREE.Bone, radius: number, material: THREE.Material): void {
  const geom = new THREE.SphereGeometry(radius, 12, 10);
  const mesh = new THREE.Mesh(geom, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = `${parent.name}_joint`;
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
  const jointMat = new THREE.MeshStandardMaterial({ color: 0xf2994a, roughness: 0.38, metalness: 0.06 });
  const headMat = new THREE.MeshStandardMaterial({ color: 0xe8d6c1, roughness: 0.55, metalness: 0.0 });

  attachSegment(hips, spine, 0.07, torsoMat);
  attachSegment(spine, spine1, 0.068, torsoMat);
  attachSegment(spine1, spine2, 0.066, torsoMat);
  attachSegment(spine2, neck, 0.055, torsoMat);
  attachSegment(neck, head, 0.045, torsoMat);

  attachSegment(spine2, leftShoulder, 0.04, limbMat);
  attachSegment(leftShoulder, leftArm, 0.043, limbMat);
  attachSegment(leftArm, leftForeArm, 0.039, limbMat);
  attachSegment(leftForeArm, leftHand, 0.033, limbMat);

  attachSegment(spine2, rightShoulder, 0.04, limbMat);
  attachSegment(rightShoulder, rightArm, 0.043, limbMat);
  attachSegment(rightArm, rightForeArm, 0.039, limbMat);
  attachSegment(rightForeArm, rightHand, 0.033, limbMat);

  attachSegment(hips, leftUpLeg, 0.052, limbMat);
  attachSegment(leftUpLeg, leftLeg, 0.05, limbMat);
  attachSegment(leftLeg, leftFoot, 0.045, limbMat);
  attachSegment(leftFoot, leftToeBase, 0.03, limbMat);

  attachSegment(hips, rightUpLeg, 0.052, limbMat);
  attachSegment(rightUpLeg, rightLeg, 0.05, limbMat);
  attachSegment(rightLeg, rightFoot, 0.045, limbMat);
  attachSegment(rightFoot, rightToeBase, 0.03, limbMat);

  attachJointSphere(hips, 0.075, jointMat);
  attachJointSphere(spine2, 0.06, jointMat);
  attachJointSphere(leftHand, 0.038, jointMat);
  attachJointSphere(rightHand, 0.038, jointMat);
  attachJointSphere(leftFoot, 0.035, jointMat);
  attachJointSphere(rightFoot, 0.035, jointMat);

  const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12), headMat);
  headMesh.position.set(0, 0.11, 0);
  headMesh.castShadow = true;
  headMesh.receiveShadow = true;
  head.add(headMesh);

  return { root, skinnedMesh };
}
