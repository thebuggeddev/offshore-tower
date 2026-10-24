import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import {
  pass, mrt, output, normalView, texture, positionWorld, vec3, float,
  mix, mx_noise_float, smoothstep, fog, color, rangeFogFactor,
} from 'three/tsl';
import posterUrl from './poster.png';

const SKY = 0x9ba29b;
const MODEL_URL = '/models/Offshore_Observation_Tower.glb';
const canvasHost = document.getElementById('scene');
const loading = document.getElementById('loading');
loading.style.backgroundImage = `url("${posterUrl}")`;
const loadingText = document.getElementById('loading-text');
const loadingBar = document.getElementById('loading-bar');
const resetButton = document.getElementById('reset-view');
const wireframeButton = document.getElementById('wireframe-view');
const statusText = document.getElementById('status');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

let renderer;
let scene;
let camera;
let controls;
let postProcessing;
let model;
let wireframeRoot;
let wireframeMaterial;
let wireframeSources = [];
let wireframeEnabled = false;
let animationActive = true;
let needsRender = true;
const initialPosition = new THREE.Vector3(25.5, 18.7, 41.5);
const initialTarget = new THREE.Vector3(-0.35, 11.25, 0);

function showError(error) {
  console.error(error);
  loading.classList.remove('is-complete');
  loading.classList.add('has-error');
  loadingText.textContent = 'The interactive renderer could not start. The rendered model preview remains available.';
  document.getElementById('loading-progress').hidden = true;
  statusText.textContent = 'Static preview';
}

function setProgress(value, label) {
  const percent = Math.max(0, Math.min(100, Math.round(value)));
  loadingBar.style.transform = `scaleX(${percent / 100})`;
  loadingText.textContent = label || `Loading tower · ${percent}%`;
}

function finishLoading() {
  needsRender = true;
  setProgress(100, 'Tower ready');
  loading.classList.add('is-complete');
  wireframeButton.disabled = false;
  statusText.textContent = wireframeEnabled ? 'Wireframe inspection' : '360° inspection';
  document.body.dataset.ready = 'true';
  window.__SCENE_READY__ = true;
  window.__TOWER_READY__ = true;
}

function applyTSLWeathering(source, cache) {
  if (cache.has(source)) return cache.get(source);
  const material = renderer.library.fromMaterial(source);
  material.name = source.name;

  const baseColor = source.map ? texture(source.map).rgb : color(source.color);
  const tint = vec3(source.color.r, source.color.g, source.color.b);
  const broadNoise = mx_noise_float(positionWorld.mul(vec3(0.42, 0.34, 0.42))).mul(0.5).add(0.5);
  const verticalNoise = mx_noise_float(positionWorld.mul(vec3(3.4, 0.16, 3.4))).mul(0.5).add(0.5);
  const panelVariation = mix(float(0.91), float(1.055), broadNoise);
  const rainVariation = mix(float(0.89), float(1.025), verticalNoise);
  const weatheredColor = baseColor.mul(tint).mul(panelVariation).mul(rainVariation);
  const patinaMask = smoothstep(float(0.7), float(0.94), verticalNoise).mul(0.13);
  material.colorNode = mix(weatheredColor, vec3(0.12, 0.054, 0.022), patinaMask);

  const roughness = source.roughnessMap
    ? texture(source.roughnessMap).g.mul(source.roughness)
    : float(source.roughness);
  const metalness = source.metalnessMap
    ? texture(source.metalnessMap).b.mul(source.metalness)
    : float(source.metalness);
  material.roughnessNode = roughness.add(verticalNoise.mul(0.045)).clamp(0.22, 1.0);
  material.metalnessNode = metalness;
  material.userData = { ...source.userData, tslWeathering: 'world-space patina and vertical salt streaks' };
  cache.set(source, material);
  return material;
}

function prepareModel(root) {
  const weatheredMaterials = new Map();
  const weatheredNames = new Set([
    'Orange_Steel', 'Dark_Structural_Steel', 'Weathered_Roof',
    'Oxidized_Joints', 'Saltworn_Foundation',
  ]);
  let meshCount = 0;
  let triangleCount = 0;
  root.traverse((object) => {
    if (!object.isMesh) return;
    meshCount += 1;
    object.castShadow = true;
    object.receiveShadow = true;
    const geometry = object.geometry;
    triangleCount += geometry.index
      ? geometry.index.count / 3
      : geometry.attributes.position.count / 3;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    const converted = materials.map((material) => (
      weatheredNames.has(material.name)
        ? applyTSLWeathering(material, weatheredMaterials)
        : material
    ));
    object.material = Array.isArray(object.material) ? converted : converted[0];
  });
  root.userData.inspection = { meshCount, triangleCount: Math.round(triangleCount) };
  return root.userData.inspection;
}

function createWireframe() {
  model.updateMatrixWorld(true);
  const modelInverse = model.matrixWorld.clone().invert();
  wireframeMaterial = new THREE.LineBasicNodeMaterial({ color: 0x202722 });
  wireframeRoot = new THREE.Group();
  wireframeRoot.name = 'Inspection_Wireframe';
  wireframeRoot.visible = false;
  wireframeSources = [];

  model.traverse((object) => {
    if (!object.isMesh) return;
    const geometry = new THREE.WireframeGeometry(object.geometry);
    const lines = new THREE.LineSegments(geometry, wireframeMaterial);
    lines.name = `${object.name}_Wireframe`;
    lines.matrixAutoUpdate = false;
    lines.matrix.multiplyMatrices(modelInverse, object.matrixWorld);
    lines.matrixWorldNeedsUpdate = true;
    lines.castShadow = false;
    lines.receiveShadow = false;
    wireframeRoot.add(lines);
    wireframeSources.push({ object, visible: object.visible });
  });
  model.add(wireframeRoot);
}

function setWireframe(enabled) {
  wireframeEnabled = Boolean(enabled);
  if (model) {
    if (wireframeEnabled && !wireframeRoot) createWireframe();
    if (wireframeRoot) {
      wireframeRoot.visible = wireframeEnabled;
      wireframeSources.forEach(({ object, visible }) => {
        object.visible = wireframeEnabled ? false : visible;
      });
    }
  }
  wireframeButton.setAttribute('aria-pressed', String(wireframeEnabled));
  wireframeButton.setAttribute('aria-label', wireframeEnabled ? 'Hide wireframe' : 'Show wireframe');
  wireframeButton.title = wireframeEnabled ? 'Show solid materials (F)' : 'Show wireframe (F)';
  document.body.dataset.wireframe = String(wireframeEnabled);
  if (document.body.dataset.ready === 'true') {
    statusText.textContent = wireframeEnabled ? 'Wireframe inspection' : '360° inspection';
  }
  needsRender = true;
  return wireframeEnabled;
}

function resetView() {
  if (reducedMotion) {
    camera.position.copy(initialPosition);
    controls.target.copy(initialTarget);
    controls.update();
    return;
  }
  const startPosition = camera.position.clone();
  const startTarget = controls.target.clone();
  const start = performance.now();
  const duration = 240;
  function step(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(startPosition, initialPosition, eased);
    controls.target.lerpVectors(startTarget, initialTarget, eased);
    controls.update();
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function setInspectionView(name) {
  const views = {
    reference: [initialPosition, initialTarget],
    rear: [new THREE.Vector3(-24, 18, -40), initialTarget],
    left: [new THREE.Vector3(-43, 17, 8), initialTarget],
    right: [new THREE.Vector3(43, 17, -8), initialTarget],
    above: [new THREE.Vector3(12, 44, 15), new THREE.Vector3(0, 10, 0)],
    below: [new THREE.Vector3(19, -8, 28), new THREE.Vector3(0, 9, 0)],
    detail: [new THREE.Vector3(10, 18.2, 15), new THREE.Vector3(0, 16.8, 0)],
  };
  const view = views[name] || views.reference;
  camera.position.copy(view[0]);
  controls.target.copy(view[1]);
  controls.update();
}

function bindKeyboard() {
  const canvas = renderer.domElement;
  canvas.addEventListener('keydown', (event) => {
    const key = event.key;
    if (key === 'Home') {
      event.preventDefault();
      resetView();
      return;
    }
    if (key === 'f' || key === 'F') {
      event.preventDefault();
      setWireframe(!wireframeEnabled);
      return;
    }
    const offset = camera.position.clone().sub(controls.target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    let handled = true;
    if (key === 'ArrowLeft') spherical.theta -= 0.075;
    else if (key === 'ArrowRight') spherical.theta += 0.075;
    else if (key === 'ArrowUp') spherical.phi -= 0.06;
    else if (key === 'ArrowDown') spherical.phi += 0.06;
    else if (key === '+' || key === '=') spherical.radius *= 0.92;
    else if (key === '-' || key === '_') spherical.radius *= 1.08;
    else if (['w', 'a', 's', 'd', 'W', 'A', 'S', 'D'].includes(key)) {
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
      const direction = new THREE.Vector3();
      if (key.toLowerCase() === 'a') direction.addScaledVector(right, -1);
      if (key.toLowerCase() === 'd') direction.add(right);
      if (key.toLowerCase() === 'w') direction.add(up);
      if (key.toLowerCase() === 's') direction.addScaledVector(up, -1);
      direction.multiplyScalar(spherical.radius * 0.025);
      camera.position.add(direction);
      controls.target.add(direction);
    } else handled = false;
    if (!handled) return;
    event.preventDefault();
    spherical.phi = THREE.MathUtils.clamp(spherical.phi, controls.minPolarAngle, controls.maxPolarAngle);
    spherical.radius = THREE.MathUtils.clamp(spherical.radius, controls.minDistance, controls.maxDistance);
    if (!['w', 'a', 's', 'd', 'W', 'A', 'S', 'D'].includes(key)) {
      camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
    }
    controls.update();
  });
}

async function start() {
  renderer = new THREE.WebGPURenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 1.08;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  await renderer.init();
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute(
    'aria-label',
    'Interactive 3D offshore tower. Drag to orbit, scroll to zoom, right-drag to pan. Keyboard: arrows orbit, W A S D pan, plus and minus zoom, F toggles wireframe, Home resets.',
  );
  canvasHost.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(SKY);
  scene.fogNode = fog(color(SKY), rangeFogFactor(float(58), float(138)));

  camera = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.1, 320);
  camera.position.copy(initialPosition);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(initialTarget);
  // Settle immediately after input so demand rendering never depends on
  // dozens of residual-damping frames on software or low-power renderers.
  controls.enableDamping = false;
  controls.enablePan = true;
  controls.screenSpacePanning = true;
  controls.minDistance = 7;
  controls.maxDistance = 96;
  controls.minPolarAngle = 0.025;
  controls.maxPolarAngle = Math.PI - 0.025;
  controls.rotateSpeed = 0.62;
  controls.zoomSpeed = 0.78;
  controls.panSpeed = 0.68;
  controls.update();
  controls.saveState();
  controls.addEventListener('change', () => { needsRender = true; });

  const hemisphere = new THREE.HemisphereLight(0xdde4df, 0x44463f, 1.9);
  scene.add(hemisphere);
  const key = new THREE.DirectionalLight(0xfff2df, 1.4);
  key.position.set(-16, 30, 22);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.radius = 5;
  key.shadow.camera.left = -18;
  key.shadow.camera.right = 18;
  key.shadow.camera.top = 28;
  key.shadow.camera.bottom = -12;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 85;
  key.shadow.bias = -0.00025;
  key.shadow.normalBias = 0.025;
  key.target.position.set(0, 10, 0);
  scene.add(key, key.target);
  const fill = new THREE.DirectionalLight(0xdce8ef, 0.62);
  fill.position.set(18, 18, -22);
  scene.add(fill);

  const pmrem = new THREE.PMREMGenerator(renderer);
  const environmentScene = new RoomEnvironment();
  const environment = pmrem.fromScene(environmentScene, 0.04);
  scene.environment = environment.texture;
  scene.environmentIntensity = 0.42;
  environmentScene.dispose();
  pmrem.dispose();

  const groundMaterial = new THREE.MeshStandardNodeMaterial({ roughness: 0.97, metalness: 0 });
  const groundNoise = mx_noise_float(positionWorld.mul(vec3(0.075, 0.075, 0.075))).mul(0.5).add(0.5);
  groundMaterial.colorNode = mix(color(0x8f978f), color(0xa1a79f), groundNoise);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), groundMaterial);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.09;
  ground.receiveShadow = true;
  scene.add(ground);

  const scenePass = pass(scene, camera);
  scenePass.setMRT(mrt({ output, normal: normalView }));
  const sceneColor = scenePass.getTextureNode('output');
  const aoPass = ao(
    scenePass.getTextureNode('depth'),
    scenePass.getTextureNode('normal'),
    camera,
  );
  aoPass.resolutionScale = 0.35;
  aoPass.radius.value = 0.42;
  aoPass.thickness.value = 1.15;
  aoPass.distanceFallOff.value = 0.85;
  aoPass.scale.value = 1.05;
  aoPass.samples.value = 4;
  const occlusion = mix(vec3(1), aoPass.getTextureNode().rgb, float(0.48));
  postProcessing = new THREE.PostProcessing(renderer);
  postProcessing.outputNode = sceneColor.mul(occlusion);

  resetButton.addEventListener('click', resetView);
  wireframeButton.addEventListener('click', () => setWireframe(!wireframeEnabled));
  bindKeyboard();
  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
    renderer.setSize(innerWidth, innerHeight);
    needsRender = true;
  });
  document.addEventListener('visibilitychange', () => {
    animationActive = !document.hidden;
    if (animationActive) needsRender = true;
  });

  renderer.setAnimationLoop(() => {
    if (!animationActive) return;
    controls.update();
    if (needsRender) {
      // Wireframe inspection does not need GTAO; bypassing its normal/depth
      // passes also keeps line rendering lightweight on the WebGL2 fallback.
      if (wireframeEnabled) renderer.render(scene, camera);
      else postProcessing.render();
      window.__RENDERED_FRAMES__ = (window.__RENDERED_FRAMES__ || 0) + 1;
      needsRender = false;
    }
  });

  const loader = new GLTFLoader();
  loader.load(
    MODEL_URL,
    (gltf) => {
      model = gltf.scene;
      const stats = prepareModel(model);
      scene.add(model);
      const bounds = new THREE.Box3().setFromObject(model);
      const size = bounds.getSize(new THREE.Vector3());
      window.__VIEWER_STATS__ = {
        ...stats,
        bounds: size.toArray(),
        materialsWithTSL: weatheredMaterialsCount(model),
        backend: renderer.backend.isWebGPUBackend ? 'WebGPU' : 'WebGL2',
      };
      window.__TOWER_VIEWER__ = {
        scene, camera, controls, renderer, model, setInspectionView, resetView, setWireframe,
        get wireframeEnabled() { return wireframeEnabled; },
        requestRender: () => { needsRender = true; },
      };
      finishLoading();
    },
    (event) => {
      if (event.lengthComputable && event.total > 0) {
        setProgress((event.loaded / event.total) * 94);
      } else {
        loadingText.textContent = 'Loading detailed tower geometry…';
      }
    },
    showError,
  );
}

function weatheredMaterialsCount(root) {
  const names = new Set();
  root.traverse((object) => {
    if (!object.isMesh) return;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => {
      if (material.userData.tslWeathering) names.add(material.name);
    });
  });
  return names.size;
}

start().catch(showError);
