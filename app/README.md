# Offshore Observation Tower

A fully procedural offshore observation tower, built with Blender Python and delivered as an optimized glTF 2.0 asset with an interactive Three.js viewer. No downloaded models, image planes, external model assets, or external texture assets are used.

## Project contents

- `public/models/Offshore_Observation_Tower.glb` — optimized, Y-up model in meters; 149,104 exported triangles in 51 assembly/material mesh batches, with baked base-color, ORM, and tangent-normal textures embedded.
- `tools/Offshore_Tower_Generator.py` — complete Blender 4.5 LTS procedural generator, including geometry, material baking, GLB export, and six verification renders.
- `src/scene.js` — Three.js r180 viewer with TSL weathering, PBR materials, soft lighting, GTAO, WebGPU with WebGL2 fallback, and demand rendering.
- `src/poster.png` — bundled Blender render used during loading and if interactive rendering cannot start.

The model includes the tapered corrugated shaft, detailed control room and recessed openings, balcony and roof, communications equipment, connected industrial boom, lower utility deck and stairs, and the complete braced offshore foundation.

## Run the viewer

Requires a current Node.js LTS release and a modern browser with WebGL2 or WebGPU.

```bash
npm install
npm run dev
```

Open the local address printed by Vite. All model and image assets are served locally; no API keys or backend services are required.

## Production build

```bash
npm run build
npm run preview
```

The production site is written to `dist/`.

## Controls

- **Mouse:** drag to orbit, scroll to zoom, right-drag to pan.
- **Touch:** one-finger drag to orbit, pinch to zoom, two-finger drag to pan.
- **Keyboard:** focus the canvas, then use arrow keys to orbit, W/A/S/D to pan, plus/minus to zoom, and Home to reset.
- **Wireframe:** use the Wireframe button, or focus the canvas and press F, to switch between solid materials and mesh wireframe. The display mode remains active while navigating or resetting the camera.
- **Reset view:** returns to the reference-inspired camera.

## Regenerate the asset

Run from this project directory with Blender 4.5 LTS available on `PATH`:

```bash
blender -b --python tools/Offshore_Tower_Generator.py -- \
  --output-dir ./tower_output --render
```

Optional quality overrides:

```bash
blender -b --python tools/Offshore_Tower_Generator.py -- \
  --output-dir ./tower_output --render --texture-size 1024 --samples 64
```

The generator writes `tower_output/Offshore_Observation_Tower.glb`, `Tower_Asset_Manifest.json`, baked texture maps, the reference render, and rear/left/right/above/below verification renders. To use a regenerated model in this viewer, replace `public/models/Offshore_Observation_Tower.glb` with the new GLB and rebuild.

## Verification

The production viewer was exercised in Chromium using the WebGL2 software fallback. Solid/wireframe switching, orbit, zoom, pan, button and keyboard reset, keyboard navigation, mobile layout, inspection views, and idle demand rendering passed with no page errors, console errors, failed requests, HTTP error responses, or rendering compatibility warnings.
