import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { LANE_CONFIG } from '../simulationTypes';

/**
 * Creates a cached high-DPI canvas texture with 3D spherical gradient
 * and crisp centered basket item count.
 */
const textureCache = new Map();

function getCustomerBadgeTexture(basket) {
  if (textureCache.has(basket)) {
    return textureCache.get(basket);
  }

  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');

  // Determine color scheme based on basket size
  let colorLight = '#38bdf8'; // Cyan (1-4 items)
  let colorMid = '#0284c7';
  let colorDark = '#0369a1';

  if (basket >= 12) {
    colorLight = '#f87171'; // Coral Red (12+ items)
    colorMid = '#ef4444';
    colorDark = '#991b1b';
  } else if (basket >= 8) {
    colorLight = '#fbbf24'; // Amber Orange (8-11 items)
    colorMid = '#f59e0b';
    colorDark = '#92400e';
  } else if (basket >= 5) {
    colorLight = '#4ade80'; // Emerald Green (5-7 items)
    colorMid = '#22c55e';
    colorDark = '#166534';
  }

  // Draw 3D Spherical Radial Gradient
  const grad = ctx.createRadialGradient(90, 90, 16, 128, 128, 116);
  grad.addColorStop(0, colorLight);
  grad.addColorStop(0.55, colorMid);
  grad.addColorStop(1, colorDark);

  ctx.beginPath();
  ctx.arc(128, 128, 116, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  // Outer glossy rim
  ctx.lineWidth = 8;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
  ctx.stroke();

  // Top specular glint highlight
  const glint = ctx.createRadialGradient(85, 70, 4, 85, 70, 48);
  glint.addColorStop(0, 'rgba(255, 255, 255, 0.9)');
  glint.addColorStop(1, 'rgba(255, 255, 255, 0)');
  ctx.beginPath();
  ctx.arc(85, 70, 48, 0, Math.PI * 2);
  ctx.fillStyle = glint;
  ctx.fill();

  // Bold Centered Number
  ctx.fillStyle = '#ffffff';
  ctx.font = '900 112px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.85)';
  ctx.shadowBlur = 12;
  ctx.shadowOffsetX = 2;
  ctx.shadowOffsetY = 4;
  ctx.fillText(String(basket), 128, 132);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  textureCache.set(basket, texture);
  return texture;
}

/**
 * Procedural floor grid texture with polished supermarket tile appearance.
 */
function createFloorTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Dark polished concrete base
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, 512, 512);

  // Tile joints
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 2.5;

  const tileSize = 64;
  for (let x = 0; x <= 512; x += tileSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 512);
    ctx.stroke();
  }
  for (let y = 0; y <= 512; y += tileSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(512, y);
    ctx.stroke();
  }

  // Subtle floor specular texture
  for (let i = 0; i < 300; i++) {
    const rx = Math.random() * 512;
    const ry = Math.random() * 512;
    ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.035})`;
    ctx.fillRect(rx, ry, Math.random() * 10 + 4, Math.random() * 10 + 4);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(14, 9);
  return texture;
}

export default function CheckoutFloor({
  currentQueues = [3, 2, 4, 3, 4, 3, 3, 4, 4, 2],
  openStatus = [true, true, true, true, true, true, true, true, true, true],
  capacities = [6, 6, 8, 8, 8, 8, 8, 8, 8, 6],
  laneTypes = ['express', 'express', 'standard', 'standard', 'standard', 'standard', 'standard', 'standard', 'large_basket', 'flexible'],
  laneCustomers = [],
  recommendedAction = 2,
  selectedLaneIdx = null,
  onSelectLane
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const targetLightRef = useRef(null);
  const targetBoxRef = useRef(null);
  const laneObjectsRef = useRef([]);

  // Station spacing: 10 stations centered across X axis
  const laneSpacing = 2.45;
  const startX = -((10 - 1) * laneSpacing) / 2; // -11.025

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 450;

    // 1. Scene & Camera Setup (Elevated slight isometric angle)
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);
    scene.fog = new THREE.FogExp2(0x090d16, 0.022);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 100);
    // Camera placed slightly above and in front of counters looking inward (closer framing)
    camera.position.set(0, 8.8, 12.0);
    camera.lookAt(0, 0.8, -1.2);
    cameraRef.current = camera;

    // 2. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.3;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 3. Realistic Store Illumination
    const ambientLight = new THREE.AmbientLight(0x94a3b8, 1.25);
    scene.add(ambientLight);

    const storeCeilingLight = new THREE.DirectionalLight(0xffffff, 2.2);
    storeCeilingLight.position.set(4, 18, 10);
    storeCeilingLight.castShadow = true;
    storeCeilingLight.shadow.mapSize.width = 1024;
    storeCeilingLight.shadow.mapSize.height = 1024;
    scene.add(storeCeilingLight);

    const storeFillLight = new THREE.DirectionalLight(0x38bdf8, 0.65);
    storeFillLight.position.set(-16, 12, -4);
    scene.add(storeFillLight);

    // Cyan Target Spotlight
    const targetLight = new THREE.PointLight(0x00e5ff, 4.5, 9.0, 1.4);
    targetLight.position.set(startX + 2 * laneSpacing, 3.5, -0.2);
    scene.add(targetLight);
    targetLightRef.current = targetLight;

    // 4. Polished Supermarket Floor Plane
    const floorGeo = new THREE.PlaneGeometry(42, 28);
    const floorMat = new THREE.MeshStandardMaterial({
      map: createFloorTexture(),
      roughness: 0.22,
      metalness: 0.22
    });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.set(0, 0, -2);
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    // Background Wall / Store Aisles
    const backWall = new THREE.Mesh(
      new THREE.PlaneGeometry(42, 9),
      new THREE.MeshStandardMaterial({ color: 0x0c1322, roughness: 0.85 })
    );
    backWall.position.set(0, 4.5, -14);
    scene.add(backWall);

    // 5. Shared Materials
    const metalMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      roughness: 0.32,
      metalness: 0.78
    });
    const chromeMat = new THREE.MeshStandardMaterial({
      color: 0xf1f5f9,
      roughness: 0.08,
      metalness: 0.96
    });
    const conveyorMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.9
    });
    const cashierMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.65
    });

    // 6. Build 10 Physical Checkout Stations
    const laneObjects = [];

    for (let i = 0; i < 10; i++) {
      const laneGroup = new THREE.Group();
      const laneX = startX + i * laneSpacing;
      laneGroup.position.set(laneX, 0, 0);

      // Soft Floor Contact Shadow underneath counter
      const shadowPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(1.8, 3.4),
        new THREE.MeshBasicMaterial({
          color: 0x020617,
          transparent: true,
          opacity: 0.7
        })
      );
      shadowPlane.rotation.x = -Math.PI / 2;
      shadowPlane.position.set(0, 0.01, 0.3);
      laneGroup.add(shadowPlane);

      // A. Counter Body
      const counterBase = new THREE.Mesh(
        new THREE.BoxGeometry(1.5, 0.75, 2.7),
        metalMat
      );
      counterBase.position.set(0, 0.375, 0.2);
      counterBase.castShadow = true;
      counterBase.receiveShadow = true;
      laneGroup.add(counterBase);

      // Rounded Front Counter Bumper
      const bumper = new THREE.Mesh(
        new THREE.CylinderGeometry(0.7, 0.7, 0.75, 16, 1, false, -Math.PI / 2, Math.PI),
        metalMat
      );
      bumper.rotation.y = Math.PI / 2;
      bumper.position.set(0, 0.375, 1.55);
      bumper.castShadow = true;
      laneGroup.add(bumper);

      // Conveyor Belt
      const conveyor = new THREE.Mesh(
        new THREE.PlaneGeometry(1.05, 1.8),
        conveyorMat
      );
      conveyor.rotation.x = -Math.PI / 2;
      conveyor.position.set(0, 0.76, 0.3);
      conveyor.receiveShadow = true;
      laneGroup.add(conveyor);

      // Scanner Glass Window
      const scanner = new THREE.Mesh(
        new THREE.PlaneGeometry(0.35, 0.25),
        new THREE.MeshBasicMaterial({ color: 0xef4444 })
      );
      scanner.rotation.x = -Math.PI / 2;
      scanner.position.set(0, 0.765, 0.8);
      laneGroup.add(scanner);

      // POS Touchscreen Monitor on Mount
      const posStand = new THREE.Mesh(
        new THREE.CylinderGeometry(0.04, 0.04, 0.35, 8),
        chromeMat
      );
      posStand.position.set(0.45, 0.92, 0.6);
      laneGroup.add(posStand);

      const posMonitor = new THREE.Mesh(
        new THREE.BoxGeometry(0.35, 0.25, 0.04),
        new THREE.MeshStandardMaterial({
          color: 0x0284c7,
          emissive: 0x0284c7,
          emissiveIntensity: 0.8
        })
      );
      posMonitor.rotation.x = -0.35;
      posMonitor.position.set(0.45, 1.15, 0.6);
      laneGroup.add(posMonitor);

      // B. Cashier Figure Silhouette seated at register
      const cashierChair = new THREE.Mesh(
        new THREE.CylinderGeometry(0.25, 0.25, 0.45, 12),
        cashierMat
      );
      cashierChair.position.set(0, 0.225, 1.1);
      laneGroup.add(cashierChair);

      const cashierTorso = new THREE.Mesh(
        new THREE.BoxGeometry(0.45, 0.55, 0.3),
        cashierMat
      );
      cashierTorso.position.set(0, 0.75, 1.1);
      laneGroup.add(cashierTorso);

      const cashierHead = new THREE.Mesh(
        new THREE.SphereGeometry(0.18, 16, 16),
        cashierMat
      );
      cashierHead.position.set(0, 1.15, 1.1);
      laneGroup.add(cashierHead);

      // C. Chrome Guide Rails / Stanchions
      const rail1 = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.85, 8), chromeMat);
      rail1.position.set(-0.85, 0.425, 1.8);
      laneGroup.add(rail1);

      const rail2 = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.85, 8), chromeMat);
      rail2.position.set(-0.85, 0.425, -1.2);
      laneGroup.add(rail2);

      const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 3.0, 8), chromeMat);
      bar.rotation.x = Math.PI / 2;
      bar.position.set(-0.85, 0.75, 0.3);
      laneGroup.add(bar);

      // D. Queue Group
      const queueGroup = new THREE.Group();
      laneGroup.add(queueGroup);

      // Empty Slot Rings
      const emptySlotsGroup = new THREE.Group();
      for (let s = 0; s < 8; s++) {
        const ringGeo = new THREE.RingGeometry(0.28, 0.36, 24);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0x475569,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.35
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(0, 0.02, -1.35 - s * 1.05);
        emptySlotsGroup.add(ring);
      }
      laneGroup.add(emptySlotsGroup);

      scene.add(laneGroup);
      laneObjects.push({
        laneId: i,
        laneGroup,
        queueGroup,
        emptySlotsGroup,
        posMonitor
      });
    }

    laneObjectsRef.current = laneObjects;

    // 7. RL Target Glowing 3D Volume Box
    const targetBoxGeo = new THREE.BoxGeometry(2.15, 2.6, 11.2);
    const targetBoxMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.16,
      side: THREE.DoubleSide
    });
    const targetBox = new THREE.Mesh(targetBoxGeo, targetBoxMat);
    targetBox.position.set(startX + 2 * laneSpacing, 1.3, -3.2);

    const edgesGeo = new THREE.EdgesGeometry(targetBoxGeo);
    const edgesMat = new THREE.LineBasicMaterial({
      color: 0x22d3ee,
      linewidth: 2,
      transparent: true,
      opacity: 0.8
    });
    const edgesLine = new THREE.LineSegments(edgesGeo, edgesMat);
    targetBox.add(edgesLine);

    scene.add(targetBox);
    targetBoxRef.current = targetBox;

    // 8. Miniature Shopping Carts & Ambient Walking Shoppers
    const cartMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      roughness: 0.2,
      metalness: 0.8
    });

    const cartPositions = [
      { x: -9.5, z: 4.8, rot: 0.2 },
      { x: -3.5, z: 5.4, rot: -0.4 },
      { x: 2.8, z: 4.9, rot: 0.1 },
      { x: 8.2, z: 5.6, rot: 0.35 }
    ];

    cartPositions.forEach((pos) => {
      const cartGroup = new THREE.Group();
      cartGroup.position.set(pos.x, 0, pos.z);
      cartGroup.rotation.y = pos.rot;

      const basket = new THREE.Mesh(
        new THREE.BoxGeometry(0.7, 0.45, 0.9),
        new THREE.MeshStandardMaterial({ color: 0x64748b, wireframe: true })
      );
      basket.position.set(0, 0.45, 0);
      cartGroup.add(basket);

      const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.65, 8), cartMat);
      handle.rotation.z = Math.PI / 2;
      handle.position.set(0, 0.72, 0.45);
      cartGroup.add(handle);

      for (let wx = -0.3; wx <= 0.3; wx += 0.6) {
        for (let wz = -0.35; wz <= 0.35; wz += 0.7) {
          const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.03, 8), cartMat);
          wheel.rotation.z = Math.PI / 2;
          wheel.position.set(wx, 0.06, wz);
          cartGroup.add(wheel);
        }
      }

      // Shopper pushing cart
      const shopperTorso = new THREE.Mesh(
        new THREE.BoxGeometry(0.35, 0.8, 0.25),
        cashierMat
      );
      shopperTorso.position.set(0, 0.85, 0.7);
      cartGroup.add(shopperTorso);

      const shopperHead = new THREE.Mesh(
        new THREE.SphereGeometry(0.15, 12, 12),
        cashierMat
      );
      shopperHead.position.set(0, 1.4, 0.7);
      cartGroup.add(shopperHead);

      scene.add(cartGroup);
    });

    // 9. Resize Handling
    const handleResize = () => {
      if (!containerRef.current || !renderer || !camera) return;
      const w = containerRef.current.clientWidth || 800;
      const h = containerRef.current.clientHeight || 450;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    // 10. Animation Loop
    let animId;
    let clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Breathing pulse for RL Target Box
      if (targetBoxRef.current) {
        const pulse = Math.sin(elapsedTime * 3) * 0.05 + 0.16;
        targetBoxRef.current.material.opacity = pulse;
      }

      // Align 2.5D token planes toward camera
      if (laneObjectsRef.current) {
        laneObjectsRef.current.forEach((lane) => {
          if (lane.queueGroup) {
            lane.queueGroup.children.forEach((groupOrMesh) => {
              if (groupOrMesh.isGroup) {
                groupOrMesh.children.forEach((child) => {
                  if (child.isMesh && child.geometry.type === 'PlaneGeometry') {
                    child.quaternion.copy(camera.quaternion);
                  }
                });
              } else if (groupOrMesh.isMesh && groupOrMesh.geometry.type === 'PlaneGeometry') {
                groupOrMesh.quaternion.copy(camera.quaternion);
              }
            });
          }
        });
      }

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Sync dynamic simulation data (customers, queues, target lane)
  useEffect(() => {
    if (!laneObjectsRef.current.length) return;

    // 1. Update RL Target Position & Spotlight
    const targetX = startX + recommendedAction * laneSpacing;
    if (targetBoxRef.current) {
      targetBoxRef.current.position.x = targetX;
      targetBoxRef.current.visible = true;
    }
    if (targetLightRef.current) {
      targetLightRef.current.position.x = targetX;
    }

    // 2. Update Customer Badges in Each Lane
    laneObjectsRef.current.forEach((laneObj) => {
      const laneId = laneObj.laneId;
      const queueGroup = laneObj.queueGroup;
      const emptySlotsGroup = laneObj.emptySlotsGroup;
      const maxCap = capacities[laneId] || 8;
      const qLen = currentQueues[laneId] !== undefined ? currentQueues[laneId] : 0;
      const isOpen = openStatus[laneId] !== undefined ? openStatus[laneId] : true;
      const laneData = laneCustomers[laneId] || {};
      const customers = laneData.queue || [];
      const serving = laneData.serving || null;

      // Update POS screen color
      if (laneObj.posMonitor) {
        laneObj.posMonitor.material.color.setHex(isOpen ? 0x0284c7 : 0xe11d48);
        laneObj.posMonitor.material.emissive.setHex(isOpen ? 0x0284c7 : 0xe11d48);
      }

      // Clear previous badges
      while (queueGroup.children.length > 0) {
        const child = queueGroup.children[0];
        queueGroup.remove(child);
      }

      // Empty slot visibility: show dashed rings for slots >= qLen
      if (emptySlotsGroup) {
        emptySlotsGroup.children.forEach((ring, idx) => {
          ring.visible = idx < maxCap && idx >= qLen;
        });
      }

      const renderCustomerFigure = (basket, zPos, isServing = false) => {
        const custGroup = new THREE.Group();
        if (isServing) {
          custGroup.position.set(-0.52, 0, zPos);
        } else {
          custGroup.position.set(0, 0, zPos);
        }

        // Contact ground shadow
        const shadow = new THREE.Mesh(
          new THREE.CircleGeometry(0.24, 16),
          new THREE.MeshBasicMaterial({ color: 0x020617, transparent: true, opacity: 0.65 })
        );
        shadow.rotation.x = -Math.PI / 2;
        shadow.position.y = 0.015;
        custGroup.add(shadow);

        // Silhouette Torso
        const torso = new THREE.Mesh(
          new THREE.CylinderGeometry(0.15, 0.2, 0.52, 10),
          new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.8 })
        );
        torso.position.y = 0.35;
        custGroup.add(torso);

        // Silhouette Head
        const head = new THREE.Mesh(
          new THREE.SphereGeometry(0.14, 10, 10),
          new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.7 })
        );
        head.position.y = 0.72;
        custGroup.add(head);

        // Spherical Glossy Basket Badge floating at head level
        const tex = getCustomerBadgeTexture(basket);
        const badgeMat = new THREE.MeshBasicMaterial({
          map: tex,
          transparent: true,
          side: THREE.DoubleSide
        });
        const badge = new THREE.Mesh(new THREE.PlaneGeometry(0.78, 0.78), badgeMat);
        badge.position.set(0, 1.15, 0);
        custGroup.add(badge);

        queueGroup.add(custGroup);
      };

      // Render Serving Customer at register
      if (serving) {
        renderCustomerFigure(serving.basket, 0.4, true);
      }

      // Render Waiting Customers in Queue
      if (customers.length > 0) {
        customers.forEach((cust, idx) => {
          renderCustomerFigure(cust.basket, -1.35 - idx * 1.05, false);
        });
      } else {
        // Fallback for visual representation if queue length is given without entity list
        for (let idx = 0; idx < qLen; idx++) {
          const fallbackBasket = ((laneId * 3 + idx * 4) % 15) + 2;
          renderCustomerFigure(fallbackBasket, -1.35 - idx * 1.05, false);
        }
      }
    });
  }, [
    recommendedAction,
    currentQueues,
    capacities,
    laneCustomers,
    openStatus
  ]);

  return (
    <div className="relative w-full h-full flex flex-col rounded-xl border border-slate-800 bg-[#090d16] overflow-hidden shadow-2xl select-none">
      {/* 1. Header Bar with Title and Status */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800/80 bg-slate-900/80 backdrop-blur z-20">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-200 tracking-wide">
            Supermarket Checkout Area (10 Lanes)
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800/60 text-cyan-400 font-medium">
            3D Spatial Simulation Floor
          </span>
        </div>
        <div className="flex items-center gap-4 text-[11px] text-slate-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_6px_#22d3ee]" />
            <span className="text-cyan-300 font-medium">RL Optimized Target (Lane #{recommendedAction})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Active Checkout Stations</span>
          </div>
        </div>
      </div>

      {/* 2. Three.js 3D WebGL Canvas Container */}
      <div className="relative flex-1 w-full h-[420px] min-h-[390px] overflow-hidden">
        {/* Canvas Mount */}
        <div ref={containerRef} className="w-full h-full" />

        {/* 3. Overhead Station Signposts across the 10 stations */}
        <div className="absolute top-2 inset-x-2 flex items-center justify-between pointer-events-none z-10 px-2">
          {LANE_CONFIG.map((cfg) => {
            const laneId = cfg.id;
            const isTarget = recommendedAction === laneId;
            const isOpen = openStatus[laneId] !== undefined ? openStatus[laneId] : true;
            const type = laneTypes[laneId] || cfg.type;

            let typeName = cfg.name;
            if (type === 'express') typeName = 'Express';
            else if (type === 'large_basket') typeName = 'Bulk Cart';
            else if (type === 'flexible') typeName = 'Self-Serve';
            else typeName = 'Standard';

            return (
              <div
                key={laneId}
                onClick={() => onSelectLane && onSelectLane(laneId)}
                className="flex-1 flex flex-col items-center pointer-events-auto cursor-pointer"
              >
                {/* Floating RL Target Badge */}
                {isTarget && (
                  <div className="mb-1 px-2 py-0.5 rounded-full bg-cyan-400 text-slate-950 font-black text-[9px] tracking-wider uppercase shadow-[0_0_12px_#22d3ee] flex items-center gap-1 animate-bounce">
                    <span>★</span>
                    <span>RL TARGET</span>
                  </div>
                )}

                {/* Overhead Station Box */}
                <div
                  className={`px-2 py-1 rounded-md text-center transition-all ${
                    isTarget
                      ? 'bg-cyan-950/90 border border-cyan-400 text-cyan-200 shadow-[0_0_10px_rgba(6,182,212,0.4)]'
                      : 'bg-slate-950/80 border border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="text-[11px] font-mono font-bold leading-tight">
                    {cfg.label}
                  </div>
                  <div className="text-[9px] text-slate-400 font-medium leading-tight">
                    {typeName}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* 4. Bottom Station Occupancy & Open Status Overlay */}
        <div className="absolute bottom-2 inset-x-2 flex items-center justify-between pointer-events-none z-10 px-2">
          {LANE_CONFIG.map((cfg) => {
            const laneId = cfg.id;
            const maxCap = capacities[laneId] || cfg.capacity;
            const qLen = currentQueues[laneId] !== undefined ? currentQueues[laneId] : 0;
            const isOpen = openStatus[laneId] !== undefined ? openStatus[laneId] : true;
            const isTarget = recommendedAction === laneId;

            return (
              <div
                key={laneId}
                onClick={() => onSelectLane && onSelectLane(laneId)}
                className={`flex-1 text-center py-1 px-1 rounded-md pointer-events-auto cursor-pointer transition-all ${
                  isTarget ? 'bg-cyan-950/60 border border-cyan-500/50' : 'bg-slate-950/60'
                }`}
              >
                <div className="text-[10px] font-mono font-bold text-slate-200">
                  {qLen} / {maxCap}
                </div>
                <div
                  className={`text-[9px] font-medium leading-tight ${
                    isOpen ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {isOpen ? 'Open' : 'Closed'}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
