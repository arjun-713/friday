"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

/** Original illustrative router, not a manufacturer repair or disassembly guide. */
export default function DeviceScene({ expanded }: { expanded: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const expandedRef = useRef(expanded);
  const [available, setAvailable] = useState(true);
  useEffect(() => { expandedRef.current = expanded; }, [expanded]);

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" }); }
    catch { setAvailable(false); return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.5;
    element.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 50);
    camera.position.set(5.5, 4.7, 7.5);
    camera.lookAt(0, 0.65, 0);
    scene.add(new THREE.AmbientLight(0xc6dbc9, 2.2));
    const key = new THREE.DirectionalLight(0xf3ffe9, 5); key.position.set(-3, 6, 5); scene.add(key);
    const rim = new THREE.DirectionalLight(0x83cba2, 4); rim.position.set(4, 2, -3); scene.add(rim);
    const fill = new THREE.DirectionalLight(0xe3e9ff, 1.5); fill.position.set(-4, 1, -2); scene.add(fill);
    const device = new THREE.Group(); scene.add(device);
    const shell = new THREE.MeshStandardMaterial({ color: 0xdce4d9, roughness: 0.3, metalness: 0.4 });
    const edge = new THREE.MeshStandardMaterial({ color: 0x4d5b51, roughness: 0.4, metalness: 0.65 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x17251d, roughness: 0.55, metalness: 0.2 });
    const boardMat = new THREE.MeshStandardMaterial({ color: 0x284e38, metalness: 0.4, roughness: 0.4 });
    const gold = new THREE.MeshStandardMaterial({ color: 0xafb993, metalness: 0.75, roughness: 0.35 });
    function box(w: number, h: number, d: number, material: THREE.Material, x: number, y: number, z: number, parent: THREE.Group = device, radius = 0.04) {
      const mesh = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 3, radius), material);
      mesh.position.set(x, y, z); parent.add(mesh); return mesh;
    }
    const bottom = new THREE.Group(); device.add(bottom);
    box(3.7, 0.2, 2.35, dark, 0, 0, 0, bottom, 0.09);
    box(3.68, 0.11, 2.32, edge, 0, 0.13, 0, bottom);
    const lid = new THREE.Group(); device.add(lid);
    box(3.72, 0.25, 2.36, shell, 0, 0.31, 0, lid, 0.11);
    for (let i = 0; i < 22; i++) {
      box(0.032, 0.008, 0.57, edge, -1.4 + i * 0.135, 0.44, -0.61, lid, 0.003);
    }
    const emblem = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.016, 8, 40, Math.PI * 1.6), edge);
    emblem.rotation.x = -Math.PI / 2; emblem.position.set(0, 0.444, 0.25); lid.add(emblem);
    box(0.025, 0.012, 0.15, edge, 0, 0.443, 0.25, lid, 0.004);
    const board = new THREE.Group(); device.add(board);
    box(3.34, 0.045, 1.98, boardMat, 0, 0.2, 0, board);
    box(0.6, 0.08, 0.6, dark, -0.4, 0.25, 0, board);
    box(0.34, 0.07, 0.48, edge, 0.6, 0.25, 0.2, board);
    for (let i = 0; i < 14; i++) {
      box(0.035, 0.02, 0.13, gold, -0.72 + (i % 7) * 0.105, 0.25, i < 7 ? -0.36 : 0.36, board, 0.002);
    }
    for (let i = 0; i < 5; i++) {
      box(0.34, 0.21, 0.12, i === 4 ? gold : edge, -1.2 + i * 0.5, 0.12, -1.18, bottom, 0.025);
      const ledMat = new THREE.MeshBasicMaterial({ color: i === 4 ? 0xe3bb7d : 0x9fedb2 });
      box(0.07, 0.032, 0.02, ledMat, -0.45 + i * 0.22, 0.25, 1.185, lid, 0.005);
    }
    [-1.55, 0, 1.55].forEach((x) => {
      const antenna = new THREE.Group(); antenna.position.set(x, 0.18, -0.95); antenna.rotation.z = -x * 0.11; bottom.add(antenna);
      box(0.16, 2.05, 0.14, shell, 0, 0.96, 0, antenna, 0.065);
      box(0.2, 0.22, 0.2, edge, 0, 0.05, 0, antenna);
    });
    const plane = new THREE.Mesh(new THREE.CircleGeometry(3.0, 80), new THREE.MeshBasicMaterial({ color: 0x354a3c, transparent: true, opacity: 0.16, side: THREE.DoubleSide }));
    plane.rotation.x = -Math.PI / 2; plane.position.y = -0.48; scene.add(plane);
    const ring = new THREE.Mesh(new THREE.RingGeometry(2.7, 2.705, 100), new THREE.MeshBasicMaterial({ color: 0x84b893, transparent: true, opacity: 0.32, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = -0.47; scene.add(ring);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    let visible = true, frame = 0, pointerX = 0, pointerY = 0, last = 0;
    const move = (event: PointerEvent) => {
      const bounds = element.getBoundingClientRect();
      pointerX = (event.clientX - bounds.left) / bounds.width - 0.5;
      pointerY = (event.clientY - bounds.top) / bounds.height - 0.5;
    };
    const reset = () => { pointerX = 0; pointerY = 0; };
    const resize = new ResizeObserver(() => {
      const { width, height } = element.getBoundingClientRect();
      renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
    }); resize.observe(element);
    const visibility = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }); visibility.observe(element);
    function render(time: number) {
      frame = requestAnimationFrame(render);
      if (!visible || document.hidden || time - last < 32) return;
      last = time;
      const target = expandedRef.current ? 1.18 : 0;
      lid.position.y += (target - lid.position.y) * (reduced.matches ? 1 : 0.065);
      board.position.y += (target * 0.45 - board.position.y) * (reduced.matches ? 1 : 0.065);
      device.rotation.y += ((reduced.matches ? -0.2 : -0.2 + pointerX * 0.42) - device.rotation.y) * 0.06;
      device.rotation.x += ((reduced.matches ? 0 : pointerY * 0.1) - device.rotation.x) * 0.06;
      renderer.render(scene, camera);
    }
    element.addEventListener("pointermove", move); element.addEventListener("pointerleave", reset);
    frame = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(frame); resize.disconnect(); visibility.disconnect();
      element.removeEventListener("pointermove", move); element.removeEventListener("pointerleave", reset);
      const materials = new Set<THREE.Material>();
      scene.traverse((object) => { if (object instanceof THREE.Mesh) { object.geometry.dispose(); (Array.isArray(object.material) ? object.material : [object.material]).forEach((material: THREE.Material) => materials.add(material)); } });
      materials.forEach((material) => material.dispose()); renderer.dispose(); renderer.domElement.remove();
    };
  }, []);

  return <div ref={host} className="device-canvas" role="img" aria-label="Interactive 3D illustration of a router with an optional exploded component view">{!available && <div className="device-fallback"><svg viewBox="0 0 300 220" aria-hidden="true" fill="none"><path d="M60 140V45m90 95V25m90 115V45" stroke="#c7dbc9" strokeWidth="10" strokeLinecap="round"/><rect x="30" y="130" width="240" height="60" rx="18" fill="#c7dbc9"/><path d="M75 160h20m15 0h20m15 0h20" stroke="#326043" strokeWidth="5" strokeLinecap="round"/></svg><span>Router illustration</span></div>}</div>;
}
