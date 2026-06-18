import type { FunctionComponent } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import * as THREE from 'three';

interface ParticleBrainProps {
  height?: number;
  density?: number;
}

const ParticleBrain: FunctionComponent<ParticleBrainProps> = ({ height = 460, density = 1 }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const particlesRef = useRef<THREE.Points | null>(null);
  const linesRef = useRef<THREE.LineSegments | null>(null);
  const raycasterRef = useRef<THREE.Raycaster | null>(null);
  const mouseRef = useRef({ x: 0, y: 0, vx: 0, vy: 0 });
  const animationRef = useRef<number | null>(null);
  const rotationRef = useRef({ x: 0, y: 0 });
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    // Guard SSR
    if (typeof window === 'undefined' || !containerRef.current) {
      return;
    }

    // Check for reduced motion preference
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(motionQuery.matches);

    // Compute particle count (400..2000 range)
    const particleCount = Math.max(400, Math.min(2000, Math.floor(1200 * density)));

    let scene: THREE.Scene | null = null;
    let renderer: THREE.WebGLRenderer | null = null;
    let camera: THREE.PerspectiveCamera | null = null;
    let particles: THREE.Points | null = null;
    let lines: THREE.LineSegments | null = null;
    let raycaster: THREE.Raycaster | null = null;
    let frameId: number | null = null;

    const init = () => {
      // Create scene
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0e14);
      sceneRef.current = scene;

      // Create camera
      const rect = containerRef.current!.getBoundingClientRect();
      const width = rect.width || 800;
      const aspect = width / height;
      camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
      camera.position.z = 2.5;

      // Create renderer with capped pixel ratio
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      containerRef.current!.appendChild(renderer.domElement);
      rendererRef.current = renderer;
      raycasterRef.current = new THREE.Raycaster();

      // Generate particle positions: dual-lobe brain structure
      const positions = new Float32Array(particleCount * 3);
      const colors = new Float32Array(particleCount * 3);

      const cyan = new THREE.Color(0x22d3ee);
      const violet = new THREE.Color(0xa855f7);

      for (let i = 0; i < particleCount; i++) {
        // Alternate between two lobes slightly offset
        const lobe = i % 2;
        const lobeOffset = lobe === 0 ? -0.6 : 0.6;

        // Perturbed sphere within each lobe
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(Math.random() * 2 - 1);
        const radius = 0.6 + Math.random() * 0.3;

        positions[i * 3] = Math.sin(phi) * Math.cos(theta) * radius + lobeOffset;
        positions[i * 3 + 1] = Math.sin(phi) * Math.sin(theta) * radius + (Math.random() - 0.5) * 0.4;
        positions[i * 3 + 2] = Math.cos(phi) * radius + (Math.random() - 0.5) * 0.2;

        // Interpolate color between cyan and violet based on height
        const t = (positions[i * 3 + 1] + 1) / 2;
        const color = cyan.clone().lerp(violet, t);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
      }

      // Create particles geometry
      const particleGeom = new THREE.BufferGeometry();
      particleGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      particleGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      // Particle material: additive, soft glow
      const particleMat = new THREE.PointsMaterial({
        size: 0.08,
        sizeAttenuation: true,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
      });

      particles = new THREE.Points(particleGeom, particleMat);
      scene.add(particles);
      particlesRef.current = particles;

      // Build synapse connections (sparse, nearby only)
      const posArray = particleGeom.attributes.position.array as Float32Array;
      const lineSegments = [];

      for (let i = 0; i < particleCount; i++) {
        let connectedThisFrame = 0;
        for (let j = i + 1; j < particleCount; j++) {
          if (lineSegments.length >= 600) break; // Cap segments

          const dx = posArray[j * 3] - posArray[i * 3];
          const dy = posArray[j * 3 + 1] - posArray[i * 3 + 1];
          const dz = posArray[j * 3 + 2] - posArray[i * 3 + 2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist < 0.8) {
            lineSegments.push(i, j);
            connectedThisFrame++;
            if (connectedThisFrame >= 3) break; // Limit connections per particle
          }
        }
        if (lineSegments.length >= 600) break;
      }

      // Create lines geometry
      const lineGeom = new THREE.BufferGeometry();
      lineGeom.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      lineGeom.setIndex(new THREE.BufferAttribute(new Uint32Array(lineSegments), 1));

      const lineMat = new THREE.LineBasicMaterial({
        color: 0x22d3ee,
        transparent: true,
        opacity: 0.15,
        blending: THREE.AdditiveBlending,
        linewidth: 1,
      });

      lines = new THREE.LineSegments(lineGeom, lineMat);
      scene.add(lines);
      linesRef.current = lines;
    };

    const animate = () => {
      if (!scene || !renderer || !camera || !particles || prefersReducedMotion) {
        return;
      }

      // Auto-rotate
      rotationRef.current.y += 0.0003;
      rotationRef.current.x += 0.00015;

      // Apply pointer parallax with smoothing
      const targetRot = { x: mouseRef.current.vy * 0.5, y: mouseRef.current.vx * 0.5 };
      rotationRef.current.x += (targetRot.x - rotationRef.current.x) * 0.1;
      rotationRef.current.y += (targetRot.y - rotationRef.current.y) * 0.1;

      particles.rotation.x = rotationRef.current.x;
      particles.rotation.y = rotationRef.current.y;

      if (linesRef.current) {
        linesRef.current.rotation.x = rotationRef.current.x;
        linesRef.current.rotation.y = rotationRef.current.y;
      }

      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };

    const onPointerMove = (evt: PointerEvent) => {
      const rect = renderer!.domElement.getBoundingClientRect();
      const x = (evt.clientX - rect.left) / rect.width;
      const y = (evt.clientY - rect.top) / rect.height;

      mouseRef.current.vx = (x - 0.5) * 2;
      mouseRef.current.vy = (y - 0.5) * 2;
    };

    const onWindowResize = () => {
      if (!renderer || !camera || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const w = rect.width || 800;
      const h = height;

      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };

    try {
      init();
      animate();

      // Add listeners
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('resize', onWindowResize);

      return () => {
        // Cleanup
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('resize', onWindowResize);

        if (frameId !== null) {
          cancelAnimationFrame(frameId);
        }

        if (renderer && containerRef.current) {
          containerRef.current.removeChild(renderer.domElement);
        }

        // Dispose Three.js resources
        if (particles) {
          particles.geometry.dispose();
          (particles.material as THREE.Material).dispose();
        }

        if (lines) {
          lines.geometry.dispose();
          (lines.material as THREE.Material).dispose();
        }

        if (renderer) {
          renderer.dispose();
        }
      };
    } catch (err) {
      // Graceful fallback on WebGL error
      console.warn('WebGL unavailable, rendering fallback:', err);
      return undefined;
    }
  }, [height, density, prefersReducedMotion]);

  // Static fallback for reduced motion or WebGL failure
  if (prefersReducedMotion) {
    return (
      <div
        ref={containerRef}
        data-testid="particle-brain"
        role="img"
        aria-label="An abstract neural cloud representing a language model's internal structure."
        style={{
          width: '100%',
          height: `${height}px`,
          background: 'radial-gradient(circle at 50% 50%, rgba(34, 211, 238, 0.15) 0%, rgba(168, 85, 247, 0.08) 30%, #0a0e14 100%)',
          borderRadius: '12px',
        }}
      />
    );
  }

  return (
    <div
      ref={containerRef}
      data-testid="particle-brain"
      role="img"
      aria-label="An abstract neural cloud representing a language model's internal structure."
      style={{
        width: '100%',
        height: `${height}px`,
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '12px',
        background: '#0a0e14',
      }}
    />
  );
};

export default ParticleBrain;
