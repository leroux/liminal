// ─── Tonnetz hexagonal grid renderer ────────────────────────────────
// Vertices = pitch classes. Triangles = triads.
// Horizontal = fifths (+7). Diagonal up-right = major thirds (+4).
// Down-pointing triangle = major triad. Up-pointing = minor triad.

import {
  NOTE_NAMES, pcsToBits, bitsToArray, closeVoicing, makeTriad,
} from './core.js';
import { plrNeighbors } from './navigation.js';
import { type TriadDissonance } from './dissonance.js';

// ─── Grid geometry ──────────────────────────────────────────────────

const COLS = 15;
const ROWS = 8;
const DX = 68;  // horizontal spacing between vertices
const DY = 59;  // vertical spacing (DX * sqrt(3)/2 for equilateral)
const VERTEX_R = 14;
const PAD_X = 50;
const PAD_Y = 40;

interface Vertex {
  q: number;
  r: number;
  pc: number;
  x: number;
  y: number;
}

interface TriangleCell {
  q: number;
  r: number;
  isDown: boolean;  // down-pointing = major
  root: number;
  isMajor: boolean;
  bits: number;
  vertices: [Vertex, Vertex, Vertex];
  cx: number;
  cy: number;
}

function buildGrid(): { vertices: Vertex[][]; triangles: TriangleCell[] } {
  const vertices: Vertex[][] = [];

  for (let r = 0; r < ROWS; r++) {
    const row: Vertex[] = [];
    for (let q = 0; q < COLS; q++) {
      const pc = ((q * 7 + r * 4) % 12 + 12) % 12;
      const x = PAD_X + q * DX + r * (DX / 2);
      const y = PAD_Y + r * DY;
      row.push({ q, r, pc, x, y });
    }
    vertices.push(row);
  }

  const triangles: TriangleCell[] = [];

  for (let r = 0; r < ROWS - 1; r++) {
    for (let q = 0; q < COLS - 1; q++) {
      // Down-pointing triangle: (q,r), (q+1,r), (q,r+1)
      // PCs: base, base+7, base+4 → {root, root+4, root+7} = major triad
      const v0 = vertices[r][q];
      const v1 = vertices[r][q + 1];
      const v2 = vertices[r + 1][q];
      const rootMaj = v0.pc;
      const bitsMaj = pcsToBits([rootMaj, (rootMaj + 4) % 12, (rootMaj + 7) % 12]);

      triangles.push({
        q, r, isDown: true,
        root: rootMaj, isMajor: true, bits: bitsMaj,
        vertices: [v0, v1, v2],
        cx: (v0.x + v1.x + v2.x) / 3,
        cy: (v0.y + v1.y + v2.y) / 3,
      });

      // Up-pointing triangle: (q+1,r), (q,r+1), (q+1,r+1)
      // PCs: base+7, base+4, base+11 → minor triad rooted at base+4
      const v3 = vertices[r + 1][q + 1];
      const rootMin = (v0.pc + 4) % 12;
      const bitsMin = pcsToBits([rootMin, (rootMin + 3) % 12, (rootMin + 7) % 12]);

      triangles.push({
        q, r, isDown: false,
        root: rootMin, isMajor: false, bits: bitsMin,
        vertices: [v1, v2, v3],
        cx: (v1.x + v2.x + v3.x) / 3,
        cy: (v1.y + v2.y + v3.y) / 3,
      });
    }
  }

  return { vertices, triangles };
}

// ─── Color helpers ──────────────────────────────────────────────────

function consonanceColor(score: number, alpha: number = 1.0): string {
  // score: 0=dissonant (cool purple), 1=consonant (warm gold)
  const hue = score * score * 50;    // 0→0 (red-purple), 1→50 (gold)
  const sat = 70 + score * 20;       // 70-90%
  const light = 25 + score * 30;     // 25-55%
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

function plrColor(op: string): string {
  switch (op) {
    case 'P': return '#ff6b6b';  // red
    case 'L': return '#4ecdc4';  // teal
    case 'R': return '#45b7d1';  // blue
    default: return '#888';
  }
}

// ─── Renderer ───────────────────────────────────────────────────────

export interface TonnetzState {
  activeTriad: { root: number; isMajor: boolean } | null;
  dissonanceMap: Map<number, TriadDissonance>;
  hoveredTriangle: TriangleCell | null;
}

export class TonnetzRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private grid: ReturnType<typeof buildGrid>;
  private dpr: number;

  state: TonnetzState = {
    activeTriad: null,
    dissonanceMap: new Map(),
    hoveredTriangle: null,
  };

  onTriadClick: ((root: number, isMajor: boolean, bits: number, midi: number[]) => void) | null = null;
  onTriadHover: ((tri: TriangleCell | null) => void) | null = null;
  onAddToProgression: ((bits: number) => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.dpr = window.devicePixelRatio || 1;
    this.grid = buildGrid();

    this.resize();
    this.setupEvents();
  }

  resize(): void {
    const w = PAD_X * 2 + (COLS - 1) * DX + (ROWS - 1) * DX / 2;
    const h = PAD_Y * 2 + (ROWS - 1) * DY;

    this.canvas.width = w * this.dpr;
    this.canvas.height = h * this.dpr;
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  private setupEvents(): void {
    this.canvas.addEventListener('click', (e) => {
      const tri = this.hitTest(e.offsetX, e.offsetY);
      if (tri) {
        this.state.activeTriad = { root: tri.root, isMajor: tri.isMajor };
        const midi = closeVoicing(tri.bits, 4);
        this.onTriadClick?.(tri.root, tri.isMajor, tri.bits, midi);
        this.render();
      }
    });

    this.canvas.addEventListener('mousemove', (e) => {
      const tri = this.hitTest(e.offsetX, e.offsetY);
      if (tri !== this.state.hoveredTriangle) {
        this.state.hoveredTriangle = tri;
        this.canvas.style.cursor = tri ? 'pointer' : 'default';
        this.onTriadHover?.(tri);
        this.render();
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.state.hoveredTriangle = null;
      this.canvas.style.cursor = 'default';
      this.render();
    });

    this.canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const tri = this.hitTest(e.offsetX, e.offsetY);
      if (tri) this.onAddToProgression?.(tri.bits);
    });
  }

  private hitTest(mx: number, my: number): TriangleCell | null {
    // Find closest triangle centroid
    let best: TriangleCell | null = null;
    let bestDist = Infinity;

    for (const tri of this.grid.triangles) {
      // Point-in-triangle test
      if (this.pointInTriangle(mx, my, tri.vertices)) {
        const dx = mx - tri.cx;
        const dy = my - tri.cy;
        const d = dx * dx + dy * dy;
        if (d < bestDist) {
          bestDist = d;
          best = tri;
        }
      }
    }
    return best;
  }

  private pointInTriangle(px: number, py: number, verts: [Vertex, Vertex, Vertex]): boolean {
    const [v0, v1, v2] = verts;
    const d1 = this.sign(px, py, v0.x, v0.y, v1.x, v1.y);
    const d2 = this.sign(px, py, v1.x, v1.y, v2.x, v2.y);
    const d3 = this.sign(px, py, v2.x, v2.y, v0.x, v0.y);
    const hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    const hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(hasNeg && hasPos);
  }

  private sign(px: number, py: number, x1: number, y1: number, x2: number, y2: number): number {
    return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2);
  }

  // ─── Rendering ──────────────────────────────────────────────────

  render(): void {
    const { ctx } = this;
    const w = this.canvas.width / this.dpr;
    const h = this.canvas.height / this.dpr;

    // Background
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, w, h);

    // Determine PLR neighbors of active triad
    const plrMap = new Map<string, string>();
    if (this.state.activeTriad) {
      const neighbors = plrNeighbors(this.state.activeTriad.root, this.state.activeTriad.isMajor);
      for (const n of neighbors) {
        plrMap.set(`${n.root}-${n.isMajor}`, n.op);
      }
    }

    // Draw triangles
    for (const tri of this.grid.triangles) {
      const key = `${tri.root}-${tri.isMajor}`;
      const isActive = this.state.activeTriad &&
        tri.root === this.state.activeTriad.root &&
        tri.isMajor === this.state.activeTriad.isMajor;
      const isPLR = plrMap.has(key);
      const isHovered = tri === this.state.hoveredTriangle;

      // Consonance coloring
      const diss = this.state.dissonanceMap.get(tri.bits);
      const consonance = diss ? diss.consonance : 0.5;

      this.drawTriangle(tri, consonance, isActive ?? false, isPLR, isHovered, plrMap.get(key));
    }

    // Draw edges (grid lines)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 0.5;
    for (let r = 0; r < ROWS; r++) {
      for (let q = 0; q < COLS; q++) {
        const v = this.grid.vertices[r][q];
        // Right neighbor
        if (q < COLS - 1) {
          const vr = this.grid.vertices[r][q + 1];
          ctx.beginPath();
          ctx.moveTo(v.x, v.y);
          ctx.lineTo(vr.x, vr.y);
          ctx.stroke();
        }
        // Down-left neighbor
        if (r < ROWS - 1) {
          const vd = this.grid.vertices[r + 1][q];
          ctx.beginPath();
          ctx.moveTo(v.x, v.y);
          ctx.lineTo(vd.x, vd.y);
          ctx.stroke();
        }
        // Down-right diagonal
        if (r < ROWS - 1 && q < COLS - 1) {
          const vdr = this.grid.vertices[r + 1][q + 1];
          ctx.beginPath();
          ctx.moveTo(v.x, v.y);
          ctx.lineTo(vdr.x, vdr.y);
          ctx.stroke();
        }
      }
    }

    // Draw vertices
    for (let r = 0; r < ROWS; r++) {
      for (let q = 0; q < COLS; q++) {
        const v = this.grid.vertices[r][q];
        this.drawVertex(v);
      }
    }

    // Draw active chord highlight ring
    if (this.state.activeTriad) {
      for (const tri of this.grid.triangles) {
        if (tri.root === this.state.activeTriad.root && tri.isMajor === this.state.activeTriad.isMajor) {
          this.drawActiveRing(tri);
        }
      }
    }
  }

  private drawTriangle(
    tri: TriangleCell,
    consonance: number,
    isActive: boolean,
    isPLR: boolean,
    isHovered: boolean,
    plrOp?: string,
  ): void {
    const { ctx } = this;
    const [v0, v1, v2] = tri.vertices;

    ctx.beginPath();
    ctx.moveTo(v0.x, v0.y);
    ctx.lineTo(v1.x, v1.y);
    ctx.lineTo(v2.x, v2.y);
    ctx.closePath();

    // Fill based on state
    if (isActive) {
      const hue = consonance * 50;
      ctx.fillStyle = `hsla(${hue}, 90%, 50%, 0.6)`;
      ctx.fill();
    } else if (isPLR) {
      ctx.fillStyle = plrOp ? plrColor(plrOp).replace(')', ', 0.25)').replace('rgb', 'rgba') : 'rgba(255,255,255,0.1)';
      // Parse hex to rgba
      const hex = plrOp ? plrColor(plrOp) : '#888';
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.2)`;
      ctx.fill();
    } else if (isHovered) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.fill();
    } else {
      // Subtle consonance-based fill
      ctx.fillStyle = consonanceColor(consonance, 0.15);
      ctx.fill();
    }

    // Border for PLR neighbors
    if (isPLR && plrOp) {
      ctx.strokeStyle = plrColor(plrOp);
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Hover outline
    if (isHovered && !isActive) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Chord name in center
    const name = NOTE_NAMES[tri.root] + (tri.isMajor ? '' : 'm');
    ctx.fillStyle = isActive
      ? 'rgba(255, 255, 255, 0.95)'
      : isPLR
        ? 'rgba(255, 255, 255, 0.7)'
        : isHovered
          ? 'rgba(255, 255, 255, 0.6)'
          : 'rgba(255, 255, 255, 0.25)';
    ctx.font = isActive ? 'bold 10px system-ui' : '9px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(name, tri.cx, tri.cy);
  }

  private drawVertex(v: Vertex): void {
    const { ctx } = this;

    // Highlight vertex if it belongs to active chord
    let isInActive = false;
    if (this.state.activeTriad) {
      const tri = makeTriad(this.state.activeTriad.root, this.state.activeTriad.isMajor);
      const pcs = bitsToArray(tri.bits);
      isInActive = pcs.includes(v.pc);
    }

    ctx.beginPath();
    ctx.arc(v.x, v.y, isInActive ? VERTEX_R * 0.5 : VERTEX_R * 0.35, 0, Math.PI * 2);
    ctx.fillStyle = isInActive ? 'rgba(255, 220, 100, 0.9)' : 'rgba(255, 255, 255, 0.15)';
    ctx.fill();

    // Note name
    ctx.fillStyle = isInActive ? '#fff' : 'rgba(255, 255, 255, 0.35)';
    ctx.font = isInActive ? 'bold 9px system-ui' : '8px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(NOTE_NAMES[v.pc], v.x, v.y - (isInActive ? 12 : 10));
  }

  private drawActiveRing(tri: TriangleCell): void {
    const { ctx } = this;
    const [v0, v1, v2] = tri.vertices;

    ctx.beginPath();
    ctx.moveTo(v0.x, v0.y);
    ctx.lineTo(v1.x, v1.y);
    ctx.lineTo(v2.x, v2.y);
    ctx.closePath();

    ctx.strokeStyle = 'rgba(255, 220, 100, 0.8)';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Glow
    ctx.shadowColor = 'rgba(255, 220, 100, 0.4)';
    ctx.shadowBlur = 12;
    ctx.strokeStyle = 'rgba(255, 220, 100, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}
