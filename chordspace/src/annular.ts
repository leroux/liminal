// ─── Annular (Tymoczko polar) projection ────────────────────────────
// Angle = pitch-class sum mod 12 (maps to clock position)
// Radius = evenness (DFT magnitude) — even chords near center, clustered at rim
// Augmented triads / dim7 at center, chromatic clusters at edge.

import {
  NOTE_NAMES, bitsToArray, closeVoicing,
} from './core.js';

export interface AnnularNode {
  bits: number;
  label: string;
  cardinality: number;
  isNamed: boolean;
  names: string[];
  consonance: number;
  evennessVal: number;
  pcSum: number;       // sum of pitch classes mod 12
  angle: number;       // radians
  radius: number;      // normalized 0–1, 0=center (even), 1=rim (clustered)
  screenX: number;
  screenY: number;
  idealX: number;      // ideal position before collision resolution
  idealY: number;
  dotR: number;        // visual radius
}

const CARD_COLORS: Record<number, string> = {
  2: '#ff6b6b',
  3: '#ffd764',
  4: '#4ecdc4',
  5: '#45b7d1',
  6: '#a78bfa',
  7: '#f472b6',
};

export function buildAnnularNodes(
  nodes: { bits: number; label: string; cardinality: number; isNamed: boolean; names: string[]; consonance: number; evennessVal: number }[],
): AnnularNode[] {
  return nodes.map(n => {
    const pcs = bitsToArray(n.bits);
    const pcSum = pcs.reduce((a, b) => a + b, 0) % 12;
    const angle = (pcSum / 12) * Math.PI * 2 - Math.PI / 2; // 0 at top
    // Invert evenness: even chords (aug triad, dim7) go to center
    const radius = 1 - n.evennessVal;

    return {
      ...n,
      pcSum,
      angle,
      radius,
      screenX: 0,
      screenY: 0,
      idealX: 0,
      idealY: 0,
      dotR: 3 + n.cardinality * 1.0,
    };
  });
}

export function layoutAnnular(nodes: AnnularNode[], cx: number, cy: number, maxR: number): void {
  // Set ideal positions from polar coordinates
  for (const node of nodes) {
    const r = 20 + node.radius * (maxR - 30);
    node.idealX = cx + Math.cos(node.angle) * r;
    node.idealY = cy + Math.sin(node.angle) * r;
    node.screenX = node.idealX;
    node.screenY = node.idealY;
  }

  resolveCollisions(nodes, cx, cy, maxR);
}

// ─── Collision resolution ────────────────────────────────────────────
// Force-directed separation: push overlapping nodes apart while a weak
// spring pulls them back toward their ideal (angle, radius) position.

function resolveCollisions(nodes: AnnularNode[], cx: number, cy: number, maxR: number): void {
  const PADDING = 2;          // extra gap between node edges
  const SPRING = 0.1;         // fraction to pull back toward ideal each iteration
  const ITERATIONS = 80;
  const BOUNDARY = maxR + 20; // max distance from center

  for (let iter = 0; iter < ITERATIONS; iter++) {
    let maxMove = 0;

    // Pairwise separation
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        const dx = b.screenX - a.screenX;
        const dy = b.screenY - a.screenY;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.001;
        const minDist = a.dotR + b.dotR + PADDING;

        if (dist < minDist) {
          const overlap = (minDist - dist) / 2;
          const nx = (dx / dist) * overlap;
          const ny = (dy / dist) * overlap;
          a.screenX -= nx;
          a.screenY -= ny;
          b.screenX += nx;
          b.screenY += ny;
          if (overlap > maxMove) maxMove = overlap;
        }
      }
    }

    // Spring: pull each node back toward its ideal position
    for (const node of nodes) {
      const dix = node.idealX - node.screenX;
      const diy = node.idealY - node.screenY;
      node.screenX += dix * SPRING;
      node.screenY += diy * SPRING;

      // Clamp inside boundary (keep nodes on canvas)
      const dcx = node.screenX - cx;
      const dcy = node.screenY - cy;
      const distFromCenter = Math.sqrt(dcx * dcx + dcy * dcy);
      if (distFromCenter > BOUNDARY) {
        const scale = BOUNDARY / distFromCenter;
        node.screenX = cx + dcx * scale;
        node.screenY = cy + dcy * scale;
      }
    }

    // Early exit if displacements are tiny
    if (iter > 10 && maxMove < 0.1) break;
  }
}

// ─── Renderer ───────────────────────────────────────────────────────

export class AnnularRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr: number;
  private nodes: AnnularNode[] = [];
  private hoveredNode: AnnularNode | null = null;
  private _activeNode: AnnularNode | null = null;
  private width = 0;
  private height = 0;

  // Zoom / pan
  private panX = 0;
  private panY = 0;
  private zoom = 1;
  private isPanning = false;
  private lastMouse = { x: 0, y: 0 };

  onChordClick: ((node: AnnularNode, midi: number[]) => void) | null = null;
  onChordHover: ((node: AnnularNode | null) => void) | null = null;
  onAddToProgression: ((bits: number) => void) | null = null;

  get activeNode(): AnnularNode | null { return this._activeNode; }
  set activeNode(n: AnnularNode | null) { this._activeNode = n; }

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.dpr = window.devicePixelRatio || 1;
    this.setupEvents();
  }

  setNodes(nodes: AnnularNode[]): void {
    this.nodes = nodes;
    // Reset zoom/pan when dataset changes
    this.panX = 0;
    this.panY = 0;
    this.zoom = 1;
  }

  getNodes(): AnnularNode[] { return this.nodes; }

  focusOn(node: AnnularNode): void {
    this._activeNode = node;
    this.panX = this.width / 2 - node.screenX * this.zoom;
    this.panY = this.height / 2 - node.screenY * this.zoom;
    this.render();
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.canvas.width = width * this.dpr;
    this.canvas.height = height * this.dpr;
    this.canvas.style.width = width + 'px';
    this.canvas.style.height = height + 'px';
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  private toWorld(sx: number, sy: number): [number, number] {
    return [(sx - this.panX) / this.zoom, (sy - this.panY) / this.zoom];
  }

  private setupEvents(): void {
    this.canvas.addEventListener('click', (e) => {
      if (this.isPanning) return;
      const node = this.hitTest(e.offsetX, e.offsetY);
      if (node) {
        this._activeNode = node;
        const midi = closeVoicing(node.bits, 4);
        this.onChordClick?.(node, midi);
        this.render();
      }
    });

    this.canvas.addEventListener('mousemove', (e) => {
      if (this.isPanning) {
        this.panX += e.offsetX - this.lastMouse.x;
        this.panY += e.offsetY - this.lastMouse.y;
        this.lastMouse = { x: e.offsetX, y: e.offsetY };
        this.render();
        return;
      }

      const node = this.hitTest(e.offsetX, e.offsetY);
      if (node !== this.hoveredNode) {
        this.hoveredNode = node;
        this.canvas.style.cursor = node ? 'pointer' : 'grab';
        this.onChordHover?.(node);
        this.render();
      }
    });

    this.canvas.addEventListener('mousedown', (e) => {
      const node = this.hitTest(e.offsetX, e.offsetY);
      if (!node) {
        this.isPanning = true;
        this.lastMouse = { x: e.offsetX, y: e.offsetY };
        this.canvas.style.cursor = 'grabbing';
      }
    });

    this.canvas.addEventListener('mouseup', () => {
      this.isPanning = false;
      this.canvas.style.cursor = 'grab';
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.isPanning = false;
      this.hoveredNode = null;
      this.render();
    });

    this.canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const node = this.hitTest(e.offsetX, e.offsetY);
      if (node) this.onAddToProgression?.(node.bits);
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const mx = e.offsetX, my = e.offsetY;
      // Zoom toward cursor
      this.panX = mx - (mx - this.panX) * factor;
      this.panY = my - (my - this.panY) * factor;
      this.zoom *= factor;
      this.zoom = Math.max(0.3, Math.min(6, this.zoom));
      this.render();
    }, { passive: false });
  }

  private hitTest(mx: number, my: number): AnnularNode | null {
    const [wx, wy] = this.toWorld(mx, my);
    let best: AnnularNode | null = null;
    let bestDist = Infinity;
    for (const node of this.nodes) {
      const dx = wx - node.screenX;
      const dy = wy - node.screenY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const hitR = Math.max(node.dotR + 4, 8);
      if (dist < hitR && dist < bestDist) {
        bestDist = dist;
        best = node;
      }
    }
    return best;
  }

  render(): void {
    const { ctx, width, height } = this;
    const cx = width / 2;
    const cy = height / 2;
    const maxR = Math.min(cx, cy) - 20;

    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // Draw concentric rings
    for (let i = 1; i <= 4; i++) {
      const r = (i / 4) * maxR;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 0.5 / this.zoom;
      ctx.stroke();
    }

    // Draw radial lines
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(angle) * maxR, cy + Math.sin(angle) * maxR);
      ctx.strokeStyle = 'rgba(255,255,255,0.03)';
      ctx.lineWidth = 0.5 / this.zoom;
      ctx.stroke();
    }

    // Draw clock labels (pitch class sums)
    ctx.font = `${9 / this.zoom}px system-ui`;
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2 - Math.PI / 2;
      const r = maxR + 15 / this.zoom;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      ctx.fillText(String(i), x, y);
    }

    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.font = `${9 / this.zoom}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText('even', cx, cy - 8 / this.zoom);
    ctx.fillText('clustered', cx, cy + maxR + 20 / this.zoom);

    // Draw ideal position ghost dots (subtle, shows where nodes "want" to be)
    for (const node of this.nodes) {
      if (Math.abs(node.screenX - node.idealX) + Math.abs(node.screenY - node.idealY) > 2) {
        ctx.beginPath();
        ctx.arc(node.idealX, node.idealY, 1.5 / this.zoom, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.06)';
        ctx.fill();
      }
    }

    // Draw nodes
    for (const node of this.nodes) {
      this.drawNode(node);
    }

    // Active highlight
    if (this._activeNode) {
      this.drawHighlight(this._activeNode);
    }

    ctx.restore();

    // Tooltip (drawn in screen space, unaffected by zoom/pan)
    if (this.hoveredNode) {
      this.drawTooltip(this.hoveredNode);
    }

    // Stats (fixed overlay)
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`${this.nodes.length} chords  ·  scroll to zoom  ·  drag to pan`, 8, 8);
  }

  private drawNode(node: AnnularNode): void {
    const { ctx } = this;
    const isActive = node === this._activeNode;
    const isHovered = node === this.hoveredNode;
    const dimmed = (node as any)._inKey === false;

    const color = CARD_COLORS[node.cardinality] || '#888';
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    let alpha = node.isNamed ? 0.8 : 0.25;
    if (dimmed) alpha *= 0.15;
    const dotR = isHovered ? node.dotR * 1.3 : node.dotR;

    if (node.isNamed && node.consonance > 0.6 && !dimmed) {
      ctx.shadowColor = `rgba(${r}, ${g}, ${b}, 0.25)`;
      ctx.shadowBlur = 6;
    }

    ctx.beginPath();
    ctx.arc(node.screenX, node.screenY, dotR, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    ctx.fill();
    ctx.shadowBlur = 0;

    if (isHovered || isActive) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = (isActive ? 2 : 1) / this.zoom;
      ctx.stroke();
    }

    if (node.isNamed && !dimmed && (this.zoom > 1.2 || isHovered)) {
      ctx.fillStyle = isHovered ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.4)';
      ctx.font = `${Math.max(7, 8 / Math.sqrt(this.zoom))}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(node.label, node.screenX, node.screenY + dotR + 2 / this.zoom);
    }
  }

  private drawHighlight(node: AnnularNode): void {
    const { ctx } = this;
    ctx.beginPath();
    ctx.arc(node.screenX, node.screenY, node.dotR + 4 / this.zoom, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffd764';
    ctx.lineWidth = 2 / this.zoom;
    ctx.stroke();
  }

  private drawTooltip(node: AnnularNode): void {
    const { ctx } = this;

    // Convert node's world position to screen position for tooltip placement
    const sx = node.screenX * this.zoom + this.panX;
    const sy = node.screenY * this.zoom + this.panY;

    const pcs = bitsToArray(node.bits);
    const lines = [
      node.label,
      `Notes: ${pcs.map(pc => NOTE_NAMES[pc]).join(' ')}`,
      `Consonance: ${(node.consonance * 100).toFixed(0)}%`,
      `Evenness: ${(node.evennessVal * 100).toFixed(0)}%`,
      `PC Sum: ${node.pcSum}`,
    ];

    const tipX = sx + 15;
    const tipY = sy - 10;
    const lineH = 14;
    const tipW = 180;
    const tipH = lines.length * lineH + 8;

    const rx = Math.min(tipX, this.width - tipW - 10);
    const ry = Math.max(10, Math.min(tipY, this.height - tipH - 10));

    ctx.fillStyle = 'rgba(15, 15, 25, 0.92)';
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(rx, ry, tipW, tipH, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(lines[0], rx + 6, ry + 5);

    ctx.font = '9px system-ui';
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    for (let i = 1; i < lines.length; i++) {
      ctx.fillText(lines[i], rx + 6, ry + 5 + i * lineH);
    }
  }
}
