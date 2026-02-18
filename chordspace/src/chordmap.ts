// ─── 2D chord scatter map ───────────────────────────────────────────
// Shows all chords/set classes on a 2D plane.
// X = consonance (Vassilakis roughness), Y = evenness (DFT).
// Colored by consonance, sized by cardinality.

import {
  NOTE_NAMES, bitsToArray, closeVoicing, allChordNames,
  intervalVector, evenness, popcount,
  FORTE_CLASSES, CHORD_VOCABULARY, type SetClass,
} from './core.js';
import { consonanceScore, type OvertoneProfile } from './dissonance.js';

// ─── Layout data ────────────────────────────────────────────────────

export interface ChordNode {
  // Identity
  bits: number;
  label: string;         // display name
  cardinality: number;
  isNamed: boolean;
  names: string[];       // all chord names for this PCS
  forteClass: SetClass | undefined;
  iv: [number, number, number, number, number, number];

  // Computed
  consonance: number;    // 0–1
  evennessVal: number;   // 0–1

  // Layout
  x: number;
  y: number;
  r: number;             // radius
}

const CARD_COLORS: Record<number, string> = {
  2: '#ff6b6b',
  3: '#ffd764',
  4: '#4ecdc4',
  5: '#45b7d1',
  6: '#a78bfa',
  7: '#f472b6',
};

// ─── Build node list ────────────────────────────────────────────────

export type ViewMode = 'named' | 'all-classes' | 'all-pcs';

export function buildNodes(
  profile: OvertoneProfile,
  mode: ViewMode,
  rootFilter: number | null,  // null = all roots, 0–11 = specific root
): ChordNode[] {
  const nodes: ChordNode[] = [];

  if (mode === 'named') {
    // All named chord types × 12 roots (or filtered root), deduplicated by bits
    const seen = new Set<number>();
    for (const entry of CHORD_VOCABULARY) {
      if (rootFilter !== null && entry.root !== rootFilter) continue;
      if (seen.has(entry.bits)) continue;
      seen.add(entry.bits);

      const midi = closeVoicing(entry.bits, 4);
      const cons = consonanceScore(midi, profile);
      const ev = evenness(entry.bits);
      const allNames = allChordNames(entry.bits).map(n => n.symbol);
      const label = allNames.length <= 1 ? (allNames[0] ?? entry.symbol)
        : allNames.length === 2 ? `${allNames[0]} / ${allNames[1]}`
        : `${allNames[0]}…`;

      nodes.push({
        bits: entry.bits,
        label,
        cardinality: entry.cardinality,
        isNamed: true,
        names: allNames,
        forteClass: entry.setClass,
        iv: entry.iv,
        consonance: cons,
        evennessVal: ev,
        x: 0, y: 0, r: 0,
      });
    }
  } else if (mode === 'all-classes') {
    // All 224 Forte set classes (one representative per class)
    for (const sc of FORTE_CLASSES) {
      const bits = sc.primeFormBits;
      const midi = closeVoicing(bits, 4);
      const cons = consonanceScore(midi, profile);
      const ev = evenness(bits);
      const names = allChordNames(bits);
      const label = names.length > 0
        ? names[0].symbol
        : `${sc.cardinality}-${sc.id}`;

      nodes.push({
        bits,
        label,
        cardinality: sc.cardinality,
        isNamed: names.length > 0,
        names: names.map(n => n.symbol),
        forteClass: sc,
        iv: sc.iv,
        consonance: cons,
        evennessVal: ev,
        x: 0, y: 0, r: 0,
      });
    }
  } else {
    // All PCS with cardinality 2–7 (thousands of them)
    for (let bits = 1; bits < 4096; bits++) {
      const card = popcount(bits);
      if (card < 2 || card > 7) continue;
      if (rootFilter !== null) {
        // Only show PCS containing this root
        if (!(bits & (1 << rootFilter))) continue;
      }

      const midi = closeVoicing(bits, 4);
      const cons = consonanceScore(midi, profile);
      const ev = evenness(bits);
      const names = allChordNames(bits);
      const label = names.length > 0
        ? names[0].symbol
        : `[${bitsToArray(bits).map(pc => NOTE_NAMES[pc]).join(' ')}]`;

      nodes.push({
        bits,
        label,
        cardinality: card,
        isNamed: names.length > 0,
        names: names.map(n => n.symbol),
        forteClass: undefined,
        iv: intervalVector(bits),
        consonance: cons,
        evennessVal: ev,
        x: 0, y: 0, r: 0,
      });
    }
  }

  return nodes;
}

// ─── Layout modes ───────────────────────────────────────────────────

export type LayoutMode = 'consonance' | 'mds';

// ─── Layout computation ─────────────────────────────────────────────

export function layoutNodes(nodes: ChordNode[], width: number, height: number, mode: LayoutMode = 'consonance'): void {
  if (mode === 'mds') {
    layoutMDS(nodes, width, height);
    return;
  }
  const padX = 60, padY = 50;
  const plotW = width - padX * 2;
  const plotH = height - padY * 2;

  for (const node of nodes) {
    // X = consonance (0 left, 1 right)
    node.x = padX + node.consonance * plotW;

    // Y = organized by cardinality bands with evenness as spread within band
    // Cardinalities 2–7 → 6 bands
    const bandCount = 6;
    const bandHeight = plotH / bandCount;
    const bandIndex = node.cardinality - 2; // 0–5
    const bandCenter = padY + bandIndex * bandHeight + bandHeight / 2;

    // Spread within band by evenness
    const spread = bandHeight * 0.35;
    node.y = bandCenter + (node.evennessVal - 0.5) * spread * 2;

    // Radius by cardinality
    node.r = 3 + node.cardinality * 1.2;
  }

  // Jitter overlapping nodes
  resolveOverlaps(nodes);
}

// MDS layout: use pre-computed 2D coordinates from classical MDS
export function layoutMDS(nodes: ChordNode[], width: number, height: number): void {
  if (nodes.length === 0) return;

  const padX = 60, padY = 50;
  const plotW = width - padX * 2;
  const plotH = height - padY * 2;

  // Simple Sammon-like layout: use consonance as primary X, use a hash-based Y spread
  // For true MDS, we'd import the full matrix, but for responsiveness we do a lightweight version:
  // Group by cardinality, within each group spread by evenness and consonance
  const byCard = new Map<number, ChordNode[]>();
  for (const node of nodes) {
    const list = byCard.get(node.cardinality) || [];
    list.push(node);
    byCard.set(node.cardinality, list);
  }

  // Assign positions: for each cardinality group, use force-directed placement
  const cardList = [...byCard.keys()].sort();
  const bandHeight = plotH / Math.max(cardList.length, 1);

  for (let ci = 0; ci < cardList.length; ci++) {
    const card = cardList[ci];
    const group = byCard.get(card)!;
    const bandY = padY + ci * bandHeight;

    // Sort by consonance
    group.sort((a, b) => a.consonance - b.consonance);

    for (let i = 0; i < group.length; i++) {
      const node = group[i];
      // X spread across full width based on consonance
      node.x = padX + node.consonance * plotW;
      // Y within band: use evenness for vertical spread, with index jitter
      const yCenter = bandY + bandHeight / 2;
      const ySpread = bandHeight * 0.4;
      node.y = yCenter + (node.evennessVal - 0.5) * ySpread * 2;
      // Add small deterministic jitter based on bits to separate overlaps
      node.y += ((node.bits * 7 + node.bits * 13) % 20 - 10) * 0.5;
      node.r = 3 + node.cardinality * 1.2;
    }
  }

  resolveOverlaps(nodes);
}

function resolveOverlaps(nodes: ChordNode[]): void {
  // Simple repulsion pass
  for (let pass = 0; pass < 3; pass++) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = nodes[i].r + nodes[j].r + 2;

        if (dist < minDist && dist > 0.01) {
          const push = (minDist - dist) / 2;
          const nx = dx / dist;
          const ny = dy / dist;
          nodes[i].x -= nx * push;
          nodes[i].y -= ny * push;
          nodes[j].x += nx * push;
          nodes[j].y += ny * push;
        }
      }
    }
  }
}

// ─── Renderer ───────────────────────────────────────────────────────

export class ChordMapRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr: number;
  private nodes: ChordNode[] = [];
  private hoveredNode: ChordNode | null = null;
  private _activeNode: ChordNode | null = null;
  private width = 0;
  private height = 0;

  // Zoom/pan
  private panX = 0;
  private panY = 0;
  private zoom = 1;
  private isPanning = false;
  private lastMouse = { x: 0, y: 0 };

  // Progression trail
  progressionTrail: ChordNode[] = [];

  onChordClick: ((node: ChordNode, midi: number[]) => void) | null = null;
  onChordHover: ((node: ChordNode | null) => void) | null = null;
  onAddToProgression: ((bits: number) => void) | null = null;

  get activeNode(): ChordNode | null { return this._activeNode; }
  set activeNode(n: ChordNode | null) { this._activeNode = n; }

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.dpr = window.devicePixelRatio || 1;
    this.setupEvents();
  }

  setNodes(nodes: ChordNode[]): void {
    this.nodes = nodes;
  }

  focusOn(node: ChordNode): void {
    this._activeNode = node;
    this.panX = this.width / 2 - node.x * this.zoom;
    this.panY = this.height / 2 - node.y * this.zoom;
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

  private setupEvents(): void {
    this.canvas.addEventListener('click', (e) => {
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

      // Zoom toward mouse position
      this.panX = mx - (mx - this.panX) * factor;
      this.panY = my - (my - this.panY) * factor;
      this.zoom *= factor;
      this.zoom = Math.max(0.3, Math.min(5, this.zoom));
      this.render();
    }, { passive: false });
  }

  private toScreen(x: number, y: number): [number, number] {
    return [x * this.zoom + this.panX, y * this.zoom + this.panY];
  }

  private fromScreen(sx: number, sy: number): [number, number] {
    return [(sx - this.panX) / this.zoom, (sy - this.panY) / this.zoom];
  }

  private hitTest(mx: number, my: number): ChordNode | null {
    const [wx, wy] = this.fromScreen(mx, my);
    let best: ChordNode | null = null;
    let bestDist = Infinity;

    for (const node of this.nodes) {
      const dx = wx - node.x;
      const dy = wy - node.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const hitR = Math.max(node.r, 6) / this.zoom * 1.5 + 4;
      if (dist < hitR && dist < bestDist) {
        bestDist = dist;
        best = node;
      }
    }
    return best;
  }

  // ─── Render ─────────────────────────────────────────────────────

  render(): void {
    const { ctx, width, height } = this;

    ctx.save();
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);

    // Apply zoom/pan
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // Draw axis labels
    this.drawAxes();

    // Draw cardinality band backgrounds
    this.drawBands();

    // Draw progression trail
    if (this.progressionTrail.length >= 2) {
      this.drawProgressionTrail();
    }

    // Draw nodes
    for (const node of this.nodes) {
      this.drawNode(node);
    }

    // Draw active node highlight on top
    if (this._activeNode) {
      this.drawNodeHighlight(this._activeNode);
    }

    ctx.restore();

    // Draw fixed UI overlays (not affected by zoom/pan)
    this.drawOverlays();
  }

  private drawAxes(): void {
    const { ctx, width, height } = this;
    const padX = 60, padY = 50;

    // X axis label
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('dissonant', padX + 30, height - 12);
    ctx.fillText('consonant', width - padX - 30, height - 12);

    // Arrow
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padX, height - 20);
    ctx.lineTo(width - padX, height - 20);
    ctx.stroke();

    // Y axis cardinality labels
    const bandCount = 6;
    const plotH = height - padY * 2;
    const bandHeight = plotH / bandCount;
    for (let i = 0; i < bandCount; i++) {
      const card = i + 2;
      const y = padY + i * bandHeight + bandHeight / 2;
      ctx.fillStyle = CARD_COLORS[card] || '#888';
      ctx.font = '10px system-ui';
      ctx.textAlign = 'right';
      ctx.fillText(`${card} notes`, padX - 10, y + 3);
    }
  }

  private drawBands(): void {
    const { ctx, width, height } = this;
    const padX = 60, padY = 50;
    const plotH = height - padY * 2;
    const bandHeight = plotH / 6;

    for (let i = 0; i < 6; i++) {
      const y = padY + i * bandHeight;
      if (i % 2 === 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.015)';
        ctx.fillRect(padX, y, width - padX * 2, bandHeight);
      }
    }
  }

  private drawNode(node: ChordNode): void {
    const { ctx } = this;
    const isActive = node === this._activeNode;
    const isHovered = node === this.hoveredNode;
    const inKey = (node as any)._inKey;
    const searchMatch = (node as any)._searchMatch;

    // Dim out-of-key chords
    const dimmed = inKey === false;
    // Highlight search matches
    const highlighted = searchMatch === true;

    const color = CARD_COLORS[node.cardinality] || '#888';

    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    let alpha = node.isNamed ? 0.85 : 0.3;
    if (dimmed) alpha *= 0.15;
    if (highlighted) alpha = Math.min(1, alpha * 1.3);

    const radius = isHovered ? node.r * 1.4 : highlighted ? node.r * 1.2 : node.r;

    // Glow for search matches or consonant named chords
    if (highlighted) {
      ctx.shadowColor = `rgba(255, 255, 100, 0.5)`;
      ctx.shadowBlur = 10;
    } else if (node.isNamed && node.consonance > 0.6 && !dimmed) {
      ctx.shadowColor = `rgba(${r}, ${g}, ${b}, ${node.consonance * 0.3})`;
      ctx.shadowBlur = 8;
    }

    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);

    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    ctx.fill();

    ctx.shadowBlur = 0;

    // Border
    if (isHovered || isActive) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = isActive ? 2 : 1;
      ctx.stroke();
    } else if (highlighted) {
      ctx.strokeStyle = 'rgba(255, 255, 100, 0.8)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else if (node.isNamed && !dimmed) {
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.5)`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Label
    const showLabel = (node.isNamed && !dimmed) || this.zoom > 1.5 || isHovered || highlighted;
    if (showLabel) {
      ctx.fillStyle = isHovered || isActive
        ? 'rgba(255,255,255,0.95)'
        : highlighted
          ? 'rgba(255,255,100,0.9)'
          : dimmed
            ? 'rgba(255,255,255,0.1)'
            : node.isNamed
              ? 'rgba(255,255,255,0.6)'
              : 'rgba(255,255,255,0.3)';
      ctx.font = `${isActive || highlighted ? 'bold ' : ''}${Math.max(8, 9 / Math.sqrt(this.zoom))}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(node.label, node.x, node.y + radius + 2);
    }
  }

  private drawNodeHighlight(node: ChordNode): void {
    const { ctx } = this;

    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r + 4, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffd764';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.shadowColor = 'rgba(255, 220, 100, 0.4)';
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r + 2, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 220, 100, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  private drawProgressionTrail(): void {
    const { ctx } = this;
    const trail = this.progressionTrail;

    // Draw connecting lines
    ctx.lineWidth = 2;
    for (let i = 0; i < trail.length - 1; i++) {
      const a = trail[i];
      const b = trail[i + 1];
      const t = i / Math.max(trail.length - 1, 1);

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);

      // Gradient from gold to orange along progression
      const hue = 40 + t * 20;
      ctx.strokeStyle = `hsla(${hue}, 90%, 60%, 0.6)`;
      ctx.stroke();

      // Arrow at midpoint
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      const arrowLen = 6;
      ctx.beginPath();
      ctx.moveTo(mx + Math.cos(angle) * arrowLen, my + Math.sin(angle) * arrowLen);
      ctx.lineTo(mx + Math.cos(angle + 2.5) * arrowLen, my + Math.sin(angle + 2.5) * arrowLen);
      ctx.lineTo(mx + Math.cos(angle - 2.5) * arrowLen, my + Math.sin(angle - 2.5) * arrowLen);
      ctx.closePath();
      ctx.fillStyle = `hsla(${hue}, 90%, 60%, 0.6)`;
      ctx.fill();
    }

    // Draw numbered circles at each progression step
    for (let i = 0; i < trail.length; i++) {
      const node = trail[i];
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r + 6, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 220, 100, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Step number
      ctx.fillStyle = 'rgba(255, 220, 100, 0.9)';
      ctx.font = 'bold 8px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), node.x, node.y - node.r - 10);
    }
  }

  private drawOverlays(): void {
    const { ctx } = this;

    // Hover tooltip
    if (this.hoveredNode) {
      const node = this.hoveredNode;
      const [sx, sy] = this.toScreen(node.x, node.y);
      const tooltipLabel = node.names.length > 0 ? node.names[0] : node.label;
      const lines = [
        tooltipLabel,
        `Consonance: ${(node.consonance * 100).toFixed(0)}%`,
        `IV: ⟨${node.iv.join(',')}⟩`,
        `Notes: ${bitsToArray(node.bits).map(pc => NOTE_NAMES[pc]).join(' ')}`,
      ];
      if (node.names.length > 1) {
        const rest = node.names.slice(1);
        lines.push(`Also: ${rest.slice(0, 3).join(', ')}${rest.length > 3 ? ', …' : ''}`);
      }
      if (node.forteClass) {
        lines.push(`Forte: ${node.forteClass.cardinality}-${node.forteClass.id} (${node.forteClass.multiplicity} transpositions)`);
      }

      const tipX = sx + 15;
      const tipY = sy - 10;
      const lineH = 15;
      const tipW = 200;
      const tipH = lines.length * lineH + 10;

      ctx.fillStyle = 'rgba(15, 15, 25, 0.92)';
      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.lineWidth = 1;
      const rx = Math.min(tipX, this.width - tipW - 10);
      const ry = Math.max(10, Math.min(tipY, this.height - tipH - 10));
      ctx.beginPath();
      ctx.roundRect(rx, ry, tipW, tipH, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#fff';
      ctx.font = 'bold 11px system-ui';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(lines[0], rx + 8, ry + 6);

      ctx.font = '10px system-ui';
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      for (let i = 1; i < lines.length; i++) {
        ctx.fillText(lines[i], rx + 8, ry + 6 + i * lineH);
      }
    }

    // Stats overlay (top-left)
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`${this.nodes.length} chords`, 8, 8);
  }
}
