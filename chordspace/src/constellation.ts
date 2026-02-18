// ─── Constellation view ─────────────────────────────────────────────
// Navigation-first chord explorer. One chord at center, neighbors radiate out.
// Click a neighbor to navigate. The whole space unfolds from where you are.

import {
  NOTE_NAMES, bitsToArray, popcount, allChordNames, closeVoicing,
  CHORD_VOCABULARY,
} from './core.js';
import { consonanceScore, type OvertoneProfile } from './dissonance.js';
import { combinedRanking } from './navigation.js';

// ─── Types ──────────────────────────────────────────────────────────

export interface ConstellationNode {
  bits: number;
  label: string;
  cardinality: number;
  consonance: number;
  isCenter: boolean;
  // Ranking info (neighbors only)
  vld: number;
  commonTones: number;
  isPLR: boolean;
  score: number;
  // Layout
  screenX: number;
  screenY: number;
  dotR: number;
  targetX: number;
  targetY: number;
}

const CARD_COLORS: Record<number, string> = {
  2: '#ff6b6b',
  3: '#ffd764',
  4: '#4ecdc4',
  5: '#45b7d1',
  6: '#a78bfa',
  7: '#f472b6',
};

// ─── Build neighbor pool ────────────────────────────────────────────

function buildCandidatePool(
  _profile: OvertoneProfile,
  keyPCs: number[] | null,
  cardinalityFilter: Set<number> | null,
  rootFilter: number | null,
): { bits: number; label: string }[] {
  const seen = new Set<number>();
  const pool: { bits: number; label: string }[] = [];

  for (const entry of CHORD_VOCABULARY) {
    // Key filter: skip chords not in key
    if (keyPCs) {
      const pcs = bitsToArray(entry.bits);
      if (!pcs.every(pc => keyPCs.includes(pc))) continue;
    }

    // Cardinality filter
    if (cardinalityFilter && cardinalityFilter.size < 6) {
      if (!cardinalityFilter.has(popcount(entry.bits))) continue;
    }

    // Root filter
    if (rootFilter !== null) {
      if (entry.root !== rootFilter) continue;
    }

    if (seen.has(entry.bits)) continue;
    seen.add(entry.bits);

    pool.push({ bits: entry.bits, label: entry.symbol });
  }

  return pool;
}

// ─── Renderer ───────────────────────────────────────────────────────

export class ConstellationRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr: number;
  private width = 0;
  private height = 0;

  private centerNode: ConstellationNode | null = null;
  private neighbors: ConstellationNode[] = [];
  private hoveredNode: ConstellationNode | null = null;
  private animProgress = 1; // 0→1, 1 = settled
  private animFrame = 0;

  private _profile: OvertoneProfile | null = null;
  private _pool: { bits: number; label: string }[] = [];
  private _cardinalityFilter: Set<number> | null = null;
  private _rootFilter: number | null = null;
  private _searchQuery: string = '';

  onChordClick: ((bits: number, midi: number[]) => void) | null = null;
  onChordHover: ((node: ConstellationNode | null) => void) | null = null;
  onAddToProgression: ((bits: number) => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.dpr = window.devicePixelRatio || 1;
    this.setupEvents();
  }

  setProfile(
    profile: OvertoneProfile,
    keyPCs: number[] | null,
    cardinalityFilter?: Set<number> | null,
    rootFilter?: number | null,
    searchQuery?: string,
  ): void {
    this._profile = profile;
    this._cardinalityFilter = cardinalityFilter ?? null;
    this._rootFilter = rootFilter ?? null;
    this._searchQuery = searchQuery ?? '';
    this._pool = buildCandidatePool(profile, keyPCs, this._cardinalityFilter, this._rootFilter);
  }

  setSearch(query: string): void {
    this._searchQuery = query;
    if (this.animProgress >= 1) this.render();
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

  // Navigate to a chord — compute its neighbors and animate
  navigateTo(bits: number): void {
    if (!this._profile) return;
    const profile = this._profile;

    const names = allChordNames(bits);
    const label = names.length > 0 ? names[0].symbol : `[${bitsToArray(bits).map(pc => NOTE_NAMES[pc]).join(' ')}]`;
    const midi = closeVoicing(bits, 4);
    const cons = consonanceScore(midi, profile);
    const card = popcount(bits);

    const cx = this.width / 2;
    const cy = this.height / 2;

    this.centerNode = {
      bits, label, cardinality: card, consonance: cons,
      isCenter: true, vld: 0, commonTones: card, isPLR: false, score: 0,
      screenX: cx, screenY: cy, targetX: cx, targetY: cy,
      dotR: 28,
    };

    // Get ranked neighbors using combinedRanking
    const ranked = combinedRanking(bits, this._pool, profile);
    const maxNeighbors = Math.min(10, ranked.length);
    const ringR = Math.min(cx, cy) * 0.55;

    this.neighbors = [];
    for (let i = 0; i < maxNeighbors; i++) {
      const n = ranked[i];
      const angle = (i / maxNeighbors) * Math.PI * 2 - Math.PI / 2;
      // Closer neighbors (lower score) get placed nearer to center
      const distFactor = 0.6 + (i / maxNeighbors) * 0.4;
      const tx = cx + Math.cos(angle) * ringR * distFactor;
      const ty = cy + Math.sin(angle) * ringR * distFactor;

      const nMidi = closeVoicing(n.bits, 4);
      const nCons = consonanceScore(nMidi, profile);

      this.neighbors.push({
        bits: n.bits,
        label: n.label,
        cardinality: popcount(n.bits),
        consonance: nCons,
        isCenter: false,
        vld: n.vld,
        commonTones: n.commonTones,
        isPLR: n.isPLR,
        score: n.score,
        screenX: cx, screenY: cy, // start from center
        targetX: tx, targetY: ty,
        dotR: 14 + n.commonTones * 2,
      });
    }

    // Start animation
    this.animProgress = 0;
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    this.animate();
  }

  randomChord(): number {
    const pool = this._pool.length > 0 ? this._pool : CHORD_VOCABULARY.map(e => ({ bits: e.bits, label: e.symbol }));
    const idx = Math.floor(Math.random() * pool.length);
    return pool[idx].bits;
  }

  private animate(): void {
    this.animProgress = Math.min(1, this.animProgress + 0.06);
    const t = easeOutCubic(this.animProgress);

    const cx = this.width / 2;
    const cy = this.height / 2;

    for (const n of this.neighbors) {
      n.screenX = cx + (n.targetX - cx) * t;
      n.screenY = cy + (n.targetY - cy) * t;
    }

    this.render();

    if (this.animProgress < 1) {
      this.animFrame = requestAnimationFrame(() => this.animate());
    }
  }

  // ─── Events ──────────────────────────────────────────────────────

  private setupEvents(): void {
    this.canvas.addEventListener('click', (e) => {
      const node = this.hitTest(e.offsetX, e.offsetY);
      if (node && !node.isCenter) {
        // Navigate to clicked neighbor
        const midi = closeVoicing(node.bits, 4);
        this.onChordClick?.(node.bits, midi);
        this.navigateTo(node.bits);
      } else if (node && node.isCenter) {
        // Re-audition center chord
        const midi = closeVoicing(node.bits, 4);
        this.onChordClick?.(node.bits, midi);
      }
    });

    this.canvas.addEventListener('mousemove', (e) => {
      const node = this.hitTest(e.offsetX, e.offsetY);
      if (node !== this.hoveredNode) {
        this.hoveredNode = node;
        this.canvas.style.cursor = node ? 'pointer' : 'default';
        this.onChordHover?.(node);
        if (this.animProgress >= 1) this.render();
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.hoveredNode = null;
      this.onChordHover?.(null);
      if (this.animProgress >= 1) this.render();
    });

    // Right-click to add to progression
    this.canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const node = this.hitTest(e.offsetX, e.offsetY);
      if (node) {
        this.onAddToProgression?.(node.bits);
      }
    });
  }

  private hitTest(mx: number, my: number): ConstellationNode | null {
    // Check center first
    if (this.centerNode) {
      const dx = mx - this.centerNode.screenX;
      const dy = my - this.centerNode.screenY;
      if (Math.sqrt(dx * dx + dy * dy) < this.centerNode.dotR + 8) {
        return this.centerNode;
      }
    }

    // Check neighbors
    let best: ConstellationNode | null = null;
    let bestDist = Infinity;
    for (const n of this.neighbors) {
      const dx = mx - n.screenX;
      const dy = my - n.screenY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < n.dotR + 8 && dist < bestDist) {
        bestDist = dist;
        best = n;
      }
    }
    return best;
  }

  // ─── Rendering ───────────────────────────────────────────────────

  render(): void {
    const { ctx, width, height } = this;
    if (width === 0 || height === 0) return;

    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);

    if (!this.centerNode) {
      ctx.fillStyle = 'rgba(255,255,255,0.2)';
      ctx.font = '14px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Click Random or select a chord to begin', width / 2, height / 2);
      return;
    }

    // Draw connection lines from center to neighbors
    for (const n of this.neighbors) {
      this.drawConnection(n);
    }

    // Draw neighbors
    for (const n of this.neighbors) {
      this.drawNeighborNode(n);
    }

    // Draw center chord (on top)
    this.drawCenterNode(this.centerNode);

    // Hint text
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('click neighbor to navigate · right-click to add to progression', 12, height - 8);
  }

  private drawConnection(neighbor: ConstellationNode): void {
    const { ctx } = this;
    const center = this.centerNode!;

    // Line thickness & opacity based on VLD (lower = stronger connection)
    const maxVLD = 8;
    const strength = 1 - Math.min(neighbor.vld, maxVLD) / maxVLD;
    const lineWidth = 1 + strength * 3;
    const alpha = 0.08 + strength * 0.2;

    const color = CARD_COLORS[neighbor.cardinality] || '#888';
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    ctx.beginPath();
    ctx.moveTo(center.screenX, center.screenY);
    ctx.lineTo(neighbor.screenX, neighbor.screenY);
    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    ctx.lineWidth = lineWidth;
    ctx.stroke();

    // PLR connections get a special indicator
    if (neighbor.isPLR) {
      const mx = (center.screenX + neighbor.screenX) / 2;
      const my = (center.screenY + neighbor.screenY) / 2;
      ctx.beginPath();
      ctx.arc(mx, my, 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 215, 100, 0.4)`;
      ctx.fill();
    }
  }

  private drawCenterNode(node: ConstellationNode): void {
    const { ctx } = this;
    const isHovered = node === this.hoveredNode;
    const isSearchMatch = this._searchQuery.length > 0 &&
      node.label.toLowerCase().includes(this._searchQuery.toLowerCase());

    const color = CARD_COLORS[node.cardinality] || '#888';
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    // Glow
    ctx.shadowColor = `rgba(${r}, ${g}, ${b}, 0.4)`;
    ctx.shadowBlur = 20;

    // Circle
    const dotR = isHovered ? node.dotR * 1.1 : node.dotR;
    ctx.beginPath();
    ctx.arc(node.screenX, node.screenY, dotR, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
    ctx.fill();
    // Search match: double gold ring; normal: single gold ring
    if (isSearchMatch) {
      ctx.strokeStyle = '#ffd764';
      ctx.lineWidth = 5;
      ctx.stroke();
    } else {
      ctx.strokeStyle = '#ffd764';
      ctx.lineWidth = 3;
      ctx.stroke();
    }
    ctx.shadowBlur = 0;

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 16px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(node.label, node.screenX, node.screenY);

    // Notes below
    const pcs = bitsToArray(node.bits);
    ctx.font = '10px system-ui';
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText(pcs.map(pc => NOTE_NAMES[pc]).join(' '), node.screenX, node.screenY + dotR + 14);

    // Consonance above
    ctx.fillText(`${(node.consonance * 100).toFixed(0)}% consonant`, node.screenX, node.screenY - dotR - 10);
  }

  private drawNeighborNode(node: ConstellationNode): void {
    const { ctx } = this;
    const isHovered = node === this.hoveredNode;
    const isSearchMatch = this._searchQuery.length > 0 &&
      node.label.toLowerCase().includes(this._searchQuery.toLowerCase());

    const color = CARD_COLORS[node.cardinality] || '#888';
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    const alpha = 0.7 + (isHovered ? 0.3 : 0);
    const dotR = isHovered ? node.dotR * 1.2 : node.dotR;

    // Glow for hovered or search match
    if (isHovered || isSearchMatch) {
      ctx.shadowColor = isSearchMatch ? 'rgba(255, 215, 100, 0.5)' : `rgba(${r}, ${g}, ${b}, 0.3)`;
      ctx.shadowBlur = isSearchMatch ? 16 : 12;
    }

    // Circle
    ctx.beginPath();
    ctx.arc(node.screenX, node.screenY, dotR, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    ctx.fill();

    if (isSearchMatch) {
      ctx.strokeStyle = '#ffd764';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (isHovered) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.shadowBlur = 0;

    // Label
    ctx.fillStyle = isSearchMatch ? '#ffd764' : (isHovered ? '#fff' : 'rgba(255,255,255,0.8)');
    ctx.font = `${(isHovered || isSearchMatch) ? 'bold ' : ''}12px system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(node.label, node.screenX, node.screenY);

    // VLD badge below
    ctx.font = '9px system-ui';
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    let badge = `VLD:${node.vld}`;
    if (node.isPLR) badge += ' PLR';
    if (node.commonTones > 0) badge += ` CT:${node.commonTones}`;
    ctx.fillText(badge, node.screenX, node.screenY + dotR + 10);
  }

  getCenterBits(): number | null {
    return this.centerNode?.bits ?? null;
  }
}

// ─── Easing ─────────────────────────────────────────────────────────

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}
