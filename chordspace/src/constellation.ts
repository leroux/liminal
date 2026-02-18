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
  isSecondDegree?: boolean;
  parentBits?: number;
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
  private secondDegree: ConstellationNode[] = [];
  private hoveredNode: ConstellationNode | null = null;
  private animProgress = 1; // 0→1, 1 = settled
  private animFrame = 0;

  private _profile: OvertoneProfile | null = null;
  private _pool: { bits: number; label: string }[] = [];
  private _cardinalityFilter: Set<number> | null = null;
  private _rootFilter: number | null = null;
  private _searchQuery: string = '';
  private _locked = false;
  private _showHelp = false;

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

  setLocked(v: boolean): void {
    this._locked = v;
    if (this.animProgress >= 1) this.render();
  }

  isLocked(): boolean { return this._locked; }

  toggleHelp(): void {
    this._showHelp = !this._showHelp;
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
    const maxNeighbors = Math.min(12, ranked.length);
    const ringR = Math.min(cx, cy) * 0.52;

    this.neighbors = [];
    for (let i = 0; i < maxNeighbors; i++) {
      const n = ranked[i];
      const angle = (i / maxNeighbors) * Math.PI * 2 - Math.PI / 2;
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
        screenX: cx, screenY: cy,
        targetX: tx, targetY: ty,
        dotR: 14 + n.commonTones * 2,
      });
    }

    // ── Second-degree: neighbors-of-neighbors ──
    const firstDegreeBits = new Set([bits, ...this.neighbors.map(n => n.bits)]);
    const secondDegreeSeen = new Set<number>();
    this.secondDegree = [];

    // Expand the top 5 first-degree parents, add up to 2 second-degree each
    for (const parent of this.neighbors.slice(0, 6)) {
      const n2Ranked = combinedRanking(parent.bits, this._pool, profile);
      let added = 0;
      for (const n2 of n2Ranked) {
        if (added >= 2) break;
        if (firstDegreeBits.has(n2.bits) || secondDegreeSeen.has(n2.bits)) continue;
        secondDegreeSeen.add(n2.bits);

        const parentAngle = Math.atan2(parent.targetY - cy, parent.targetX - cx);
        const offset = added === 0 ? -0.3 : 0.3;
        const angle2 = parentAngle + offset;
        const r2 = ringR * 1.75;
        const tx2 = Math.max(16, Math.min(this.width - 16, cx + Math.cos(angle2) * r2));
        const ty2 = Math.max(16, Math.min(this.height - 16, cy + Math.sin(angle2) * r2));

        const n2Midi = closeVoicing(n2.bits, 4);
        const n2Cons = consonanceScore(n2Midi, profile);

        this.secondDegree.push({
          bits: n2.bits,
          label: n2.label,
          cardinality: popcount(n2.bits),
          consonance: n2Cons,
          isCenter: false,
          isSecondDegree: true,
          parentBits: parent.bits,
          vld: n2.vld,
          commonTones: n2.commonTones,
          isPLR: n2.isPLR,
          score: n2.score,
          screenX: cx, screenY: cy,
          targetX: tx2, targetY: ty2,
          dotR: 9,
        });
        added++;
      }
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
    for (const n of this.secondDegree) {
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
      if (!node) return;

      if (node.isCenter) {
        // Always re-audition center
        const midi = closeVoicing(node.bits, 4);
        this.onChordClick?.(node.bits, midi);
        return;
      }

      // Neighbor or second-degree
      const midi = closeVoicing(node.bits, 4);
      this.onChordClick?.(node.bits, midi);

      if (!this._locked) {
        this.navigateTo(node.bits);
      }
    });

    this.canvas.addEventListener('mousemove', (e) => {
      const node = this.hitTest(e.offsetX, e.offsetY);
      if (node !== this.hoveredNode) {
        this.hoveredNode = node;
        this.canvas.style.cursor = node ? (this._locked && !node.isCenter ? 'copy' : 'pointer') : 'default';
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

    // Check first-degree neighbors
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
    if (best) return best;

    // Check second-degree
    for (const n of this.secondDegree) {
      const dx = mx - n.screenX;
      const dy = my - n.screenY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < n.dotR + 6 && dist < bestDist) {
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

    // Draw second-degree connections (behind everything)
    for (const n of this.secondDegree) {
      this.drawSecondDegreeConnection(n);
    }

    // Draw first-degree connections
    for (const n of this.neighbors) {
      this.drawConnection(n);
    }

    // Draw second-degree nodes
    for (const n of this.secondDegree) {
      this.drawSecondDegreeNode(n);
    }

    // Draw first-degree neighbors
    for (const n of this.neighbors) {
      this.drawNeighborNode(n);
    }

    // Draw center chord (on top)
    this.drawCenterNode(this.centerNode);

    // Hint text
    ctx.fillStyle = 'rgba(255,255,255,0.1)';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('click to navigate · right-click → add to prog · lock = preview only', 12, height - 8);

    // Help legend
    if (this._showHelp) {
      this.drawHelpLegend();
    }
  }

  private drawSecondDegreeConnection(node: ConstellationNode): void {
    const { ctx } = this;
    const parent = this.neighbors.find(n => n.bits === node.parentBits);
    if (!parent) return;

    ctx.beginPath();
    ctx.moveTo(parent.screenX, parent.screenY);
    ctx.lineTo(node.screenX, node.screenY);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  private drawSecondDegreeNode(node: ConstellationNode): void {
    const { ctx } = this;
    const isHovered = node === this.hoveredNode;
    const isSearchMatch = this._searchQuery.length > 0 &&
      node.label.toLowerCase().includes(this._searchQuery.toLowerCase());

    const color = CARD_COLORS[node.cardinality] || '#888';
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    const alpha = isHovered ? 0.65 : 0.28;
    const dotR = isHovered ? node.dotR * 1.3 : node.dotR;

    if (isHovered || isSearchMatch) {
      ctx.shadowColor = isSearchMatch ? 'rgba(255,215,100,0.4)' : `rgba(${r},${g},${b},0.3)`;
      ctx.shadowBlur = 10;
    }

    ctx.beginPath();
    ctx.arc(node.screenX, node.screenY, dotR, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.fill();

    if (isSearchMatch) {
      ctx.strokeStyle = '#ffd764';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else if (isHovered) {
      ctx.strokeStyle = `rgba(${r},${g},${b},0.8)`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.shadowBlur = 0;

    // Label only on hover or search match
    if (isHovered || isSearchMatch) {
      ctx.fillStyle = isSearchMatch ? '#ffd764' : '#fff';
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.label, node.screenX, node.screenY);

      // Small "2°" badge
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '8px system-ui';
      ctx.fillText('2°', node.screenX, node.screenY + dotR + 9);
    }
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

    if (this._locked) {
      // Dashed ring to indicate lock
      ctx.shadowBlur = 0;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (isSearchMatch) {
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

    // Lock badge
    if (this._locked) {
      ctx.font = '11px system-ui';
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.fillText('🔒', node.screenX, node.screenY - dotR - 24);
    }
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

    // Badge below: VLD / PLR / CT info
    ctx.font = '9px system-ui';
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    let badge = `VLD:${node.vld}`;
    if (node.isPLR) badge += ' PLR';
    if (node.commonTones > 0) badge += ` CT:${node.commonTones}`;
    ctx.fillText(badge, node.screenX, node.screenY + dotR + 10);

    // "preview only" hint when locked and hovered
    if (this._locked && isHovered) {
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.font = '9px system-ui';
      ctx.fillText('preview only', node.screenX, node.screenY - dotR - 8);
    }
  }

  private drawHelpLegend(): void {
    const { ctx, width, height } = this;

    const panelW = 210;
    const lines = [
      { text: 'HOW TO READ EXPLORE', bold: true, color: 'rgba(255,255,255,0.8)' },
      { text: '' },
      { text: 'Center chord', bold: true, color: '#ffd764' },
      { text: '  Current chord. Click to re-play.' },
      { text: '  Gold ring = normal  ╌╌╌ = locked' },
      { text: '' },
      { text: 'Inner ring = 1st degree', bold: true, color: '#adf' },
      { text: '  One step away by voice leading.' },
      { text: '  Dot size  =  common tones (CT)' },
      { text: '  Line width = closeness (low VLD)' },
      { text: '  ● on line  =  PLR transformation' },
      { text: '' },
      { text: 'Outer ring = 2nd degree', bold: true, color: 'rgba(180,180,255,0.6)' },
      { text: '  Two steps away. Dashed line.' },
      { text: '  Hover to reveal label.' },
      { text: '' },
      { text: 'Dot colors by note count:', bold: true, color: 'rgba(255,255,255,0.7)' },
      { text: '  ● 2-note', color: '#ff6b6b' },
      { text: '  ● 3-note (triads)', color: '#ffd764' },
      { text: '  ● 4-note (7ths)', color: '#4ecdc4' },
      { text: '  ● 5-note', color: '#45b7d1' },
      { text: '  ● 6-note', color: '#a78bfa' },
      { text: '' },
      { text: 'VLD = Voice Leading Distance' },
      { text: '  Sum of semitones each note moves.' },
      { text: 'CT = Common Tones' },
      { text: '  Notes shared with center chord.' },
      { text: 'PLR = Neo-Riemannian ops' },
      { text: '  P/L/R = parallel, leading-tone,' },
      { text: '  relative transforms.' },
      { text: '' },
      { text: 'Click neighbor → navigate there' },
      { text: 'Lock → preview without navigating' },
      { text: 'Right-click → add to progression' },
    ];

    const lineH = 14;
    const padX = 12;
    const padY = 10;
    const panelH = lines.length * lineH + padY * 2;

    const px = width - panelW - 12;
    const py = (height - panelH) / 2;

    // Panel background
    ctx.fillStyle = 'rgba(10,10,18,0.92)';
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    roundRect(ctx, px, py, panelW, panelH, 6);
    ctx.fill();
    ctx.stroke();

    // Text
    let y = py + padY + lineH * 0.7;
    for (const line of lines) {
      if (!line.text) { y += lineH; continue; }
      ctx.font = line.bold ? 'bold 10px system-ui' : '10px system-ui';
      ctx.fillStyle = (line.color as string | undefined) ?? 'rgba(255,255,255,0.5)';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(line.text, px + padX, y);
      y += lineH;
    }
  }

  getCenterBits(): number | null {
    return this.centerNode?.bits ?? null;
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}
