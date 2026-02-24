// ─── ChordSpace app ─────────────────────────────────────────────────
// Navigation-first chord explorer with constellation, map, tonnetz, annular views.

import {
  NOTE_NAMES, bitsToArray, pcsToBits, popcount,
  closeVoicing, drop2Voicing, shellVoicing, spreadVoicing,
  rootlessAVoicing, rootlessBVoicing, quartalVoicing,
  applyRegisterConstraints, applyDoublingRules,
  catalogStats, allChordNames, CHORD_VOCABULARY,
} from './core.js';
import { ChordMapRenderer, buildNodes, layoutNodes, type ChordNode, type ViewMode } from './chordmap.js';
import { AnnularRenderer, buildAnnularNodes, layoutAnnular } from './annular.js';
import { TonnetzRenderer } from './tonnetz.js';
import { ConstellationRenderer } from './constellation.js';
import { MultiSynth, type SynthMethod } from './synth.js';
import { TIMBRES, DEFAULT_TIMBRE, consonanceScore, precomputeTriadDissonance } from './dissonance.js';
import { findNearestByVLD, vldBits } from './navigation.js';

// ─── State ──────────────────────────────────────────────────────────

let currentTimbre = DEFAULT_TIMBRE;
let currentViewMode: ViewMode = 'named';
let currentRootFilter: number | null = null;
let currentVoicing: string = 'close';
let currentInversion = 0;
let cardinalityFilter: Set<number> = new Set([2, 3, 4, 5, 6, 7]);
let currentKey: number | null = null;
let currentScale: number[] | null = null;
let currentView: 'constellation' | 'map' | 'tonnetz' | 'annular' = 'constellation';
let currentSynthMethod: SynthMethod = 'additive';
let registerConstraints = false;
let searchQuery = '';

let constellation: ConstellationRenderer;
let chordMap: ChordMapRenderer;
let tonnetz: TonnetzRenderer;
let annular: AnnularRenderer;
let synth: MultiSynth;
let analyserNode: AnalyserNode | null = null;
let scopeCanvas: HTMLCanvasElement | null = null;
let scopeCtx: CanvasRenderingContext2D | null = null;
let scopeAnimFrame = 0;
let allNodes: ChordNode[] = [];
let filteredNodes: ChordNode[] = [];

// ─── Progression ────────────────────────────────────────────────────

interface ProgChord {
  bits: number;
  label: string;
  inversion: number;  // 0=root, 1=1st, 2=2nd, 3=3rd
  duration: number;   // beats (0.5 = 8th, 1 = quarter, 2 = half, 4 = whole)
}

let progression: ProgChord[] = [];
let progTempo = 100;
let progLoop = false;
let progPlaying = false;
let progInterval: number | null = null;

// Arp
let progArpMode: 'off' | 'up' | 'down' | 'updown' | 'random' = 'off';
let progArpRate = 0.5;   // beats per arp note
let progArpGate = 0.75;  // fraction of arp interval
let arpSubTimeout: number | null = null;

// DnD
let dragSrcIdx = -1;

// ─── Scale definitions ──────────────────────────────────────────────

const SCALES: Record<string, number[]> = {
  'major':     [0, 2, 4, 5, 7, 9, 11],
  'minor':     [0, 2, 3, 5, 7, 8, 10],
  'dorian':    [0, 2, 3, 5, 7, 9, 10],
  'mixolydian':[0, 2, 4, 5, 7, 9, 10],
  'lydian':    [0, 2, 4, 6, 7, 9, 11],
  'phrygian':  [0, 1, 3, 5, 7, 8, 10],
  'harmmin':   [0, 2, 3, 5, 7, 8, 11],
  'melmin':    [0, 2, 3, 5, 7, 9, 11],
  'pentatonic':[0, 2, 4, 7, 9],
  'blues':     [0, 3, 5, 6, 7, 10],
  'wholetone': [0, 2, 4, 6, 8, 10],
  'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

function getScalePCs(root: number, scaleIntervals: number[]): number[] {
  return scaleIntervals.map(i => (i + root) % 12);
}

function chordInKey(bits: number, keyPCs: number[]): boolean {
  const pcs = bitsToArray(bits);
  return pcs.every(pc => keyPCs.includes(pc));
}

// ─── Voicing helper ─────────────────────────────────────────────────

function voiceChord(bits: number): number[] {
  let midi: number[];
  switch (currentVoicing) {
    case 'close': midi = closeVoicing(bits, 4, currentInversion); break;
    case 'drop2': midi = drop2Voicing(bits, 4, currentInversion); break;
    case 'shell': midi = shellVoicing(bits, 4); break;
    case 'spread': midi = spreadVoicing(bits, 3); break;
    case 'rootlessA': midi = rootlessAVoicing(bits, 4); break;
    case 'rootlessB': midi = rootlessBVoicing(bits, 3); break;
    case 'quartal': midi = quartalVoicing(bits, 4); break;
    default: midi = closeVoicing(bits, 4, currentInversion);
  }
  if (registerConstraints) {
    midi = applyRegisterConstraints(midi);
    midi = applyDoublingRules(midi, bits);
  }
  return midi;
}

// Voice a progression chord using its own inversion setting
function voiceProgChord(p: ProgChord): number[] {
  const saved = currentInversion;
  currentInversion = p.inversion;
  const midi = voiceChord(p.bits);
  currentInversion = saved;
  return midi;
}

// ─── Chord label helper ─────────────────────────────────────────────

function chordLabel(bits: number): string {
  const names = allChordNames(bits);
  if (names.length > 0) return names[0].symbol;
  return `[${bitsToArray(bits).map(pc => NOTE_NAMES[pc]).join(' ')}]`;
}

// ─── Recompute filtered nodes (shared by map/annular views) ─────────

function recomputeNodes(): void {
  const profile = TIMBRES[currentTimbre];
  allNodes = buildNodes(profile, currentViewMode, currentRootFilter);

  filteredNodes = allNodes.filter(n => cardinalityFilter.has(n.cardinality));

  // Search highlight
  if (searchQuery.length > 0) {
    const q = searchQuery.toLowerCase();
    const intervalMatch = q.match(/^[\d\s,]+$/);
    let intervalPattern: number[] | null = null;
    if (intervalMatch) {
      intervalPattern = q.split(/[\s,]+/).filter(s => s.length > 0).map(Number);
      if (intervalPattern.some(isNaN)) intervalPattern = null;
    }

    for (const node of filteredNodes) {
      let match = node.label.toLowerCase().includes(q) ||
        node.names.some(n => n.toLowerCase().includes(q)) ||
        (node.forteClass && `${node.forteClass.cardinality}-${node.forteClass.id}`.includes(q));

      if (!match && intervalPattern && intervalPattern.length >= 2) {
        const nodePcs = bitsToArray(node.bits);
        match = intervalPattern.every(iv => nodePcs.includes(iv % 12));
      }

      if (!match && (q.includes(' ') || q.includes(','))) {
        const noteTokens = q.split(/[\s,]+/).filter(s => s.length > 0);
        const NOTE_MAP: Record<string, number> = {
          'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3,
          'e': 4, 'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8,
          'a': 9, 'a#': 10, 'bb': 10, 'b': 11,
        };
        const notePcs = noteTokens.map(t => NOTE_MAP[t]).filter(n => n !== undefined);
        if (notePcs.length >= 2) {
          const nodePcs = bitsToArray(node.bits);
          match = notePcs.every(pc => nodePcs.includes(pc));
        }
      }

      (node as any)._searchMatch = match;
    }
  } else {
    for (const node of filteredNodes) {
      (node as any)._searchMatch = undefined;
    }
  }

  // Key filter
  if (currentScale) {
    for (const node of filteredNodes) {
      (node as any)._inKey = chordInKey(node.bits, currentScale);
    }
  } else {
    for (const node of filteredNodes) {
      (node as any)._inKey = undefined;
    }
  }
}

// ─── Rebuild current view ──────────────────────────────────────────

function rebuild(): void {
  if (currentView === 'constellation') {
    // Constellation manages its own data; pass all active filters
    const profile = TIMBRES[currentTimbre];
    constellation.setProfile(profile, currentScale, cardinalityFilter, currentRootFilter, searchQuery);
    const wrap = document.getElementById('canvas-wrap')!;
    const rect = wrap.getBoundingClientRect();
    constellation.resize(rect.width, rect.height);
    // Re-navigate to current center if one exists, to pick up key filter changes
    const centerBits = constellation.getCenterBits();
    if (centerBits !== null) {
      constellation.navigateTo(centerBits);
    } else {
      constellation.render();
    }
  } else {
    recomputeNodes();
    if (currentView === 'map') {
      renderMap();
    } else if (currentView === 'tonnetz') {
      const dissMap = precomputeTriadDissonance(TIMBRES[currentTimbre], 4);
      tonnetz.state.dissonanceMap = dissMap;
      tonnetz.resize();
      tonnetz.render();
    } else {
      rebuildAnnular();
    }
  }

  updateStats();
}

function renderMap(): void {
  const canvas = document.getElementById('chordmap') as HTMLCanvasElement;
  const rect = canvas.parentElement!.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  chordMap.resize(w, h);
  layoutNodes(filteredNodes, w, h);
  chordMap.setNodes(filteredNodes);
  chordMap.render();
}

function updateStats(): void {
  const statsEl = document.getElementById('stats')!;
  if (currentView === 'constellation') {
    const centerBits = constellation.getCenterBits();
    if (centerBits !== null) {
      // Show active filter summary
      const parts: string[] = ['Explore'];
      if (currentKey !== null && currentScale !== null) {
        const SCALE_NAMES: Record<string, string> = {
          major: 'major', minor: 'minor', dorian: 'Dorian', mixolydian: 'Mixolydian',
          lydian: 'Lydian', phrygian: 'Phrygian', harmmin: 'harm min', melmin: 'mel min',
          pentatonic: 'pentatonic', blues: 'blues', wholetone: 'whole tone', chromatic: 'chromatic',
        };
        const keyScaleEl = document.getElementById('key-scale') as HTMLSelectElement;
        const scaleName = SCALE_NAMES[keyScaleEl?.value] ?? keyScaleEl?.value ?? '';
        parts.push(`${NOTE_NAMES[currentKey]} ${scaleName}`);
      }
      if (currentRootFilter !== null) {
        parts.push(`root ${NOTE_NAMES[currentRootFilter]}`);
      }
      if (cardinalityFilter.size < 6) {
        const cards = [...cardinalityFilter].sort();
        parts.push(`${cards.join('/')}‑note`);
      }
      if (searchQuery) {
        parts.push(`"${searchQuery}"`);
      }
      statsEl.textContent = parts.length > 1 ? parts.join(' · ') : chordLabel(centerBits);
    } else {
      statsEl.textContent = 'Click Random to start';
    }
  } else {
    let text = `${filteredNodes.length} chords`;
    if (currentScale) {
      const inKey = filteredNodes.filter(n => (n as any)._inKey).length;
      text += ` (${inKey} in key)`;
    }
    if (searchQuery) {
      const matches = filteredNodes.filter(n => (n as any)._searchMatch).length;
      text += ` (${matches} match)`;
    }
    statsEl.textContent = text;
  }
}

// ─── Info panel ─────────────────────────────────────────────────────

function updateInfoPanel(bits: number, midi: number[]): void {
  const nameEl = document.getElementById('chord-name')!;
  const infoEl = document.getElementById('chord-info')!;

  const label = chordLabel(bits);
  nameEl.textContent = label;

  const profile = TIMBRES[currentTimbre];
  const score = consonanceScore(midi, profile);
  const pcs = bitsToArray(bits);
  const names = allChordNames(bits);

  let html = `
    <div class="info-row"><span class="label">Notes</span><span>${pcs.map(pc => NOTE_NAMES[pc]).join(' ')}</span></div>
    <div class="info-row"><span class="label">MIDI</span><span>${midi.join(' ')}</span></div>
    <div class="info-row"><span class="label">Consonance</span><span>${(score * 100).toFixed(1)}%</span></div>
    <div class="consonance-bar"><div class="consonance-fill" style="width: ${score * 100}%; background: ${scoreColor(score)}"></div></div>
    <div class="info-row"><span class="label">Cardinality</span><span>${popcount(bits)} notes</span></div>
  `;

  if (names.length > 0) {
    const symbols = names.map(n => n.symbol);
    const shown = symbols.slice(0, 4).join(' / ') + (symbols.length > 4 ? ' / …' : '');
    html += `<div class="info-row"><span class="label">Names</span><span style="font-size:9px">${shown}</span></div>`;
  }

  infoEl.innerHTML = html;

  // VLD neighbors in sidebar
  updateNeighborsPanel(bits);
}

function updateNeighborsPanel(bits: number): void {
  const el = document.getElementById('neighbors-list')!;
  const pool = filteredNodes.length > 0
    ? filteredNodes.map(n => ({ bits: n.bits, label: n.label }))
    : []; // constellation doesn't use filteredNodes, use empty for now
  const neighbors = findNearestByVLD(bits, pool, 8);

  if (neighbors.length === 0) {
    el.innerHTML = '<div class="hint-text">Navigate to see neighbors</div>';
    return;
  }

  el.innerHTML = neighbors.map(n => {
    return `<div class="neighbor-item" data-bits="${n.bits}">
      <span>${n.label}</span>
      <small>VLD:${n.distance} CT:${n.commonTones}</small>
    </div>`;
  }).join('');

  for (const item of el.querySelectorAll('.neighbor-item')) {
    item.addEventListener('click', () => {
      const nbits = parseInt((item as HTMLElement).dataset.bits!);
      const midi = voiceChord(nbits);
      synth.playChord(midi);
      startScope();
      updateInfoPanel(nbits, midi);

      if (currentView === 'constellation') {
        constellation.navigateTo(nbits);
      } else if (currentView === 'map') {
        const targetNode = filteredNodes.find(n => n.bits === nbits);
        if (targetNode) {
          chordMap.activeNode = targetNode;
          chordMap.render();
        }
      }
    });
  }
}

function scoreColor(score: number): string {
  const hue = score * score * 50;
  return `hsl(${hue}, 85%, 50%)`;
}

// ─── Progression (always-visible bar) ───────────────────────────────

function addToProgression(bits: number): void {
  progression.push({ bits, label: chordLabel(bits), inversion: currentInversion, duration: 2 });
  renderProgressionList();
}

function renderProgressionList(): void {
  const el = document.getElementById('progression-list')!;
  if (progression.length === 0) {
    el.innerHTML = '<span class="hint-text" style="font-size:9px;color:#444">right-click or use search + to add chords</span>';
    return;
  }

  const INV_LABELS = ['R', '1', '2', '3'];
  const DUR_OPTIONS = [
    { v: '0.5', label: '⅛' },
    { v: '1',   label: '♩' },
    { v: '2',   label: '𝅗𝅥' },
    { v: '4',   label: '○' },
  ];

  el.innerHTML = progression.map((p, i) => {
    const vld = i > 0 ? vldBits(progression[i - 1].bits, p.bits) : -1;
    const sep = i > 0
      ? `<div class="prog-sep" data-before="${i}">→<br><span style="font-size:7px;color:#333">${vld >= 0 ? vld : ''}</span></div>`
      : '';

    const invOpts = INV_LABELS.map((l, v) =>
      `<option value="${v}"${p.inversion === v ? ' selected' : ''}>${l}</option>`).join('');
    const durOpts = DUR_OPTIONS.map(d =>
      `<option value="${d.v}"${p.duration === parseFloat(d.v) ? ' selected' : ''}>${d.label}</option>`).join('');

    return `${sep}<div class="prog-pill" draggable="true" data-idx="${i}">
      <span class="drag-handle" title="Drag to reorder">⠿</span>
      <span class="pill-name">${p.label}</span>
      <select class="pill-select pill-inv" title="Inversion">${invOpts}</select>
      <select class="pill-select pill-dur" title="Duration">${durOpts}</select>
      <button class="pill-btn dup" data-idx="${i}" title="Duplicate">⧉</button>
      <button class="pill-btn rem" data-idx="${i}" title="Remove">×</button>
    </div>`;
  }).join('');

  // Wire pills
  for (const pill of el.querySelectorAll<HTMLElement>('.prog-pill')) {
    const idx = parseInt(pill.dataset.idx!);

    // Audition on click (not on controls)
    pill.addEventListener('click', (e) => {
      const t = e.target as HTMLElement;
      if (t.classList.contains('pill-btn') || t.classList.contains('pill-select') || t.classList.contains('drag-handle')) return;
      const p = progression[idx];
      const midi = voiceProgChord(p);
      synth.playChord(midi);
      startScope();
      updateInfoPanel(p.bits, midi);
    });

    // Inversion
    const invSel = pill.querySelector<HTMLSelectElement>('.pill-inv')!;
    invSel.addEventListener('change', () => { progression[idx].inversion = parseInt(invSel.value); });

    // Duration
    const durSel = pill.querySelector<HTMLSelectElement>('.pill-dur')!;
    durSel.addEventListener('change', () => { progression[idx].duration = parseFloat(durSel.value); });

    // Duplicate
    pill.querySelector('.dup')!.addEventListener('click', () => {
      progression.splice(idx + 1, 0, { ...progression[idx] });
      renderProgressionList();
    });

    // Remove
    pill.querySelector('.rem')!.addEventListener('click', () => {
      progression.splice(idx, 1);
      renderProgressionList();
    });

    // Drag & drop
    pill.addEventListener('dragstart', (e) => {
      dragSrcIdx = idx;
      pill.classList.add('dragging');
      e.dataTransfer!.effectAllowed = 'move';
    });
    pill.addEventListener('dragend', () => {
      dragSrcIdx = -1;
      pill.classList.remove('dragging');
      el.querySelectorAll('.prog-pill').forEach(p => p.classList.remove('drag-over'));
    });
    pill.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer!.dropEffect = 'move';
      el.querySelectorAll('.prog-pill').forEach(p => p.classList.remove('drag-over'));
      pill.classList.add('drag-over');
    });
    pill.addEventListener('drop', (e) => {
      e.preventDefault();
      if (dragSrcIdx < 0 || dragSrcIdx === idx) return;
      const item = progression.splice(dragSrcIdx, 1)[0];
      const insertAt = dragSrcIdx < idx ? idx - 1 : idx;
      progression.splice(insertAt, 0, item);
      renderProgressionList();
    });
  }
}

function getArpNotes(midi: number[], mode: typeof progArpMode): number[] {
  const sorted = [...midi].sort((a, b) => a - b);
  switch (mode) {
    case 'up':     return sorted;
    case 'down':   return [...sorted].reverse();
    case 'updown': return [...sorted, ...sorted.slice(1, -1).reverse()];
    case 'random': {
      const r = [...sorted];
      for (let k = r.length - 1; k > 0; k--) {
        const j = Math.floor(Math.random() * (k + 1));
        [r[k], r[j]] = [r[j], r[k]];
      }
      return r;
    }
    default: return sorted;
  }
}

function playArp(midi: number[], chordMs: number): void {
  if (arpSubTimeout !== null) { clearTimeout(arpSubTimeout); arpSubTimeout = null; }
  const msPerBeat = 60000 / progTempo;
  const arpMs = progArpRate * msPerBeat;
  const noteSec = (arpMs * progArpGate) / 1000;
  const notes = getArpNotes(midi, progArpMode);
  let noteIdx = 0;
  let elapsed = 0;

  function arpTick(): void {
    if (!progPlaying || elapsed >= chordMs) return;
    synth.playChord([notes[noteIdx % notes.length]], noteSec);
    noteIdx++;
    elapsed += arpMs;
    if (elapsed < chordMs) {
      arpSubTimeout = window.setTimeout(arpTick, arpMs);
    }
  }
  arpTick();
}

function playProgression(): void {
  if (progPlaying) stopProgression();
  progPlaying = true;
  updatePlayButton();

  let i = 0;

  function playNext(): void {
    if (!progPlaying) return;
    if (i >= progression.length) {
      if (progLoop) { i = 0; }
      else { stopProgression(); return; }
    }
    const p = progression[i];
    const msPerBeat = 60000 / progTempo;
    const chordMs = p.duration * msPerBeat;
    const chordSec = chordMs / 1000;
    const midi = voiceProgChord(p);

    if (progArpMode === 'off') {
      synth.playChord(midi, chordSec * 0.92);
    } else {
      synth.stopAll();
      playArp(midi, chordMs);
    }

    startScope();
    updateInfoPanel(p.bits, midi);
    if (currentView === 'constellation') constellation.navigateTo(p.bits);

    i++;
    progInterval = window.setTimeout(playNext, chordMs);
  }

  playNext();
}

function stopProgression(): void {
  progPlaying = false;
  if (progInterval !== null) { clearTimeout(progInterval); progInterval = null; }
  if (arpSubTimeout !== null) { clearTimeout(arpSubTimeout); arpSubTimeout = null; }
  synth.stopAll();
  updatePlayButton();
}

function updatePlayButton(): void {
  const btn = document.getElementById('play-prog');
  if (btn) btn.textContent = progPlaying ? 'Stop' : 'Play';
}

// ─── View switching ─────────────────────────────────────────────────

function rebuildAnnular(): void {
  const annularCanvas = document.getElementById('annular') as HTMLCanvasElement;
  const rect = annularCanvas.parentElement!.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  annular.resize(w, h);

  const annularNodes = buildAnnularNodes(filteredNodes);
  for (let i = 0; i < annularNodes.length && i < filteredNodes.length; i++) {
    (annularNodes[i] as any)._inKey = (filteredNodes[i] as any)._inKey;
    (annularNodes[i] as any)._searchMatch = (filteredNodes[i] as any)._searchMatch;
  }

  const cx = w / 2;
  const cy = h / 2;
  const maxR = Math.min(cx, cy) - 20;
  layoutAnnular(annularNodes, cx, cy, maxR);
  annular.setNodes(annularNodes);
  annular.render();
}

function switchView(view: 'constellation' | 'map' | 'tonnetz' | 'annular'): void {
  currentView = view;
  const constellationCanvas = document.getElementById('constellation') as HTMLCanvasElement;
  const mapCanvas = document.getElementById('chordmap') as HTMLCanvasElement;
  const tonnetzCanvas = document.getElementById('tonnetz') as HTMLCanvasElement;
  const annularCanvas = document.getElementById('annular') as HTMLCanvasElement;

  constellationCanvas.style.display = view === 'constellation' ? 'block' : 'none';
  mapCanvas.style.display = view === 'map' ? 'block' : 'none';
  tonnetzCanvas.style.display = view === 'tonnetz' ? 'block' : 'none';
  annularCanvas.style.display = view === 'annular' ? 'block' : 'none';
  constellationCanvas.style.zIndex = view === 'constellation' ? '1' : '0';
  mapCanvas.style.zIndex = view === 'map' ? '1' : '0';
  tonnetzCanvas.style.zIndex = view === 'tonnetz' ? '1' : '0';
  annularCanvas.style.zIndex = view === 'annular' ? '1' : '0';

  rebuild();

  // Update active buttons
  document.getElementById('btn-constellation')!.classList.toggle('active', view === 'constellation');
  document.getElementById('btn-map')!.classList.toggle('active', view === 'map');
  document.getElementById('btn-tonnetz')!.classList.toggle('active', view === 'tonnetz');
  document.getElementById('btn-annular')!.classList.toggle('active', view === 'annular');

  // Disable "Data Set" dropdown in Explore (it has no effect there)
  const viewModeEl = document.getElementById('view-mode') as HTMLSelectElement;
  viewModeEl.disabled = (view === 'constellation');

  // Show/hide Explore-specific controls
  const exploreControls = document.getElementById('explore-controls')!;
  exploreControls.style.display = view === 'constellation' ? 'flex' : 'none';
}

// ─── Control wiring ─────────────────────────────────────────────────

function wireControls(): void {
  // View toggle buttons
  document.getElementById('btn-constellation')!.addEventListener('click', () => switchView('constellation'));
  document.getElementById('btn-map')!.addEventListener('click', () => switchView('map'));
  document.getElementById('btn-tonnetz')!.addEventListener('click', () => switchView('tonnetz'));
  document.getElementById('btn-annular')!.addEventListener('click', () => switchView('annular'));

  // Explore: Lock and Help buttons
  document.getElementById('explore-lock')!.addEventListener('click', () => {
    const locked = !constellation.isLocked();
    constellation.setLocked(locked);
    document.getElementById('explore-lock')!.classList.toggle('active', locked);
  });
  document.getElementById('explore-help')!.addEventListener('click', () => {
    constellation.toggleHelp();
    document.getElementById('explore-help')!.classList.toggle('active');
  });

  // Random button
  document.getElementById('random-btn')!.addEventListener('click', () => {
    const bits = constellation.randomChord();
    const midi = voiceChord(bits);
    synth.playChord(midi);
    startScope();
    updateInfoPanel(bits, midi);

    if (currentView === 'constellation') {
      constellation.navigateTo(bits);
    } else {
      switchView('constellation');
      constellation.navigateTo(bits);
    }
  });

  // More toggle
  document.getElementById('more-toggle')!.addEventListener('click', () => {
    const moreEl = document.getElementById('more-controls')!;
    moreEl.classList.toggle('open');
    const btn = document.getElementById('more-toggle')!;
    btn.textContent = moreEl.classList.contains('open') ? 'less' : 'more';
  });

  // View mode (for chord map)
  const viewSelect = document.getElementById('view-mode') as HTMLSelectElement;
  viewSelect.addEventListener('change', () => {
    currentViewMode = viewSelect.value as ViewMode;
    rebuild();
  });

  // Root filter
  const rootSelect = document.getElementById('root-filter') as HTMLSelectElement;
  rootSelect.addEventListener('change', () => {
    const val = rootSelect.value;
    currentRootFilter = val === 'all' ? null : parseInt(val);
    rebuild();
  });

  // Key filter
  const keyRootSelect = document.getElementById('key-root') as HTMLSelectElement;
  const keyScaleSelect = document.getElementById('key-scale') as HTMLSelectElement;
  function updateKeyFilter(): void {
    const root = keyRootSelect.value;
    const scale = keyScaleSelect.value;
    if (root === 'none') {
      currentKey = null;
      currentScale = null;
    } else {
      currentKey = parseInt(root);
      currentScale = getScalePCs(currentKey, SCALES[scale]);
    }
    rebuild();
  }
  keyRootSelect.addEventListener('change', updateKeyFilter);
  keyScaleSelect.addEventListener('change', updateKeyFilter);

  // Voicing type
  const voicingSelect = document.getElementById('voicing-type') as HTMLSelectElement;
  voicingSelect.addEventListener('change', () => {
    currentVoicing = voicingSelect.value;
  });

  // Inversion
  const invSelect = document.getElementById('inversion') as HTMLSelectElement;
  invSelect.addEventListener('change', () => {
    currentInversion = parseInt(invSelect.value);
  });

  // Timbre
  const timbreSelect = document.getElementById('timbre-select') as HTMLSelectElement;
  for (const [key, profile] of Object.entries(TIMBRES)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = profile.name;
    if (key === DEFAULT_TIMBRE) opt.selected = true;
    timbreSelect.appendChild(opt);
  }
  timbreSelect.addEventListener('change', () => {
    currentTimbre = timbreSelect.value;
    synth.timbre = TIMBRES[currentTimbre];
    rebuild();
  });

  // Cardinality checkboxes
  const cardContainer = document.getElementById('card-filters')!;
  for (let c = 2; c <= 7; c++) {
    const label = document.createElement('label');
    label.className = 'card-filter';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.dataset.card = String(c);
    cb.addEventListener('change', () => {
      if (cb.checked) cardinalityFilter.add(c);
      else cardinalityFilter.delete(c);
      rebuild();
    });
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` ${c}`));
    cardContainer.appendChild(label);
  }

  // Search with dropdown
  const searchInput = document.getElementById('search-input') as HTMLInputElement;
  const searchDropdown = document.getElementById('search-dropdown')!;
  let searchTimeout: number;
  let dropdownFocusIdx = -1;

  function searchVocab(q: string): { bits: number; label: string; altNames: string[] }[] {
    if (!q) return [];
    const ql = q.toLowerCase();
    const seen = new Set<number>();
    const results: { bits: number; label: string; altNames: string[] }[] = [];
    for (const entry of CHORD_VOCABULARY) {
      if (seen.has(entry.bits)) continue;
      const allNames = allChordNames(entry.bits).map(n => n.symbol);
      if (allNames.some(n => n.toLowerCase().includes(ql))) {
        seen.add(entry.bits);
        const label = allNames.length <= 1 ? allNames[0]
          : allNames.length === 2 ? `${allNames[0]} / ${allNames[1]}`
          : `${allNames[0]}…`;
        results.push({ bits: entry.bits, label, altNames: allNames.slice(1) });
        if (results.length >= 12) break;
      }
    }
    return results;
  }

  function navigateToChord(bits: number): void {
    const midi = voiceChord(bits);
    synth.playChord(midi);
    startScope();
    updateInfoPanel(bits, midi);

    if (currentView === 'constellation') {
      constellation.navigateTo(bits);
    } else if (currentView === 'map') {
      const node = filteredNodes.find(n => n.bits === bits);
      if (node) {
        chordMap.focusOn(node);
      } else {
        // Chord not in current filter — switch to Explore
        switchView('constellation');
        constellation.navigateTo(bits);
      }
    } else if (currentView === 'annular') {
      const node = annular.getNodes().find(n => n.bits === bits);
      if (node) {
        annular.focusOn(node);
      } else {
        switchView('constellation');
        constellation.navigateTo(bits);
      }
    } else {
      // Tonnetz or unknown — switch to constellation
      switchView('constellation');
      constellation.navigateTo(bits);
    }

    updateStats();
  }

  function renderDropdown(results: { bits: number; label: string; altNames: string[] }[]): void {
    dropdownFocusIdx = -1;
    if (results.length === 0) {
      searchDropdown.classList.remove('open');
      return;
    }
    searchDropdown.innerHTML = results.map((r, i) =>
      `<div class="search-result" data-idx="${i}">
        <span class="search-result-name">${r.label}</span>
        ${r.altNames.length > 0 ? `<span class="search-result-also">${r.altNames.slice(0, 2).join(', ')}${r.altNames.length > 2 ? ', …' : ''}</span>` : ''}
        <div class="search-result-btns">
          <button class="sr-btn go" data-bits="${r.bits}" title="Go to chord">Go</button>
          <button class="sr-btn add" data-bits="${r.bits}" title="Add to progression">+</button>
        </div>
      </div>`
    ).join('');
    searchDropdown.classList.add('open');

    for (const el of searchDropdown.querySelectorAll('.sr-btn.go')) {
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        const bits = parseInt((el as HTMLElement).dataset.bits!);
        closeDropdown();
        navigateToChord(bits);
      });
    }
    for (const el of searchDropdown.querySelectorAll('.sr-btn.add')) {
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        const bits = parseInt((el as HTMLElement).dataset.bits!);
        addToProgression(bits);
      });
    }
    for (const row of searchDropdown.querySelectorAll('.search-result')) {
      row.addEventListener('click', () => {
        const idx = parseInt((row as HTMLElement).dataset.idx!);
        const r = results[idx];
        closeDropdown();
        navigateToChord(r.bits);
      });
    }
  }

  function closeDropdown(): void {
    searchDropdown.classList.remove('open');
    searchDropdown.innerHTML = '';
    dropdownFocusIdx = -1;
  }

  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = window.setTimeout(() => {
      const q = searchInput.value.trim();
      searchQuery = q;
      const results = searchVocab(q);
      renderDropdown(results);
      rebuild(); // still highlight in views
    }, 150);
  });

  // Keyboard nav in dropdown
  searchInput.addEventListener('keydown', (e) => {
    const items = searchDropdown.querySelectorAll('.search-result');
    if (!searchDropdown.classList.contains('open') || items.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      dropdownFocusIdx = Math.min(dropdownFocusIdx + 1, items.length - 1);
      items.forEach((el, i) => el.classList.toggle('focused', i === dropdownFocusIdx));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      dropdownFocusIdx = Math.max(dropdownFocusIdx - 1, 0);
      items.forEach((el, i) => el.classList.toggle('focused', i === dropdownFocusIdx));
    } else if (e.key === 'Enter' && dropdownFocusIdx >= 0) {
      e.preventDefault();
      const focused = items[dropdownFocusIdx] as HTMLElement;
      const idx = parseInt(focused.dataset.idx!);
      const results = searchVocab(searchInput.value.trim());
      if (results[idx]) {
        closeDropdown();
        navigateToChord(results[idx].bits);
      }
    } else if (e.key === 'Escape') {
      closeDropdown();
    }
  });

  // Close dropdown when clicking outside
  document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target as Node) && !searchDropdown.contains(e.target as Node)) {
      closeDropdown();
    }
  });

  // Progression controls
  const tempoInput = document.getElementById('prog-tempo') as HTMLInputElement;
  tempoInput.addEventListener('change', () => {
    progTempo = Math.max(40, Math.min(240, parseInt(tempoInput.value) || 100));
  });

  const loopToggle = document.getElementById('prog-loop') as HTMLInputElement;
  loopToggle.addEventListener('change', () => { progLoop = loopToggle.checked; });

  document.getElementById('play-prog')!.addEventListener('click', () => {
    if (progression.length < 1) return;
    if (progPlaying) stopProgression();
    else playProgression();
  });

  document.getElementById('clear-prog')!.addEventListener('click', () => {
    stopProgression();
    progression = [];
    renderProgressionList();
  });

  document.getElementById('midi-export')!.addEventListener('click', exportProgressionMIDI);

  // Arp controls
  const arpModeSelect = document.getElementById('arp-mode') as HTMLSelectElement;
  arpModeSelect.addEventListener('change', () => {
    progArpMode = arpModeSelect.value as typeof progArpMode;
  });

  const arpRateSelect = document.getElementById('arp-rate') as HTMLSelectElement;
  arpRateSelect.addEventListener('change', () => {
    progArpRate = parseFloat(arpRateSelect.value);
  });

  const arpGateInput = document.getElementById('arp-gate') as HTMLInputElement;
  const arpGateVal = document.getElementById('arp-gate-val')!;
  arpGateInput.addEventListener('input', () => {
    progArpGate = parseFloat(arpGateInput.value);
    arpGateVal.textContent = Math.round(progArpGate * 100) + '%';
  });

  // Register constraints toggle
  const regToggle = document.getElementById('register-constraints') as HTMLInputElement;
  regToggle.addEventListener('change', () => {
    registerConstraints = regToggle.checked;
  });

  // Tuning toggle
  document.getElementById('btn-tet')!.addEventListener('click', () => {
    synth.tuning = 'tet';
    document.getElementById('btn-tet')!.classList.add('active');
    document.getElementById('btn-just')!.classList.remove('active');
  });
  document.getElementById('btn-just')!.addEventListener('click', () => {
    synth.tuning = 'just';
    document.getElementById('btn-just')!.classList.add('active');
    document.getElementById('btn-tet')!.classList.remove('active');
  });

  // Synth method selector
  const synthSelect = document.getElementById('synth-method') as HTMLSelectElement;
  synthSelect.addEventListener('change', () => {
    currentSynthMethod = synthSelect.value as SynthMethod;
    synth.method = currentSynthMethod;
  });

  // Macro knobs
  const macroNames = ['brightness', 'warmth', 'attack', 'body', 'air'] as const;
  for (const name of macroNames) {
    const slider = document.getElementById(`macro-${name}`) as HTMLInputElement;
    if (!slider) continue;
    const valEl = document.getElementById(`macro-${name}-val`);
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      synth.setMacro(name, v);
      if (valEl) valEl.textContent = v.toFixed(2);
    });
  }
}

// ─── Oscilloscope / spectrogram ──────────────────────────────────────

function startScope(): void {
  if (!analyserNode) {
    analyserNode = synth.getAnalyserNode();
    if (!analyserNode) return;
  }
  scopeCanvas = document.getElementById('scope-canvas') as HTMLCanvasElement | null;
  if (!scopeCanvas) return;
  scopeCtx = scopeCanvas.getContext('2d');
  if (scopeAnimFrame) cancelAnimationFrame(scopeAnimFrame);
  drawScope();
}

function drawScope(): void {
  if (!analyserNode || !scopeCanvas || !scopeCtx) return;
  const w = scopeCanvas.width;
  const h = scopeCanvas.height;
  const ctx = scopeCtx;

  const bufLen = analyserNode.frequencyBinCount;
  const timeBuf = new Uint8Array(bufLen);
  const freqBuf = new Uint8Array(bufLen);
  analyserNode.getByteTimeDomainData(timeBuf);
  analyserNode.getByteFrequencyData(freqBuf);

  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, w, h);

  // Waveform (top half)
  const midY = h * 0.4;
  ctx.beginPath();
  ctx.strokeStyle = '#4ecdc4';
  ctx.lineWidth = 1.5;
  const sliceW = w / bufLen;
  for (let i = 0; i < bufLen; i++) {
    const v = timeBuf[i] / 128.0;
    const y = v * midY;
    if (i === 0) ctx.moveTo(0, y);
    else ctx.lineTo(i * sliceW, y);
  }
  ctx.stroke();

  // Spectrum (bottom half) — log frequency scale, 20Hz–20kHz
  const specH = h * 0.5;
  const specY = h * 0.5;
  const sampleRate = 48000;
  const maxBin = Math.floor(20000 / (sampleRate / 2) * bufLen);
  const minBin = Math.max(1, Math.floor(20 / (sampleRate / 2) * bufLen));
  const logMin = Math.log10(minBin);
  const logMax = Math.log10(maxBin);

  for (let px = 0; px < w; px++) {
    const t = px / w;
    const binF = Math.pow(10, logMin + t * (logMax - logMin));
    const bin = Math.round(binF);
    if (bin < 0 || bin >= bufLen) continue;
    const val = freqBuf[bin] / 255;
    const barH = val * specH;
    const hue = 200 + val * 80;
    ctx.fillStyle = `hsla(${hue}, 75%, ${35 + val * 35}%, 0.9)`;
    ctx.fillRect(px, specY + specH - barH, 1, barH);
  }

  ctx.fillStyle = 'rgba(255,255,255,0.15)';
  ctx.font = '8px system-ui';
  ctx.textAlign = 'left';
  ctx.fillText('waveform', 4, 10);
  ctx.fillText('spectrum', 4, specY + 10);

  scopeAnimFrame = requestAnimationFrame(drawScope);
}

// ─── MIDI export ─────────────────────────────────────────────────────

function exportProgressionMIDI(): void {
  if (progression.length === 0) return;

  const ticksPerBeat = 480;
  const tempo = progTempo;

  const events: number[] = [];

  // Tempo meta event
  const microsecondsPerBeat = Math.round(60000000 / tempo);
  events.push(0x00, 0xFF, 0x51, 0x03,
    (microsecondsPerBeat >> 16) & 0xFF,
    (microsecondsPerBeat >> 8) & 0xFF,
    microsecondsPerBeat & 0xFF);

  let currentTick = 0;
  for (const p of progression) {
    const midi = voiceProgChord(p);
    const chordDurationTicks = Math.round(ticksPerBeat * p.duration);

    for (let i = 0; i < midi.length; i++) {
      events.push(...writeVLQ(i === 0 && currentTick > 0 ? 0 : 0));
      events.push(0x90, midi[i], 100);
    }

    for (let i = 0; i < midi.length; i++) {
      events.push(...writeVLQ(i === 0 ? chordDurationTicks : 0));
      events.push(0x80, midi[i], 0);
    }

    currentTick += chordDurationTicks;
  }

  // End of track
  events.push(0x00, 0xFF, 0x2F, 0x00);

  const trackData = new Uint8Array(events);
  const header = new Uint8Array([
    0x4D, 0x54, 0x68, 0x64,
    0x00, 0x00, 0x00, 0x06,
    0x00, 0x00,
    0x00, 0x01,
    (ticksPerBeat >> 8) & 0xFF, ticksPerBeat & 0xFF,
  ]);

  const trackHeader = new Uint8Array([
    0x4D, 0x54, 0x72, 0x6B,
    (trackData.length >> 24) & 0xFF,
    (trackData.length >> 16) & 0xFF,
    (trackData.length >> 8) & 0xFF,
    trackData.length & 0xFF,
  ]);

  const midiFile = new Uint8Array(header.length + trackHeader.length + trackData.length);
  midiFile.set(header, 0);
  midiFile.set(trackHeader, header.length);
  midiFile.set(trackData, header.length + trackHeader.length);

  const blob = new Blob([midiFile], { type: 'audio/midi' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'progression.mid';
  a.click();
  URL.revokeObjectURL(url);
}

function writeVLQ(value: number): number[] {
  if (value < 0) value = 0;
  if (value < 128) return [value];
  const bytes: number[] = [];
  bytes.unshift(value & 0x7F);
  value >>= 7;
  while (value > 0) {
    bytes.unshift((value & 0x7F) | 0x80);
    value >>= 7;
  }
  return bytes;
}

// ─── Init ───────────────────────────────────────────────────────────

function init(): void {
  try {
    const constellationCanvas = document.getElementById('constellation') as HTMLCanvasElement;
    const mapCanvas = document.getElementById('chordmap') as HTMLCanvasElement;
    const tonnetzCanvas = document.getElementById('tonnetz') as HTMLCanvasElement;
    const annularCanvas = document.getElementById('annular') as HTMLCanvasElement;

    constellation = new ConstellationRenderer(constellationCanvas);
    chordMap = new ChordMapRenderer(mapCanvas);
    tonnetz = new TonnetzRenderer(tonnetzCanvas);
    annular = new AnnularRenderer(annularCanvas);
    synth = new MultiSynth();

    // Set up constellation profile
    const profile = TIMBRES[currentTimbre];
    constellation.setProfile(profile, currentScale, cardinalityFilter, currentRootFilter, searchQuery);

    // Size constellation
    const wrap = document.getElementById('canvas-wrap')!;
    const rect = wrap.getBoundingClientRect();
    constellation.resize(rect.width, rect.height);

    // Wire constellation events
    constellation.onChordClick = (bits, _midi) => {
      const voiced = voiceChord(bits);
      synth.playChord(voiced);
      startScope();
      updateInfoPanel(bits, voiced);
    };

    constellation.onChordHover = (node) => {
      const hoverEl = document.getElementById('hover-info')!;
      if (node) {
        hoverEl.textContent = node.label;
        hoverEl.style.opacity = '1';
      } else {
        hoverEl.style.opacity = '0';
      }
    };

    constellation.onAddToProgression = (bits) => {
      addToProgression(bits);
    };

    chordMap.onAddToProgression = (bits) => addToProgression(bits);
    annular.onAddToProgression = (bits) => addToProgression(bits);
    tonnetz.onAddToProgression = (bits) => addToProgression(bits);

    // Wire chord map events
    chordMap.onChordClick = (node, _midi) => {
      const voiced = voiceChord(node.bits);
      synth.playChord(voiced);
      startScope();
      updateInfoPanel(node.bits, voiced);
    };

    chordMap.onChordHover = (node) => {
      const hoverEl = document.getElementById('hover-info')!;
      if (node) {
        hoverEl.textContent = node.label;
        hoverEl.style.opacity = '1';
      } else {
        hoverEl.style.opacity = '0';
      }
    };

    // Wire annular events
    annular.onChordClick = (node, _midi) => {
      const voiced = voiceChord(node.bits);
      synth.playChord(voiced);
      startScope();
      updateInfoPanel(node.bits, voiced);
    };

    annular.onChordHover = (node) => {
      const hoverEl = document.getElementById('hover-info')!;
      if (node) {
        hoverEl.textContent = node.label;
        hoverEl.style.opacity = '1';
      } else {
        hoverEl.style.opacity = '0';
      }
    };

    // Wire tonnetz events
    tonnetz.onTriadClick = (_root, _isMajor, bits, _midi) => {
      const voiced = voiceChord(bits);
      synth.playChord(voiced);
      startScope();
      updateInfoPanel(bits, voiced);
    };

    wireControls();

    // Print catalog stats
    const stats = catalogStats();
    console.log(`ChordSpace: ${stats.forteClasses} Forte classes, ${stats.totalNamed} named chords`);

    // Start with C major in constellation
    const cmajBits = pcsToBits([0, 4, 7]);
    constellation.navigateTo(cmajBits);
    const cmajMidi = voiceChord(cmajBits);
    updateInfoPanel(cmajBits, cmajMidi);

    // Initial progression bar
    renderProgressionList();

    // Disable Data Set dropdown in initial Explore view
    (document.getElementById('view-mode') as HTMLSelectElement).disabled = true;

    // Handle resize
    window.addEventListener('resize', () => rebuild());

    console.log('ChordSpace init complete');
  } catch (e) {
    console.error('ChordSpace init error:', e);
  }
}

document.addEventListener('DOMContentLoaded', init);
