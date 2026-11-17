// gta6-twitch-ccu-forecast-01 entry (Node 22, stdlib only, no network).
//
// Model: rolling-unlock median-ratio forecaster, blended with the full pool.
//   Grand Theft Auto VI unlocks at midnight local time, rolling region by
//   region, so the UTC clock hour of its release day sits at roughly the same
//   phase of the rolling unlock for every launch that unlocked the same way
//   (NZ ~11-13h before UTC midnight, US East ~04:00-05:00Z, US West
//   ~07:00-08:00Z). For each comparable launch we take the log-ratio of its
//   viewers in release-day hour h to its viewers in release-day hour 1 (the
//   same clock-hour slot our case's hour-1 print occupies).
//   The forecast log-ratio at each hour is a blend of two robust medians:
//   70% the median over the ROLLING pool (comparables whose unlock note says
//   they unlocked rolling / at midnight local — the same mechanics as GTA VI
//   and as the four practice launches) and 30% the median over ALL eligible
//   comparables (stability: the rolling pool is small, and a keyword miss
//   must not sink the forecast). No shrinkage is applied: the median is the
//   pinball-optimal centre, and shrinking it toward persistence only biased
//   the evening surge down (v1's shrinkage cost ~0.034 of practice loss).
//   The spread at each hour is the cross-comparable dispersion (MAD-based)
//   of the rolling pool's log-ratios, falling back to the full pool's, so
//   hours where launches disagree get honestly wide quantiles.
//   Comparables are filtered, not tuned per launch: a comparable must have a
//   real release-day hour-1 point and at least 12 of the 24 release-day
//   hours present (this drops the GTA VI pre-release category, whose "day" is
//   its Aug 27 premiere and which has no release-day data yet), and viewers
//   must be positive. No per-case tuning, no randomness: two runs are
//   byte-identical.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Weight of the rolling-unlock pool's median in the blended log-ratio. */
const ROLLING_WEIGHT = 0.7;

interface HourlyPoint { from: string; viewers: number; channels?: number | null }
interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  platforms?: string;
  hourly: HourlyPoint[];
}
interface Case {
  id?: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Median of a non-empty number array (sorted copy). */
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** Parse an ISO timestamp to ms; NaN when unparseable. */
function tms(iso: string): number {
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : NaN;
}

/**
 * True when the comparable's unlock note says it unlocked rolling at
 * midnight local time — the same mechanics as GTA VI (and as the practice
 * launches). Simultaneous global unlocks and unverified notes do not match.
 */
function isRollingUnlock(c: Comparable): boolean {
  return /rolling|midnight local/i.test(c.unlock ?? '');
}

/**
 * Build, for one comparable, a map from release-day hour (1..24, UTC clock
 * hour of its day_start) to viewers, using only exact hourly points.
 */
function releaseDayMap(c: Comparable): Map<number, number> {
  const base = tms(c.day_start);
  const out = new Map<number, number>();
  if (!Number.isFinite(base)) return out;
  for (const p of c.hourly ?? []) {
    const t = tms(p.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.round((t - base) / 3600000) + 1;
    if (h >= 1 && h <= 24 && typeof p.viewers === 'number' && p.viewers > 0) {
      if (!out.has(h)) out.set(h, p.viewers);
    }
  }
  return out;
}

/** Median + MAD-based spread of a log-ratio sample; null when too small. */
function medSpread(xs: number[]): { med: number; spread: number; n: number } | null {
  if (xs.length < 3) return null;
  const med = median(xs);
  const mad = median(xs.map((x) => Math.abs(x - med)));
  return { med, spread: 1.4826 * mad, n: xs.length };
}

function forecastCase(c: Case): Array<{ hour: number; q: number[] }> {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);

  // Per-hour log-ratios ln(viewers_h / viewers_1): full pool and rolling pool.
  const allRatios: number[][] = Array.from({ length: 25 }, () => []);
  const rollRatios: number[][] = Array.from({ length: 25 }, () => []);
  for (const comp of c.comparables ?? []) {
    const day = releaseDayMap(comp);
    const v1 = day.get(1);
    if (v1 === undefined || v1 < 1) continue;
    // Require a real release day: at least 12 of the 24 hours present.
    if (day.size < 12) continue;
    const rolling = isRollingUnlock(comp);
    for (let h = 2; h <= 24; h++) {
      const v = day.get(h);
      if (v !== undefined && v > 0) {
        const lr = Math.log(v / v1);
        allRatios[h].push(lr);
        if (rolling) rollRatios[h].push(lr);
      }
    }
  }

  return c.forecast.map(({ hour }) => {
    const all = medSpread(allRatios[hour] ?? []);
    const roll = medSpread(rollRatios[hour] ?? []);
    // Blend the two medians in log space; fall back to whichever pool has data.
    let med = 0;
    if (roll && all) med = ROLLING_WEIGHT * roll.med + (1 - ROLLING_WEIGHT) * all.med;
    else if (roll) med = roll.med;
    else if (all) med = all.med;
    else if ((allRatios[hour] ?? []).length > 0) med = median(allRatios[hour]) * 0.5;
    // Spread from the rolling pool when it has data, else the full pool's.
    let spread = (roll ?? all)?.spread ?? 0.35;
    // Safety caps: never forecast a median more than 50x hour 1, never below
    // 1/50th of it; spread kept in a sane band.
    med = Math.max(-Math.log(50), Math.min(Math.log(50), med));
    spread = Math.max(0.25, Math.min(1.1, spread));
    const q50 = p1 * Math.exp(med);
    const q = Z.map((z) => round2(Math.max(0.01, q50 * Math.exp(z * spread))));
    // Enforce non-decreasing (rounding can only tie, but stay safe).
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    return { hour, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
