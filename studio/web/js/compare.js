// Grouping specimens by the details the reader picks, and the statistics the
// results page shows for each grouping. No DOM here: the tests run this module
// under Node and check it against SciPy.
//
// The first factor picked is the main comparison: its levels sit side by side
// and each has a colour. Every further factor splits the specimens into
// strata -- a cluster of columns, a panel of curves -- and the main comparison
// is repeated inside each one. Specimens with no value for a factor are a
// level of their own ("Sex not set"), shown but never tested.

export const FACTORS = [
  { key: "genotype", label: "Genotype", plural: "genotypes" },
  { key: "age", label: "Age", plural: "ages" },
  { key: "sex", label: "Sex", plural: "sexes", names: { F: "Female", M: "Male" } },
  { key: "line", label: "Line", plural: "lines" },
  { key: "side", label: "Side", plural: "sides", names: { L: "Left", R: "Right" } },
  { key: "study", label: "Study", plural: "studies" },
  { key: "scanner", label: "Scanner", plural: "scanners" },
];
const BY_KEY = Object.fromEntries(FACTORS.map((f) => [f.key, f]));

export const factor = (key) => BY_KEY[key];
export const levelOf = (meta, key) => String(meta?.[key] ?? "").trim();

export function levelName(key, level) {
  const f = BY_KEY[key];
  if (!level) return `${f ? f.label : key} not set`;
  return f?.names?.[level.toUpperCase()] || level;
}

// ---------------------------------------------------------------- flags

// The analyses a flagged specimen can be left out of: results.EXCLUSION_SCOPES.
export const ANALYSES = [["molar", "Molar enamel"], ["incisor", "Incisor mineralization"], ["segmentation", "Segmentation volumes"]];

/** Whether a specimen's flag ({from: "all" | [analyses], note}) leaves it out of `analysis`. */
export function excludedFrom(flag, analysis) {
  return Boolean(flag) && (flag.from === "all" || (flag.from || []).includes(analysis));
}

/** "all results", or the analyses by name: "Incisor mineralization and Molar enamel". */
export function exclusionText(flag) {
  if (!flag) return "";
  if (flag.from === "all") return "all results";
  const names = ANALYSES.filter(([k]) => flag.from.includes(k)).map(([, label]) => label);
  return names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}` : names[0] || "";
}

// ---------------------------------------------------------------- level order

// Controls first, then heterozygotes, then everything else: a comparison reads
// "KO vs WT" with the reference level last.
const CONTROL = /^(wt|wild[\s_-]?type|ctrl?|ctr|control|\+\/\+)$/i;
const HET = /^(het|heterozygous?|\+\/-)$/i;
const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });

function stage(level) {
  // E18.5 < P0 < P22 < P100; anything else after, in natural order.
  const m = /^([EP])\s*(\d+(?:\.\d+)?)$/i.exec(level);
  return m ? [m[1].toUpperCase() === "E" ? 0 : 1, Number(m[2])] : [2, 0];
}

function compareLevels(key, a, b) {
  if (!a || !b) return !a && !b ? 0 : a ? -1 : 1;  // "not set" last
  if (key === "genotype") {
    const rank = (v) => (CONTROL.test(v) ? 0 : HET.test(v) ? 1 : 2);
    return rank(a) - rank(b) || natural(a, b);
  }
  if (key === "age") {
    const [sa, na] = stage(a), [sb, nb] = stage(b);
    return sa - sb || na - nb || natural(a, b);
  }
  return natural(a, b);
}

export function orderLevels(key, levels) {
  return [...new Set(levels)].sort((a, b) => compareLevels(key, a, b));
}

/** {factor: [levels in display order]} over every specimen, so hiding some never reorders or recolours the rest. */
export function levelOrder(samples) {
  const out = {};
  for (const f of FACTORS) out[f.key] = orderLevels(f.key, samples.map((s) => levelOf(s.meta, f.key)));
  return out;
}

/** Factors with at least `min` distinct values set among the specimens. */
export function factorsWithValues(samples, min = 1) {
  return FACTORS.filter((f) => new Set(samples.map((s) => levelOf(s.meta, f.key)).filter(Boolean)).size >= min)
    .map((f) => f.key);
}

// ---------------------------------------------------------------- grouping

const SEP = "\u0001";

function keyOf(meta, keys) {
  return keys.map((k) => levelOf(meta, k)).join(SEP);
}

function withinLabel(keys, meta) {
  return keys.map((k) => levelName(k, levelOf(meta, k))).join(", ");
}

function byOrder(keys, order) {
  return (a, b) => {
    for (const k of keys) {
      const list = order[k] || [];
      const d = list.indexOf(levelOf(a, k)) - list.indexOf(levelOf(b, k));
      if (d) return d;
    }
    return 0;
  };
}

/**
 * Specimens arranged for display: one stratum per combination of the split
 * factors (by[1..]), and in each one a cell per level of the main factor.
 *
 *   samples  [{name, meta}]
 *   by       ["genotype", "sex", ...]: the main factor, then the splits
 *   order    levelOrder() of every specimen
 *
 * Returns [{key, label, meta, cells: [{level, name, members: [names]}]}].
 */
export function arrange(samples, by, order) {
  const [main, ...splits] = by;
  const strata = new Map();
  for (const s of samples) {
    const key = keyOf(s.meta, splits);
    if (!strata.has(key)) strata.set(key, { key, label: withinLabel(splits, s.meta), meta: s.meta, members: [] });
    strata.get(key).members.push(s);
  }
  return [...strata.values()].sort((a, b) => byOrder(splits, order)(a.meta, b.meta)).map((st) => {
    const cells = new Map();
    for (const s of st.members) {
      const level = levelOf(s.meta, main);
      if (!cells.has(level)) cells.set(level, { level, name: levelName(main, level), members: [] });
      cells.get(level).members.push(s.name);
    }
    const list = order[main] || [];
    return { key: st.key, label: st.label, meta: st.meta,
      cells: [...cells.values()].sort((a, b) => list.indexOf(a.level) - list.indexOf(b.level)) };
  });
}

// ---------------------------------------------------------------- statistics

export function describe(values) {
  const n = values.length;
  if (!n) return { n: 0, mean: null, sd: null };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : null;
  return { n, mean, sd };
}

/**
 * Welch's unequal-variance t-test, two-sided, as scipy.stats.ttest_ind(a, b,
 * equal_var=False) computes it: {t, df, p}. Null when either side has fewer
 * than two values or neither side varies.
 */
export function welch(a, b) {
  if (a.length < 2 || b.length < 2) return null;
  const da = describe(a), db = describe(b);
  const va = da.sd ** 2 / da.n, vb = db.sd ** 2 / db.n;
  const se2 = va + vb;
  if (!(se2 > 0)) return null;
  const t = (da.mean - db.mean) / Math.sqrt(se2);
  const df = se2 ** 2 / (va ** 2 / (da.n - 1) + vb ** 2 / (db.n - 1));
  return { t, df, p: tTwoSided(t, df) };
}

/** P(|T| >= |t|) for Student's t with df degrees of freedom. */
export function tTwoSided(t, df) {
  if (!Number.isFinite(t)) return 0;
  return betaInc(df / (df + t * t), df / 2, 0.5);
}

const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** The regularized incomplete beta function I_x(a, b), by its continued fraction. */
export function betaInc(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? front * betaFraction(x, a, b) / a : 1 - front * betaFraction(1 - x, b, a) / b;
}

function betaFraction(x, a, b) {
  const tiny = 1e-300;
  const clamp = (v) => (Math.abs(v) < tiny ? tiny : v);
  let c = 1, d = 1 / clamp(1 - (a + b) * x / (a + 1)), h = d;
  for (let m = 1; m <= 1000; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2));
    d = 1 / clamp(1 + aa * d); c = clamp(1 + aa / c); h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2));
    d = 1 / clamp(1 + aa * d); c = clamp(1 + aa / c);
    const step = d * c;
    h *= step;
    if (Math.abs(step - 1) < 1e-15) break;
  }
  return h;
}

/**
 * The Welch tests one measurement is given, in the order the page lists them.
 *
 * Each factor picked is compared level against level, holding every other
 * picked factor fixed. The main comparison is also made with the splits
 * pooled -- all sexes together, say -- except age, which is never pooled:
 * it moves every measurement along the tooth more than anything compared
 * here does. Levels that are not set are shown but never tested.
 *
 *   valueOf(name) -> number | null
 *
 * Returns [{factor, a, b, within, pooled, na, nb, t, df, p}], where the test
 * is b against a and a is the earlier level (the reference, e.g. WT).
 */
export function comparisons(samples, by, order, valueOf) {
  const measured = samples.map((s) => ({ ...s, value: valueOf(s.name) })).filter((s) => s.value != null && Number.isFinite(s.value));
  const out = [];
  const testWithin = (key, holdKeys, pooledKeys) => {
    const groups = new Map();
    for (const s of measured) {
      const k = keyOf(s.meta, holdKeys);
      if (!groups.has(k)) groups.set(k, { meta: s.meta, members: [] });
      groups.get(k).members.push(s);
    }
    const rows = [];
    for (const g of [...groups.values()].sort((x, y) => byOrder(holdKeys, order)(x.meta, y.meta))) {
      // Pooling that pools nothing repeats a stratum's own test.
      if (pooledKeys.length && new Set(g.members.map((s) => keyOf(s.meta, pooledKeys))).size < 2) continue;
      const levels = (order[key] || []).filter((lv) => lv && g.members.some((s) => levelOf(s.meta, key) === lv));
      for (let i = 0; i < levels.length; i++) {
        for (let j = i + 1; j < levels.length; j++) {
          const a = g.members.filter((s) => levelOf(s.meta, key) === levels[i]).map((s) => s.value);
          const b = g.members.filter((s) => levelOf(s.meta, key) === levels[j]).map((s) => s.value);
          const res = welch(b, a);
          if (!res) continue;
          const parts = holdKeys.map((k) => levelName(k, levelOf(g.meta, k)));
          if (pooledKeys.length) parts.push(`${pooledKeys.map((k) => factor(k)?.plural || k).join(" and ")} pooled`);
          rows.push({ factor: key, a: levelName(key, levels[i]), b: levelName(key, levels[j]),
            within: parts.join(", "), pooled: pooledKeys.length > 0, na: a.length, nb: b.length, ...res });
        }
      }
    }
    return rows;
  };
  by.forEach((key, i) => {
    const rest = by.filter((_, j) => j !== i);
    if (i === 0) {
      const pooled = rest.filter((k) => k !== "age");
      if (pooled.length) out.push(...testWithin(key, rest.filter((k) => k === "age"), pooled));
    }
    out.push(...testWithin(key, rest, []));
  });
  return out;
}
