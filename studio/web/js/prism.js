// The results page's measurements as GraphPad Prism tables: a .pzfx project
// Prism opens directly, and CSV or tab-separated text laid out as Prism's
// data tables are (one column per group, one row per specimen), to import or
// paste. No DOM here: the tests run this module under Node and read what it
// writes with the R pzfx package, which is built from Prism's own files.
//
// Every table holds each specimen's own value, never a summary, so Prism
// computes the means, SDs and tests itself:
//
//   column   one measurement, no splits: a Column table, a column per group
//   grouped  one measurement split by further details: a Grouped table, a row
//            per stratum ("Female, P22"), a data set per group, a subcolumn
//            per specimen (for two-way ANOVA or multiple t tests)
//   xy       a profile along the incisor: an XY table, X the position along
//            the tooth (%), a data set per group, a subcolumn per specimen

// Prism keeps a table with more replicates than this as a HugeTable.
const MAX_TABLE_REPLICATES = 52;

// ---------------------------------------------------------------- tables from the page's figures

/** The Prism table behind one figure spec (see figure.js for the spec's shape). */
export function tableOf(spec) {
  if (spec.kind === "xy") {
    return {
      type: "xy", title: spec.title, xTitle: spec.xLabel, x: spec.x,
      datasets: spec.replicates.map((d) => ({ title: d.title, subcolumns: d.specimens })),
    };
  }
  if (!spec.splits.length) {
    return {
      type: "column", title: spec.title,
      columns: (spec.strata[0]?.cells || []).map((c) => ({ title: c.name, values: c.values })),
    };
  }
  return {
    type: "grouped", title: spec.title, rows: spec.strata.map((st) => st.label),
    datasets: spec.levels.map((lv) => ({
      title: lv.name,
      cells: spec.strata.map((st) => st.cells.find((c) => c.level === lv.level)?.values || []),
    })),
  };
}

/** A grouped table as a column table: one column per group in each stratum ("WT, Female, P22"). */
export function flatten(table) {
  if (table.type !== "grouped") return table;
  const columns = [];
  table.rows.forEach((row, r) => {
    for (const d of table.datasets) {
      if (d.cells[r].length) columns.push({ title: row ? `${d.title}, ${row}` : d.title, values: d.cells[r] });
    }
  });
  return { type: "column", title: table.title, columns };
}

// ---------------------------------------------------------------- numbers

/** A value as Prism and a spreadsheet read it: plain decimals, never exponent notation. */
export function num(v) {
  if (v == null || !Number.isFinite(v)) return "";
  if (Number.isInteger(v)) return String(v);
  const p = Number(v.toPrecision(10));
  if (Math.abs(p) >= 1e-4) return String(p);
  return p.toFixed(14).replace(/0+$/, "").replace(/\.$/, "");
}

// Decimals Prism shows a column with: about four significant figures of its largest value.
function decimals(values) {
  const big = Math.max(0, ...values.filter((v) => v != null && Number.isFinite(v)).map(Math.abs));
  if (!big) return 0;
  return Math.max(0, Math.min(6, 3 - Math.floor(Math.log10(big))));
}

// ---------------------------------------------------------------- .pzfx

const esc = (text) => String(text ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const cell = (v) => `<d>${num(v)}</d>`;
const textCell = (t) => `<d>${esc(t)}</d>`;
const subcolumn = (cells) => `<Subcolumn>\n${cells.join("\n")}\n</Subcolumn>`;

function replicatesOf(table) {
  if (table.type === "grouped") return Math.max(1, ...table.datasets.flatMap((d) => d.cells.map((c) => c.length)));
  if (table.type === "xy") return Math.max(1, ...table.datasets.map((d) => d.subcolumns.length));
  return 1;
}

function tableXml(table, id) {
  const reps = replicatesOf(table);
  const tag = reps > MAX_TABLE_REPLICATES ? "HugeTable" : "Table";
  const lines = [];
  if (table.type === "column") {
    lines.push(`<${tag} ID="${id}" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">`);
    lines.push(`<Title>${esc(table.title)}</Title>`);
    lines.push(`<RowTitlesColumn Width="1">\n<Subcolumn></Subcolumn>\n</RowTitlesColumn>`);
    for (const c of table.columns) {
      const values = c.values.map((v) => v.value);
      lines.push(`<YColumn Width="89" Decimals="${decimals(values)}" Subcolumns="1">`, `<Title>${esc(c.title)}</Title>`,
        subcolumn(values.map(cell)), "</YColumn>");
    }
  } else if (table.type === "grouped") {
    lines.push(`<${tag} ID="${id}" XFormat="none" YFormat="replicates" Replicates="${reps}" TableType="TwoWay" EVFormat="AsteriskAfterNumber">`);
    lines.push(`<Title>${esc(table.title)}</Title>`);
    lines.push(`<RowTitlesColumn Width="120">\n${subcolumn(table.rows.map(textCell))}\n</RowTitlesColumn>`);
    for (const d of table.datasets) {
      const values = d.cells.flat().map((v) => v.value);
      lines.push(`<YColumn Width="${89 * reps}" Decimals="${decimals(values)}" Subcolumns="${reps}">`, `<Title>${esc(d.title)}</Title>`);
      // Subcolumn j holds each stratum's j-th specimen of this group.
      for (let j = 0; j < reps; j++) lines.push(subcolumn(d.cells.map((c) => cell(c[j]?.value))));
      lines.push("</YColumn>");
    }
  } else if (table.type === "xy") {
    lines.push(`<${tag} ID="${id}" XFormat="numbers" YFormat="replicates" Replicates="${reps}" TableType="XY" EVFormat="AsteriskAfterNumber">`);
    lines.push(`<Title>${esc(table.title)}</Title>`);
    lines.push(`<XColumn Width="89" Decimals="${decimals(table.x)}" Subcolumns="1">`, `<Title>${esc(table.xTitle)}</Title>`,
      subcolumn(table.x.map(cell)), "</XColumn>");
    for (const d of table.datasets) {
      const values = d.subcolumns.flatMap((s) => s.values);
      lines.push(`<YColumn Width="${89 * reps}" Decimals="${decimals(values)}" Subcolumns="${reps}">`, `<Title>${esc(d.title)}</Title>`);
      for (let j = 0; j < reps; j++) lines.push(subcolumn(table.x.map((_, i) => cell(d.subcolumns[j]?.values[i]))));
      lines.push("</YColumn>");
    }
  } else {
    throw new Error(`No Prism table of type ${table.type}`);
  }
  lines.push(`</${tag}>`);
  return lines.join("\n");
}

/** Which specimen sits where in a table, a line per group: a Prism table has nowhere to keep names. */
export function specimenKey(table) {
  const names = (values) => values.map((v) => v.label).join(", ");
  if (table.type === "column") return table.columns.map((c) => `${c.title}: ${names(c.values)}`);
  if (table.type === "grouped") {
    return table.rows.flatMap((row, r) => table.datasets.filter((d) => d.cells[r].length)
      .map((d) => `${d.title}, ${row}: ${names(d.cells[r])}`));
  }
  return table.datasets.map((d) => `${d.title}: ${d.subcolumns.map((s) => s.label).join(", ")}`);
}

const isoDate = (when) => `${when.getFullYear()}-${String(when.getMonth() + 1).padStart(2, "0")}-${String(when.getDate()).padStart(2, "0")}`;

/**
 * A Prism project (.pzfx) holding `tables`, one data table each, in order.
 * `notes` go on the project's info sheet, followed by where each specimen
 * sits in each table. The layout is Prism 6's XML, which every later Prism
 * opens.
 */
export function pzfx(tables, { notes = [], when = new Date() } = {}) {
  const lines = [...notes, ""];
  for (const t of tables) lines.push(t.title, ...specimenKey(t).map((l) => `    ${l}`), "");
  const stamp = when.toISOString().replace(/\.\d+Z$/, "+00:00");
  const out = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<GraphPadPrismFile PrismXMLVersion="5.00">',
    "<Created>",
    `<OriginalVersion CreatedByProgram="GraphPad Prism" CreatedByVersion="6.0f.254" Login="" DateTime="${stamp}"></OriginalVersion>`,
    "</Created>",
    "<InfoSequence>", '<Ref ID="Info0" Selected="1"></Ref>', "</InfoSequence>",
    '<Info ID="Info0">',
    "<Title>MicroCT Studio export</Title>",
    `<Notes>\n<Font Color="#000000" Face="Helvetica">\n${lines.map((l) => `${esc(l)}<BR/>`).join("\n")}\n</Font>\n</Notes>`,
    `<Constant><Name>Experiment Date</Name><Value>${isoDate(when)}</Value></Constant>`,
    ...["Experiment ID", "Notebook ID", "Project", "Experimenter", "Protocol"]
      .map((k) => `<Constant><Name>${k}</Name><Value></Value></Constant>`),
    "</Info>",
    '<TableSequence Selected="1">',
    ...tables.map((_, i) => `<Ref ID="Table${i}"${i ? "" : ' Selected="1"'}></Ref>`),
    "</TableSequence>",
    ...tables.map((t, i) => tableXml(t, `Table${i}`)),
    "</GraphPadPrismFile>",
    "",
  ];
  return out.join("\n");
}

// ---------------------------------------------------------------- CSV and pasting

// A table as rows of cells: the header, then one row per specimen (column,
// grouped) or per position (xy). Grouped tables are flattened, since a flat
// file has no subcolumns.
function grid(table) {
  const t = flatten(table);
  if (t.type === "xy") {
    const cols = t.datasets.flatMap((d) => d.subcolumns.map((s) => ({ title: `${d.title}: ${s.label}`, values: s.values })));
    return [[t.xTitle, ...cols.map((c) => c.title)], ...t.x.map((x, i) => [num(x), ...cols.map((c) => num(c.values[i]))])];
  }
  const rows = Math.max(0, ...t.columns.map((c) => c.values.length));
  return [t.columns.map((c) => c.title),
    ...Array.from({ length: rows }, (_, i) => t.columns.map((c) => num(c.values[i]?.value)))];
}

const quote = (v) => (/[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);

/** The table as CSV: titles in the first row, a column per group, blanks where a group has fewer specimens. */
export function csv(table) {
  return grid(table).map((r) => r.map(quote).join(",")).join("\n") + "\n";
}

/** The table as tab-separated text, to paste into a Prism data table from its first title cell. */
export function tsv(table) {
  return grid(table).map((r) => r.map((v) => v.replace(/[\t\r\n]+/g, " ")).join("\t")).join("\n") + "\n";
}

/** Every value in `tables` with its specimen, one row each, for the tables with one value per specimen. */
export function longCsv(tables) {
  const rows = [["table", "group", "stratum", "specimen", "value"]];
  for (const t of tables) {
    if (t.type === "column") for (const c of t.columns) for (const v of c.values) rows.push([t.title, c.title, "", v.label, num(v.value)]);
    if (t.type === "grouped") {
      t.rows.forEach((row, r) => {
        for (const d of t.datasets) for (const v of d.cells[r]) rows.push([t.title, d.title, row, v.label, num(v.value)]);
      });
    }
  }
  return rows.map((r) => r.map(quote).join(",")).join("\n") + "\n";
}
