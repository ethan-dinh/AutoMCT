# MicroCT Studio

A local web app that runs this whole repository from one place: import scans,
crop, convert, segment, straighten the incisor, measure incisor and molar
enamel, pick landmarks, look at every volume as an adjustable heatmap, and
compare groups in charts.

```bash
conda activate microct-analysis
python -m studio                        # opens http://localhost:8765 and asks for a workspace
python -m studio --last                 # reopen the last workspace without asking
python -m studio --workspace "/Volumes/LabDrive/MicroCT-Studio" --port 8800 --no-browser
```

Nothing to install: the server is the Python standard library and the UI is
plain JavaScript modules served by it, so it runs in the existing
`microct-analysis` environment and works offline.

---

## What it does

| Page | For |
|---|---|
| **Library** | Every specimen, its group, and a track of how far it is through the pipeline, sorted into folders of your own. Select several to run steps in bulk, move them to a folder, flag them out of the results, or show them alone in Results. |
| **Import** | Drop a scan (ISQ, DICOM or BMP folder, NRRD/TIFF), or point at one on disk; its details arrive filled in from its header, its name and its folders. The studio names it, files it and, if asked, runs the pipeline. **Scan folders** imports a whole existing dataset (the Z drive's `SEG DATA`, `CMPR`, `RESULTS`) in one go, matching files to specimens by name, as a sheet of every specimen's details. |
| **Specimen** | The pipeline as numbered steps, each with its options, live progress and errors; the molar and incisor results with charts; views and files. |
| **Viewer** | The library's folders, as cards with a preview or as a list, to choose a specimen from. Any volume as a heatmap. Slices in three planes, thick slabs, maximum and mean projections, the incisor unrolled into one image, enamel/dentin ratio colouring, structure overlays, and landmark picking. |
| **Compare** | Up to four specimens side by side in 3-D, lined up on the first and turning together, on one heatmap scale in mg HA/cm³. |
| **Crop** | Draw one 3-D box on three projections of the raw scan; it is cut out before anything else runs (by the converter, or for an NRRD, as the segmentation loads it). |
| **Results** | Molar enamel, incisor mineralization curves and structure volumes, compared by whichever details you pick (KO against WT, split by sex and age, say), in projects of your own or over every specimen, with Welch t-tests and significance stars, CSV exports (including the R report's specimen list), and every chart as a Prism-style figure at a set DPI with its data as Prism tables. Every colour and every chart's title can be changed. |
| **Queue** | Running and finished jobs, with live logs. |
| **Tools** | RSQ motion check, density histogram, DICOM to TIFF, model training (turned off for now), the test suites. |

### The pipeline

```
Import -> Convert -> Segment -> Straighten incisor -> Incisor enamel
   (not for NRRD)            |                    \-> Landmarks (with the segmentation)
                             \-> Molar enamel
                             \-> ML segmentation (turned off for now)
```

| Step | Runs | Writes, in the specimen's folder |
|---|---|---|
| Convert | ISQ, DICOM, BMP and TIFF scans, through `tools/volume_pipeline.py` and the ISQ/DICOM readers | `converted/<name>.nrrd`, or beside a linked scan |
| Segment | `segment_mandible` pipeline, without its prompts | `segmentation/<name>/…` (the pipeline's own layout); the reoriented mandible beside a linked scan |
| ML segmentation | `ML-Segmentation/src/predict.py` (turned off for now) | `ml/masks/` |
| Straighten incisor | `longitudinal_incisor_analysis/cmpr.py` | `cmpr/<name>_resliced_volume.nrrd` + transform |
| Incisor enamel | `longitudinal_incisor_analysis/enamel_density.py` (molar dentin baseline) | `quant/incisor/` |
| Landmarks | `longitudinal_incisor_analysis/detect_landmarks.py`: eruption and M1-M3 found from the segmentation, marks set by hand kept, all mapped onto the trajectory | `landmarks/` |
| Molar enamel | `quantify_molar/molar_enamel.py` | `quant/molar/` |

**Which scans are converted.** An NRRD is already what the segmentation reads,
so it skips Convert and goes straight to Segment, which cuts out any crop box
as it loads the scan. Every other format (ISQ, DICOM, BMP, TIFF) is converted
to NRRD first, and the converter applies the crop box. Moving the box later
marks whichever step applied it as out of date.

**Enamel density in mg HA/cm³.** Molar enamel converts the enamel's mean and
its lower and upper percentile (5th/95th by default) with the HA calibration a
Scanco scanner writes into every ISQ. It looks for that calibration in the
conversion's `<name>_meta.json`, then in the ISQ itself, or for an NRRD in the
`<stem>_meta.json` beside it. For a scan that carries none, enter the slope and
intercept under the step's Density calibration settings. The raw intensities
are always reported as well.

The incisor profile is drawn in mg HA/cm³ the same way, beside the
enamel/dentin ratio, with the same calibration as the specimen's molar: the
molar step's once it has run, otherwise the one entered in its settings or
found in the scan's files. A switch shows raw intensity instead, and the
dentin line (the dentin peak fitted in each slice) can be drawn beside the
enamel. The incisor export carries the enamel and dentin in both units.

**The incisor curves are cleaned as the R report cleans them**
(`longitudinal_incisor_analysis/R/longitudinal_analysis.rmd`, sections 2 to 6),
after `enamel_density.py`'s own onset trim. Only slices whose fit resolved enamel
from dentin, or found the two coincident, are used; a slice pooled over more
than 12 slices either side, or a `resolved_lo` slice whose low component is not
solid enough to be secretory matrix (below 0.10), is dropped. A slice more than
0.08 from its own specimen's trend is left out as an outlier. The trend is the
report's GAM, a 40-basis thin-plate spline fitted by REML, refitted until the
outliers stop changing. The kept slices get a 21-slice running median that is
left empty at either end and wherever its window would straddle missing slices,
so a curve shows its gaps rather than bridging them, and a group's curve starts
where its specimens measure half their slices. The crossover (where the ratio
last rises through 1.0), the mature plateau (its mean 70–95% along the tooth)
and the peak are read off each specimen's cleaned curve. The plateau and peak
are also read off the enamel density's running median, in mg HA/cm³ (with the
scan's HA calibration) and as stored; the Results cards switch between the
three units, and their tables list all three beside the dentin reference, the
ratio's denominator. A specimen's page
draws its outliers in grey. The incisor export marks every slice `kept` or
`outlier`, with its `residual` from the trend and the running medians
(`ratio_med`, `density_med`). `trajectory.py` holds the port, and
`test_trajectory.py` runs the report's own R code beside it when R is installed.
A profile written by an older build of the incisor script is flagged, so it
can be re-run: one from before the onset trim, whose pre-enamel slices still
carry a ratio, or one whose run record predates the script's withdrawal of the
dentin readings from the enamel's opening stretch (`OPENING_RISE`).

**Comparing groups.** The Results page compares specimens by any of their
details. The first one picked (Genotype, say) is the main comparison: its
values sit side by side, one colour each. Up to two more (Sex, Age) split the
specimens into clusters of columns and panels of curves, with the main
comparison repeated in each. Each measurement gets its group means and SDs, and
Welch t-tests: the main comparison within every split, then with the splits
pooled. Age is never pooled, as in the R report, and the page warns when the
groups shown mix ages. Tests between the split levels themselves (female
against male within WT) sit folded under the main ones. The tests are
exploratory, with no correction for multiple tests. The browser runs them in
`web/js/compare.js`, which the tests hold to SciPy's numbers. Chips under the
picker hide any value of any detail, and the page remembers the comparison per
workspace.

**Projects.** The Results page opens on its projects: **All specimens**, then
a card for each project with how many specimens it holds, their groups and how
it compares them, and **New project**. A project is a named set of specimens
with its own comparison and filters, so a study's results are one click away.
A new one starts with the specimens in a library folder (and its folders),
every specimen, or none, ticked next in a checklist; on a project's page
(**Save as a project**, from All specimens) it can also start with the
specimens shown. **Edit specimens** (**Specimens…** on its card) ticks any
specimen of the workspace in or out; a specimen added to the library later
joins a project only once ticked there. **Rename**, **Duplicate** (the same
specimens compared another way, say) and **Remove** are on the card and on
the project's page; removing one changes no specimen, result or flag. Projects
are kept in the workspace (`studio.json`, `results_projects`), so every
browser that opens it has them and sees each other's edits; a renamed
specimen stays in its projects, and a link (`#/results/<id>`) opens one. The
card last opened is marked. The Prism export names its zip and notes after the
project. The colours and chart titles are the workspace's, shared by every
project.

**Filters.** Above the charts, one line sums up the filters: what is compared,
how many of the project's specimens are shown and which values are hidden;
**Filters** opens the controls under it (the browser remembers it open or
shut), and **Clear filters** shows every specimen of the project again. The
filters reach only the project's specimens: the comparison offers the details
that vary among them, the chips their values with their counts, and the
specimens button (**3 of 4 specimens**) a checklist of the project's
specimens alone, in the page's groups, with each one's library folder and any
flag. Tick the ones to show, a whole group from its heading, or search by name,
detail or folder ("KO P52", "Batch 2"; each word must start a word, so "male"
leaves out the females) and tick, untick or keep **Only** what the search
found. A project's filters are saved with it; those of All specimens last as
long as the browser tab. The charts, statistics, Prism exports and CSV
downloads take the specimens shown (each CSV still leaves out the ones flagged
out of its analysis). The warning that ages are pooled stays in sight with the
filters shut. Select specimens in the Library and **Show in Results** opens All
specimens with those alone.

**Figures and data for Prism.** Every chart on the Results page has an
**Export** button. It opens the chart redrawn as GraphPad Prism draws one, at
the size it will be printed: white ground, black axes on the left and bottom
only, ticks pointing out, no grid, Arial. A strip plot becomes Prism's column
scatter (each specimen a point, the mean a line, the SD or SEM a capped bar),
or bars at the mean with the points on them; split by further details, it is
Prism's grouped layout, a cluster per stratum with the groups in a legend.
Significance stars (GraphPad's: * p < 0.05, ** < 0.01, *** < 0.001, **** <
0.0001, ns otherwise) sit on brackets over each pair of groups compared within
a cluster, from the page's own Welch tests; P values instead, ns left off, or
none, are one click away. The curves keep their ±1 SD bands, the specimens'
own curves and the landmark bands, a panel per stratum on one scale. Set the
width and height in inches (1, 1.5 or 2 journal columns in a click), the
resolution in DPI (300 for colour, 600 for fine lines, 1200 for line art), the
font size, and the page's colours or Prism's black with a symbol per group;
the preview is the saved file. **PNG** and **TIFF** record their DPI, so Word,
Illustrator and journal checkers read the intended print size; **SVG** stays
vector for Illustrator or Inkscape, its text editable. The options are
remembered in the browser, and each chart's size with it.

The same dialog saves the chart's data for Prism: **Prism file (.pzfx)**, which
Prism opens as a project; **CSV**; or **Copy table**, to paste into a Prism data
table from its first title cell. Every value is one specimen's own, never a
summary, so Prism computes its own means and tests. An unsplit measurement is a
Column table (a group per column); a split one a Grouped table (a stratum per
row, a group per data set, a specimen per subcolumn: ready for two-way ANOVA or
multiple t tests); a curve an XY table (the position along the incisor as X at
every 1%, a specimen per subcolumn, with the dentin line as further data sets).
A curve's values are the ones the page's means are taken over, so Prism's mean
± SD redraws the page's. Prism tables have nowhere for specimen names, so the
project's info sheet lists which specimen sits where. **Export for Prism**, at
the top of the page, does everything at once: every chart as a figure (each at
its own remembered size, in one format and DPI), one Prism project with a table
for every measurement under every switch (each density unit and statistic,
every volume and thickness for the row and each molar, the incisor curves in
both units, the incisor metrics and the structure volumes), the same tables as
CSVs, and `all_values_long.csv` with every value beside its specimen, zipped
with a README. Everything follows the page: its comparison, the specimens
ticked, hidden values and flags. The page's own t-test tables carry the stars too.

**Colours and titles.** **Colours**, at the top of the Results page, picks the
colour of every part of the charts: each group of the main comparison, each
landmark (M1, M2, M3, eruption), the mean lines, error bars, specimen curves,
±1 SD bands, dentin line, reference line and significance brackets, the axes,
axis titles, tick labels and legend, the page's grid lines and a figure's
background and title. The charts change as a swatch moves, and each colour has
a Reset back to its default. Unset, the page keeps its theme's colours and a
figure prints black on white. Every chart's title can be renamed with the
pencil beside it (or a double-click); empty it for the original. The export
dialog edits the same title, sets whether it is drawn above the figure, renames
the figure's axis titles, and holds the same colour picker beside a live
preview. A renamed title also names the chart's files. The colours and titles
are kept in the workspace (`studio.json`), so every browser that opens it
draws the same figures. An axis title is renamed only where its original
wording appears, so switching to other units never keeps a title that no
longer fits.

**Landmarks on the curves.** Where a panel's specimens share one age, its
incisor curves carry the landmarks, each in its own colour (M1 green, M2
violet, M3 magenta, eruption yellow: categorical slots clear of the first ones
the compared groups take). A line marks each landmark's mean start and end
over the panel's specimens, in a lighter strip one standard deviation either
side of it, as the curves' ribbons are one standard deviation either side of
theirs; the band is in full colour between the strips. So a strip's width is
how far that end moves between specimens, and where one molar gives way to
the next, both strips show. Eruption is a line in its own strip. Pointing at a band gives its means ± SD.

**Comparing specimens in 3-D.** The Compare page shows up to four specimens
side by side, each as the viewer's 3-D mode draws it, from one of their views
(the molars, say). One colour window, surface level and colour scale cover
them all, set in mg HA/cm³: each specimen's stored values are converted with
its own HA calibration, the one its molar results use, so a colour is one
density in every panel, even between scans made with other scanner settings.
A scan with no calibration is left blank on that scale; **Raw intensity**
shows every specimen as stored, which compares only scans made with one
setting. The histogram draws each specimen's values on the shared scale in its
own colour, with its enamel threshold ticked on the molar view. Dragging any
panel turns them all, all the way round in any direction; Shift-drag moves
them, ⌘-drag (Ctrl-drag) rolls them, and scrolling zooms. Untick **Turn
together** to turn each on its own.

**Lining specimens up.** The segmentation's reorientation only permutes and
flips axes, so each scan keeps whatever tilt it had in the holder. Every
specimen turns about its own tissue's centre, not its scan box's, and under
**Place them: Fitted to the first** each one added is turned and moved to where
its surface lies closest to the first one's (`web/js/pose.js`): tried as
scanned and from every way of laying its principal axes on the first's, then
refined, with each surface distance capped so a part one specimen lacks (a
ramus cropped off, say) does not pull the fit. It is rigid, never scaled, so
sizes stay comparable, and the Specimens list says how far each was turned
and moved. **As scanned** leaves every one as it sits. To finish by hand, hold
Option (Alt) with any drag, or press a panel's move button, and the drag turns,
moves or rolls that specimen alone; **Line them up again** undoes it.

**Landmarks as planes.** Under **Landmarks**, each specimen's own eruption
and M1-M3 are drawn as planes in their colours (eruption yellow, M1 green, M2
violet, M3 magenta): a see-through sheet across its scan at the slice each was
marked on, a molar's where it starts and where it ends, with its rim drawn and
a line in its colour where it cuts the surface. Tissue in front of a plane
hides it, and the planes turn and move with their specimen. The marks sit in
the segmented scan's slices, so the hemimandible, incisor, molar and bone
views show them as marked; on the straightened incisor each is the
cross-section its mark maps to along the tooth. Each landmark can be turned off,
and **Sheets** sets how much a sheet covers (at none, only rims and cut lines
show). The legend and the saved image name the planes drawn, and a landmark
past either end of a view's preview is listed rather than drawn.

The surface's colour is read at one depth beneath it in every panel, and the
page warns when the previews differ in resolution, since a coarser one
averages thin enamel with what surrounds it. **Save image** writes the panels,
named, over the colour bar as one PNG. Select specimens in the Library and
**Compare in 3-D** opens them here; the address holds the view and the
specimens, so a comparison can be reloaded or sent on.

Fine differences are about as small as a voxel's noise: about 45 mg HA/cm³
per 6 µm voxel in molar enamel, 25 over three. So the colour is averaged over a
little tissue straight in from the surface (12 µm around by default; **One
point** reads a single voxel), and cut into **Steps** at round densities --
about 6, 12 (the default), 24, 48 or 96 across the window, every 5, 10, 25 ...
mg HA/cm³, or any width typed in -- each step one colour, with no lines drawn on
the surface. **Rainbow** gives every step its own hue, blue for the least dense
to red for the most, so even fine steps tell apart; its lightness does not
follow density, so read it by the colour bar. Picking it with steps off turns
them on. **Random** follows no gradient at all: each step takes a colour
unrelated to its neighbours' (the most distinct the screen has, up to 32, then
round again), so the pattern the steps make -- which density lies where --
stands out rather than blending; grey is outside the window, and **Shuffle
colours** deals them again. Unrelated colours also make every speck of noise
that crosses a step's edge stand out, so **Smooth the scan** (3-D) blurs the
scan in 3-D first, a Gaussian of 6, 12 or 18 µm (sigma) worked on the GPU, the
background kept out of the mean so a surface keeps its place. Random arrives
with it at 12 µm: its steps then join into bands, like contours on a map. **Above
/ below** reads the enamel against one density, blue below it and red above:
a specimen's enamel mean, or their average. On the molars the page opens on
**Enamel core** -- every colour spent on the middle of the enamel, the lowest
5th to the highest 95th percentile among them -- with the surface on the
enamel at the lowest enamel threshold among them; **All tissue** shows the
whole tooth. **Shading** turns the light down so colours stay true to the
scale, and the 3-D texture holds 16-bit values, so a narrow window shows no
steps of its own.

**The molar enamel results** come in three cards, each a strip plot with its
group statistics beside it. **Enamel density** switches between mg HA/cm³, raw
intensity and the enamel/dentin ratio (the median enamel over the dentin peak,
which needs no calibration), and between the mean and the lower and upper
percentiles; its table adds each specimen's dentin peak. **Volume** switches
between enamel share, enamel, dentin and whole-tooth volume, for the molar row
or for M1, M2 or M3 alone. **Enamel thickness** shows the median thickness, its
mean, its SD across the enamel's own measured points (how uneven each
specimen's enamel is, `thickness_mean_um` and `thickness_sd_um` in the molar
export, for the row and per molar), the enamel–dentin junction area, or the
thickness divided by that area (µm per mm², `thickness_um_per_mm2`), or the
mean thickness divided by the square root of that area
(`thickness_mean_per_sqrt_junction`), once Molar enamel has measured them.
The per-mm² ratio still depends on tooth size: area grows with the square of
size and thickness only in proportion, so a tooth of the same shape but 10%
smaller reads 11% higher. The mean over √area does not: both are in µm, so it
has no unit and is shown as a percentage, the enamel's mean thickness as a
share of the junction's linear size. A tooth of the same shape at any size
reads the same, so it shows how thickness varies against the size of the
junction. For the row, √area is of the three junctions together, which is no
one tooth's size; M1, M2 or M3 alone give the per-tooth proportion. It needs
no rerun: summaries from version 6 already hold the mean and the area. A specimen's own page groups its molar numbers the
same way.

**Flagging a specimen.** A scan with motion, a broken incisor tip or the wrong
genotype can be flagged, from the Library (several at once) or from its own
page, with a note saying why. A flag leaves the specimen out of every analysis,
or only the ones ticked: a broken incisor tip can leave the molars measured.
The specimen stays in the library, marked Excluded, in a **Flagged** section
below the other specimens with its note, and nothing is deleted.
The Results page leaves it out of its charts, statistics and counts, and lists
every flagged specimen at the end under **Flagged and excluded**, with its note,
what it is left out of and when it was flagged. From there a flag can be
edited, or the specimen included again. The exports follow the flags: the
molar, incisor, landmark and segmentation CSVs leave flagged specimens out; the
metadata export keeps every specimen, with `excluded_from`, `exclusion_note`
and `flagged_on`; **Download notes** gives the flagged ones alone; and the R
specimen list keeps a specimen flagged out of the incisor as a commented-out
line with its note. The flag is saved in the specimen's `sample.json` as
`excluded`.

**Folders in the library.** Specimens can be sorted into folders, by study,
batch or scanner, and folders nest ("Odam study › Batch 2"). **New folder**
makes one inside the folder open; a folder row opens it, and the path above
the title leads back up. Specimens move by dragging their rows (a selection
goes together) onto a folder row or a level of that path, with **Move to…**
on the selection, or with **Move…** on a specimen's own page; folders move the
same way, by dragging or with **Move…**. **Rename** and **Remove folder** act
on the folder open; removing one moves what it held up a level and removes no
specimen. **Import** from inside a folder files the new specimens there, and
the import form has a Folder field for choosing another. A search or filter
looks through the folder open and every folder inside it, saying which folder
each match is in, and **Run missing steps** covers the same. The Viewer
chooses a specimen through the same folders, the path above leading back up
and its search reaching into the folders inside, as cards with a preview or
as a list (**Grid** / **List**, remembered in the browser); a specimen open
in the viewer names its folder at the top, a link back to the others in it. The folders are
the library's only: nothing moves on disk, a specimen's files stay in
`samples/<name>/`, and filing one elsewhere leaves its Updated time alone. Its
folder is saved in its `sample.json` as `folder` (a path such as
`Odam study/Batch 2`), and the folders themselves in `studio.json`, so an empty
one stays until it is removed.

**Machine learning is turned off for now.** The ML segmentation step and model
training stay on screen, greyed out, and the server refuses to run them. Set
`ML_OFF = None` in `steps.py` to bring them back.

Every step is the repository's own script, called with the options shown in
its form. Nothing is re-implemented, so a fix to a tool reaches the studio
immediately.

**Conversion follows the lab's recipe** unless its options say otherwise:

```bash
python convert_ISQ_NRRD.py <scan> --autocrop --autocrop-margin-mm 0.1 --target-voxel-um 6 \
    --gauss-sigma 0.15 --nlm-h 0.3 --median-radius 0 --aniso-iterations 10 --aniso-kappa 600 \
    --tv-weight 0.15 --unsharp-amount 0.04 --reorient --reorient-tight --reorient-margin-mm 0.5 \
    --save-raw --save-dtype int16 --clamp-negative
```

Cropping to the specimen and cutting the rotated volume tight to it is what
keeps a converted scan small; without them a whole field of view
(3168 × 3168 × 412 voxels) is 8 GB. A drawn crop box replaces the automatic
crop, and **Whole scan** on the crop page keeps everything. The unfiltered copy
(`<name>_prefiltered.nrrd`) sits beside the converted scan, and a later
conversion that changes only the filters starts from it; one that changes the
crop, voxel size or rotation reads the scan again. Change the recipe for one
run in the Convert step's options, or for the whole workspace under Settings,
Default options.

### Filling in the details

Every specimen carries the same details: age, mouse line, genotype, sex and
replicate (which make up its name, `{Age}_{BaseMouse}_{Gene}_{Sex}_{N}`),
side, and the study, scan date, scanner, voxel size and notes. The import
forms arrive filled in with what the scans say:

| From | Fills in |
|---|---|
| A name in the lab's convention (`P52_Odam-Cre_WT_F_1`) | age, line, genotype, sex, replicate |
| The ISQ header | scan date; the file's own voxel size is shown, and left to the file |
| The DICOM header | scan date, scanner, sex, age |
| A SkyScan `.log` beside a BMP stack | voxel size (a BMP slice records none), scanner, scan date |
| The name typed at the scanner (ISQ, DICOM) | whatever it spells out, as a guess |
| The folders a scan sits in (`RAW/P52/…`) | an age, a date, and any line, genotype or study this workspace already uses, as guesses |

A guess is tinted until it is checked or edited, and hovering any filled-in
value says where it came from. A scan that records no voxel size (a BMP stack
without its log, a TIFF with no resolution) is ringed until one is set, since
its measurements would otherwise be wrong.

**Scan folders** shows what it found as a sheet, one row per specimen and one
column per detail:

- the **All** row sets a column for every selected specimen;
- dragging a cell's corner copies it up or down, ⌘D (Ctrl+D) copies the row
  above, and Enter moves down a row;
- cells pasted from Excel, Numbers or Google Sheets spread over the sheet.
  With their header row, columns are matched by name, and when a column holds
  scan or specimen names (`C0004019`, or just `4019`) each row goes to the
  specimen it names, in any order;
- every change can be undone.

Each row shows the name it will be imported under as the details change. A
specimen is named from its details once it has an age and a genotype, with
its sex as F or M when it is recorded; until then it keeps the name it was
found under, and a name typed in wins over both. Scanned again, specimens already in the library are listed apart, found
by their scans even when they were renamed on import. A single import follows
the same rule, and **Copy details from…** fills its empty fields from a
specimen already in the library.

## The workspace

All studio data lives in one folder, the workspace. The studio asks which one
to use each time it starts: it lists the workspaces opened on this computer
and any it finds on connected drives, and can open another folder or create a
new workspace. The × beside a workspace takes it off both lists on this
computer and leaves its folder alone; opening it again brings it back. Switch
later from Settings or by clicking the workspace name at the bottom of the
sidebar. Keep a workspace on a local or external drive, not
in a Box or Dropbox folder: previews and jobs write often.

```
<workspace>/
    studio.json                 settings, the colour each group was given, the library's folders
    samples/<name>/
        sample.json             metadata, where each file lives, step history
        raw/                    uploaded or copied raw data
        converted/  segmentation/  cmpr/  quant/  landmarks/  ml/
        preview/<view>/         downsampled copies the viewer streams
    jobs/                       one record and one log per job
    trash/                      removed specimens (recoverable)
```

Imported data can stay where it is: **linking** records the path instead of
copying. A linked scan's two large files, its conversion and its reoriented
mandible, are written beside it on its own drive, so an external drive holds
them rather than the internal one:

```
<scan's folder>/MicroCT-Studio/<name>/
    <name>.nrrd                     the conversion, and <name>_meta.json
    <name>_reoriented_volume.nrrd   the reoriented mandible
```

The specimen page shows that folder and each file's location, and renaming a
specimen renames it too. Everything smaller (the masked incisor, bone and molar
volumes, masks, CMPR, results, previews) stays in the workspace, so the later
steps and the viewer keep working with the drive unplugged. Removing a
specimen leaves its large files where they are. An uploaded or copied scan
already lives in the workspace, and its large files stay there with it; to
keep a big scan off the internal drive, import it with **From this computer**,
not by uploading. Names follow the
lab's `{Age}_{BaseMouse}_{Gene}_{Sex}_{N}` convention (`P52_Odam-Cre_WT_F_1`),
and a specimen's group is its genotype and age (`WT P52`), as in the R report.

The sex is written F or M and left out while it is not recorded, so names
from before it was part of them (`P52_Odam-Cre_WT_1`) still read correctly.
Replicates are numbered across the sexes: after `P52_Odam-Cre_WT_1` and
`P52_Odam-Cre_WT_F_2` the next male of the group is `P52_Odam-Cre_WT_M_3`, so
the number alone still tells a group's animals apart. A specimen already in
the library is not renamed on its own: **Edit details** offers the name its
details give, and applying it renames its files too.

### One workspace from a Mac and a PC

Put the workspace on an external drive and plug the drive into either
computer; the studio on each finds it in the list at startup.

- **Format the drive as exFAT.** Both systems read and write it. A Mac-only
  drive (APFS, Mac OS Extended) cannot be opened on a PC, NTFS cannot be
  written from a Mac without extra software, and FAT32 cannot hold the files
  over 4 GB that converted scans often are. The studio names each drive's
  format when you choose or create a workspace, and warns about these.
- **Paths are stored so they travel with the drive.** The same drive is
  `/Volumes/LabDrive` on a Mac and `E:\` (or another letter) on a PC. A
  specimen's files inside the workspace are recorded relative to it, and files
  on any other external drive by that drive's name plus their place on it, so
  they are found wherever that drive is mounted. Keep scans you link in place
  on the workspace's drive (or another external one), not on a computer's own
  disk, which the other computer cannot see.
- **Set up the PC once**: install the `microct-analysis` environment from
  `env/environment.yml`, get this repository, then `python -m studio`.
- Unplug the drive only when no job is running. If it disappears while the
  studio is open, the studio goes back to the workspace list until the drive
  returns.

## Built for speed

- **Jobs run in the background, in lanes.** Heavy steps (anything that loads a
  whole scan) run one at a time by default so two segmentations cannot exhaust
  memory; preview builds run beside them. A chain of steps is queued at once
  and each starts when the one it needs finishes. Change the lane sizes in
  Settings or on the Queue page.
- **Previews are built once.** After each step the studio writes a downsampled,
  16-bit copy of each new volume (cropped to its tissue, block-averaged to a
  voxel budget, gzipped). The viewer loads it in about a second, then every
  slice, projection and window change is computed in the browser. The
  intensity scale is exact: the viewer shows scanner units, and the histogram
  behind the colour window is computed from the full-resolution file.
- **Uploads start on drop** and stream straight to disk; importing from a path
  links instead of copying.
- Optional **denoise caching** in the Segment options skips the slowest stage
  on re-runs.

## Viewer keys

| Key | Does |
|---|---|
| ← → or , . | Previous / next slice (Shift: 10) |
| 1 2 3 (4) | Axial, coronal, sagittal (unrolled incisor) |
| M | Maximum projection on or off |
| R | Enamel/dentin ratio colouring (straightened incisor) |
| O | All structure overlays on or off |
| 0, + − | Fit, zoom (Ctrl-scroll zooms too; drag to pan) |
| Q, G H, J K, N W, U | Landmarks: eruption, M1/M2/M3 start and end, undo |

**Landmarks, found automatically.** The Landmarks step needs only the
segmentation and the straightened incisor, so it runs by itself: choose it in
the Library's step menu to find the landmarks of every selected specimen at
once, and "Run missing steps" includes it. It finds:

- **Eruption**, where the incisor's enamel comes out of the bone. The tooth
  is followed along its centerline, and in each cross-section rays radiate
  outward from its enamel (the labial band, the tooth's brightest tissue)
  looking for bone. Eruption is the first cross-section, walking toward the
  tip from the last one where bone covered all of the enamel, where no ray
  from the enamel touches bone, nor does for the next 0.09 mm. On
  P22_TGM2_KO_F_1 that is z 1615, six slices from the slice marked by eye
  (1621); on the 12 P22 TGM2 specimens with a usable scan it falls 424 ± 23
  slices past the front of M1's crown.
- **M1-M3**, each from where its crown starts to where it ends along the jaw,
  by the molar step's own tooth split (the molar step need not have run). Where
  an oblique contact lets two crowns overlap, the slices go to the tooth in
  front (M2 starts where M1's crown ends), as they are marked by eye. Placed
  only when all three teeth are found.

A landmark found wrongly is set by hand in the viewer (Adjust in the viewer
on the specimen's page): press its key on the right slice and Save and map,
which maps the marks as they are without finding them again. A mark set by
hand is kept every time the step runs again. The Landmarks panel tags each
mark "auto" or "by hand"; the button beside a hand mark, or "Back to the
found landmarks" for all of them, puts it back where the step found it.

Drag the edges of the histogram to set the heatmap's range; the strip under it
shows which colour each intensity gets. On the molar view the enamel threshold
and dentin peak are marked on it. Once the Molar enamel step has run, the
molar view also overlays the three teeth it split apart (M1 green, M2 violet,
M3 pink, the landmark colours); in 3-D they colour the teeth, so a wrong split
shows at a glance.

## Security

The server listens on `127.0.0.1` only, and it can run jobs and list files on
this machine, so it is meant for the person at the keyboard. It refuses
requests whose `Host` header is not its own (DNS rebinding) and changes coming
from any other origin (cross-site requests). Do not pass `--host 0.0.0.0` on a
shared network.

## Development

```bash
python -m pytest studio/tests          # workspace, previews, and HTTP end to end
```

`test_compare.py` runs the Results page's grouping and statistics under Node
and skips when Node is not installed; `test_figure_export.py` runs the figure
layout, the Prism tables and the PNG, TIFF and ZIP writers the same way, reads
what they write with Pillow, zipfile and ElementTree, and with the R pzfx
package where it is installed; `test_compare3d.py` does the same for the
Compare page's shared heatmap scale, holding it to one colour for one density
whatever each scan's calibration, and `test_pose.py` for the 3-D views' turning
and lining up, holding the fit to known turns of a phantom hemimandible. `test_trajectory.py` runs the R report's
cleanup code (R with mgcv, zoo, dplyr, readr and tibble) beside the port and
the pages' running median under Node, skipping either when it is missing.

| File | Role |
|---|---|
| `server.py` | HTTP API, uploads, event stream, static files |
| `workspace.py` | Sample records, naming, portable paths, recognizing pipeline outputs, bulk scan |
| `scaninfo.py` | What a scan's headers, names and folders say about it, to fill in its details |
| `drives.py` | Drives, their labels and formats on macOS, Windows and Linux |
| `jobs.py` | Queue, lanes, dependencies, subprocesses, progress, logs |
| `steps.py` | Each step's options, command line, progress markers and outputs; views; tools |
| `volumes.py` | Preview building, strided raw-scan reading, CMPR unwrap, crop suggestion |
| `results.py` | Cross-sample aggregation, HA calibration for the incisor profile, CSV exports |
| `trajectory.py` | The R report's cleanup of an incisor profile: quality filters, outliers against a GAM trend, gap-aware running median |
| `runners/` | Unattended wrappers: convert, segment, preview, copy |
| `web/` | The UI: `js/views/*` per page, `charts.js`, `compare.js` (grouping and Welch tests), `figure.js` (Prism-style figures in points, significance stars), `prism.js` (Prism tables: .pzfx, CSV, pasting), `files.js` (PNG DPI, TIFF, ZIP), `exporter.js` (the export dialogs), `chartstyle.js` (the Results page's colours and titles, and the colour picker), `trajectory.js` (the incisor curves' running median and coverage floor), `volume.js` (viewer engine), `render3d.js` (3-D ray marching, views that turn together), `pose.js` (each specimen's tissue centre, and fitting one to another), `scale.js` (one heatmap scale over several specimens), `histwin.js` (the histogram window), `colormaps.js` |

To add a step: describe it in `steps.py` (its `Param`s, a branch in `prepare`
that builds the command, a branch in `finish` that records what it wrote) and
add it to `PIPELINE_ORDER`. The UI builds its form and status from that
description.
