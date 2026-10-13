# Molar enamel quantification

`molar_enamel.py` measures how much of each molar is enamel: enamel volume as
a share of total molar volume, for the whole molar row and for M1, M2 and M3
separately, plus the enamel/dentin density contrast, the enamel's density in
mg HA/cm³, enamel thickness, and the patches of enamel thinner, or less
mineralized, than the enamel around them.

```bash
python molar_enamel.py -i ".../Segments/P52_Odam-Cre_WT_1" -o results --log   # one sample
python molar_enamel.py -i ".../Segments" -o results                           # every sample folder
python molar_enamel.py -i ".../Segments" -o results --threshold 7100          # the volume above one density
python molar_enamel.py -i ".../Segments" -o results --enamel-mask threshold    # the old mask: above the histogram's threshold
python molar_enamel.py -i ".../P22_WT_1_molar_volume.nrrd" -o results \
    --calibration ".../f0004028.isq"                                           # density in mg HA/cm³
python molar_enamel.py -i ".../Segments" -o results --save-points            # keep every junction point
python enamel_patches.py sweep results/*/*_junction_points.npz \
    --set around_um=40,60,90 --set thinner_by=0.2,0.25                        # the patches again, other settings
```

The input is the segmentation pipeline's molar volume
(`<sample>/volumes/<sample>_molar_volume.nrrd`, or `.tif`): scanner intensities
inside the molar mask, zero elsewhere. `-i` also accepts a sample folder or a
folder of sample folders. MicroCT Studio runs it as the **Molar enamel** step.

## Outputs

```
<output>/
├── <sample>_molar_enamel.json          summary, per-tooth table, contacts, threshold provenance, histogram
├── <sample>_molar_enamel_teeth.csv     one row per tooth
├── <sample>_molar_enamel_profile.csv   per-slice enamel share along the jaw
├── <sample>_enamel_mask.nrrd           enamel voxels (uint8), same frame as the input
├── <sample>_tooth_labels.nrrd          the teeth by number (1 = M1, 2 = M2, 3 = M3), same frame
├── <sample>_molar_enamel.png           histogram, profile, per-tooth bars
├── <sample>_thickness_map.html         the enamel coloured by its thickness, in 3-D: turn, move, zoom, read
├── <sample>_thickness_map.png          the same map from above and from one side
├── <sample>_junction_points.npz        with --save-points: every junction point's measurements
└── molar_enamel_summary.csv            batch mode: one row per sample
```

The `.html` map is self-contained (no network, no libraries): open it in any
current browser. The enamel is a solid layer coloured by its thickness, on the
grey dentin and roots. Drag to rotate, shift- or right-drag to move, scroll to
zoom, hover for the thickness under the pointer. Fading the enamel (its
opacity slider) shows the thickness of the enamel above each point on the
dentin's crown beneath. The colour bar's range can be changed, the panel gives
the thickness and the junction's area per tooth, and "Save image" keeps the
current view with its legend. `#view=top&enamel=40` at the end of its address
opens it on that view. Its **Colour** switch shows, besides the thickness,
how much thinner the enamel is than the enamel around it, or how much less
mineral its core holds (see *Patches* below): stepped orange bands, counted
patches at full colour and the rest of the judged enamel muted, what was not
judged grey. `#mode=thin` or `#mode=low_density` opens it that way.

**Measurement lines** draw the measurements themselves, to check what was
measured: each line runs from the point the thickness is reported at (its
thick end) to the point it reached, so its length is that point's thickness.
*From the junction* shows the reported measurement, from the enamel–dentin
junction to the outer surface; *from the outer surface* shows the one the
outer surface is painted by. The file holds a random sample of up to 20,000
lines per direction, spread evenly over the surface; the *Show* slider draws
a leading share of them. Lines are plain yellow, or coloured by their
thickness on the map's scale; a line that ends at a contact between teeth
rather than at the air is pink. Turning them on fades the enamel so they can
be seen inside it, and pointing at a line reads its length. A line that
crosses dentin, skips a groove, or ends on the wrong face is a pairing to
doubt. `#lines=junction` (or `outer`), `lines_share=50` and
`line_colour=value` open the map that way.

| Field | Meaning |
|---|---|
| `enamel.fraction` | Enamel voxels / molar voxels: the headline number |
| `total.volume_mm3`, `enamel.volume_mm3`, `dentin.volume_mm3` | Physical volumes, from the file's voxel size |
| `enamel.density` | The enamel's mean and lower/upper percentile (5th/95th by default), `raw` and in `mg_ha_cm3` |
| `calibration` | The HA calibration used: slope, intercept, divisor, and the file it came from (null without one) |
| `enamel_to_dentin_density` | Median enamel intensity / dentin histogram peak |
| `enamel_minus_dentin` | The same contrast as a difference, for scales that do not start at zero |
| `enamel.thickness_um` | Enamel thickness from the enamel-dentin junction: `median`, `mean` and `sd` (the sample standard deviation over the measured junction points), percentiles, share of junction points measured, `junction_area_mm2`, per tooth, the same measured from the outer surface, and the intensity levels the surfaces were placed at |
| `enamel.thickness_um.edge_blur_um` | The scan's blur at the enamel's outer face, as a Gaussian sigma in µm, read off the scan |
| `enamel.patches` | Patches of enamel thinner (`thin`) or less mineralized (`low_density`) than the enamel around them, for the row and per tooth: `share` (of the judged enamel-dentin junction under patches: the headline), `patches` and `patches_per_mm2`, mean `depth` (a share), `largest_mm2`, `shortfall` (the one-sided spread, no threshold), `share_judged`, `at_least` (the judged share past several thresholds), the largest patches with their centres in input voxels; the less dense ones also in mg HA/cm³ when calibrated, and the enamel's `core_density` |
| `teeth[]` | The same per tooth, named M1-M3, with `thickness_median_um`, `thickness_mean_um`, `thickness_sd_um`, `junction_area_mm2`, `thickness_mean_per_sqrt_junction` (the mean thickness over the square root of the junction's area, both in µm: a unitless proportion that stays the same for a tooth of the same shape at any size), `thickness_measured` and the patches (`thin_patch_share`, `low_density_patch_share`, ...), the crown's extent along the jaw (`crown_z_start`/`crown_z_end`) and the molar's landmark span (`landmark_z_start`/`landmark_z_end`, below) |
| `teeth_split` | How the teeth were parted: `dentin-N`, their dentin cores standing apart after N voxels of erosion |
| `contacts[]` | Each pair of neighbours (M1-M2, M2-M3): whether they touch, the area they share (`contact_area_mm2`), or the gap between them (`gap_um`) |
| `enamel_mask` | How the enamel was found: `local-halfway` (between its two halfway faces), with the levels its junction (`junction_level`) and outer face (`outer_level`) were placed at along the tooth, the dentin under the enamel (`dentin_under_enamel`), the enamel's local peak (`enamel_peak`), each as median and 5th/95th percentile, and how many voxels of blurred outer rim it holds; or `threshold`, with the value |
| `threshold` | The molar histogram's threshold, the way it was found (`log-valley`, `knee` where the log histogram shows no enamel peak, `otsu`, `fixed`), and its dentin and enamel peaks either side, where the local levels start |

## How it works

**The histogram's peaks.** The valley below the enamel peak of the molar's
*log* intensity histogram, using `_find_enamel_threshold` from
`segment_mandible/segmentation/molar_bone.py`, the same function the
segmentation uses to find the molars, splits it into its dentin and enamel
peaks. Enamel is a few percent of the molar, too small to show as a peak on a
linear histogram but a clear mode on a log one. Where the log histogram shows
none, the threshold falls back to the histogram's knee, and
`threshold.method` says `knee`. The top and bottom 0.01% of intensities are
dropped first (not clipped, which would make a spurious bright peak) so one
saturated streak cannot compress the histogram.

**Enamel mask.** The voxels between the enamel's two faces, its outer surface
and its junction with the dentin, each halfway between the materials on
either side (`enamel_surfaces.enamel_layer`). Halfway is read where each face
is, not once for the tooth, because on a blurred scan the enamel's brightness
is not one value:

- A thin layer never reaches the enamel's full brightness. On
  P82_Odam-Cre_WT_2 (SkyScan, 8 µm, edges blurred over σ 13-15 µm) the enamel
  peaks at 175 where it is 30-45 µm deep and at 220 where it is 90-120 µm.
- The scan dims along the jaw: under M3 its enamel peaks at 210 over dentin at
  134, under M1 at 181 over 120 (bone only 5 dimmer; beam hardening).

So every molar voxel follows the scan (smoothed by a voxel) uphill to the
enamel's local peak, within six voxels. The dentin under the enamel is read
four voxels past the junction, beside enamel beyond doubt, and pooled over 40
slices along the jaw. Beyond doubt means a peak halfway from the dentin read so
far to the enamel's brightness: the histogram's enamel peak, or the molar's
brightest 5% when that is brighter. A histogram with no enamel peak, not even
past its knee, puts its "peak" on the threshold itself. On P42_C67B6_WT_1 that
was 139, just above the crown's dentin (126); its dentin humps passed for
sure enamel, and at the far end of M3 the dentin read 70 and whole slices
passed for enamel. A voxel is on the outer face when background lies four
voxels downhill and the enamel's bright core within three voxels uphill, and on
the junction otherwise. It is enamel when its value reaches its face's
level: halfway between its local peak and the background on the outer face,
between that peak and the dentin under it at the junction. A peak standing
less than 15% of the dentin's own contrast (dentin minus background) above the
dentin is no enamel. On four scans that bar lay between the brightest 0.1% of
deep dentin and the dimmest 1% of enamel. It is the dentin's contrast, not the
enamel's, so a knockout's dimmer enamel is held to its wildtype's bar. The
outer face's blurred rim is a shell a voxel or two thick. The dentin beside the
enamel's edge, where the root's surface runs on or a worn cusp lays it bare,
passes the same per-voxel test, but it belongs to the thick dentin body, which
an opening keeps and the rim does not survive. Specks under 50 voxels go back
to the dentin (the segmentation's own seed-size rule).

The outer face is read on the unmasked scan: the molar volume is zero beyond
its mask, which costs the enamel its rim where the mask ends close to it
(2.7% of P82_Odam-Cre_WT_2's enamel). The scan is loaded for this even with
`--no-thickness`.

One threshold for the whole tooth fails on exactly these scans. On
P82_Odam-Cre_WT_2 the log histogram's enamel peak fell 0.02 short of the bar
(log prominence 0.28) and the knee put the threshold at 188, M1's median
local peak. The mask held 7.3% of the molar (M1 4.1%), and 45% of M1's crown
had no enamel under it. Its littermate WT_1 found its peak and held 16.8%.
Between the halfway faces they hold 18.5% and 19.4%, with thickness medians of
63.8 and 63.9 µm and junction areas of 5.00 and 5.10 mm². On a phantom molar row
blurred and dimmed as P82_Odam-Cre_WT_2 is, the halfway faces hold the enamel
that was built within 3%, Dice 0.96. One threshold kept 19% of it and none of
the dimmest tooth's. Thin layers read thick: a blurred layer peaks below
full brightness and its halfway points lie outside it, +11% for a cap of 5
voxels (40 µm at σ 14 µm), about 1% at 8 voxels.

On the twelve sound P22 TGM2 scans (Scanco, 6 µm), where the threshold had
found the enamel's peak, the enamel rose 2.6-4.0 points of the molar, mostly
the blurred outer rim the threshold left out. M3, still maturing at P22, rose
4.7-12 points as its dimmer enamel joined. Thickness medians fell 2-5 µm: the
junction now lies halfway to the dentin under the enamel (about 17,400) rather
than to the histogram's dentin peak (16,200), and covers more of the crown.
Every scan still splits into three teeth, M1 > M2 > M3, and crowns moved a
slice or two. On P22_TGM2_WT_M_4 the split no longer hands the outer rim of
M1's enamel beside the M1-M2 contact to M2, a sliver 16 slices long that the
old mask had left as dentin.

`--enamel-mask threshold` keeps the old mask, every molar voxel at or above
the histogram's threshold, and an absolute `--threshold` implies it. Where
the histogram's peaks stand too close to read faces from (a molar that is
mostly enamel, on a phantom), the threshold mask is used with a warning.

**Teeth.** Neighbouring molars touch enamel to enamel (M1 and M2 already do
by P22), and a contact is as thick and bright as the rest of the crown. No
threshold parts them, and neither does eroding the whole tooth: eight voxels
of erosion left M1 and M2 joined on all twelve P22 TGM2 scans, which were then
reported as two teeth, the fused pair as "M1" and M3 as "M2". Their dentin
never touches, so the teeth are split from it. The dentin is the molar without
its enamel and without anything lying between enamel layers: the seam of a
contact can read as dim as dentin and be several voxels across (on
P22_TGM2_KO_F_1, 5 voxels reading 10,700-20,000, with dentin at 16,000), and
taken for one tooth's dentin it floods the other's enamel from inside the
contact. That dentin is eroded a voxel at a time until three cores stand
apart: one or two voxels on the P22 scans. A core has to be thicker than a
blur band somewhere, so the blurred rim over the enamel is never taken for a
tooth.

Each core, grown back to its enamel-dentin junction, seeds a compact watershed
of the whole tooth on how much dimmer than enamel each voxel is. Through
enamel only the distance from each tooth's own junction counts, so two
touching teeth are cut at the seam of their contact where the scan shows one
(within a voxel of it), and otherwise midway between their two junctions --
off the true contact by half the difference where the two enamel layers
differ in thickness. Teeth are ordered along the jaw and named by size (mouse
M1 > M2 > M3), so the names do not depend on which way the scan's axis runs;
`<sample>_tooth_labels.nrrd` holds them by number. The split adds about 10 s
on a 6 µm scan.

**Crowns and landmark spans.** Each tooth's crown -- its enamel -- runs along
the jaw from its first to its last slice holding at least 0.5% of the enamel
in its fullest slice (`crown_z_start`/`crown_z_end`, in the input's z). The
landmark span is what the landmarks mark by eye as that molar's start and end:
the crown, less the slices where the crown in front of it (toward M1) still
shows, so neighbours meet without overlapping; start is the end toward M1.
The landmarks place M1-M3 from these: `molar_crowns` runs this split alone
(no measurements, a few seconds) for
`longitudinal_incisor_analysis/detect_landmarks.py`, which MicroCT Studio's
Landmarks step runs, so the landmarks do not wait for this step;
`map_landmarks.py --molars` takes them from this step's JSON instead. On P22_TGM2_KO_F_1 the spans lie within 1-4
slices of the hand-marked ends (M3's far end, 7).

**Contacts.** For each pair of neighbours, the area of the surface they share
where they touch (fused in the scan: touching, or closer than its blur
resolves), or the gap between them where they do not. The gap is measured
between the molar masks, which reach about a voxel into the blur beyond the
enamel, so it reads a voxel or two short of the space between the enamel
surfaces. On the thirteen P22 TGM2 scans, M1 and M2 share 0.09-0.18 mm² and
M3 stands 30-119 µm off M2; the cut between M1 and M2 lies 0.1-1.4 voxels
from the dimmest point of their contact's seam.

**Density.** The mean, lower and upper percentile of the enamel voxels'
intensities (`--percentiles`, default 5 95), converted to mg HA/cm³ with the
scanner's hydroxyapatite calibration:

    mg HA/cm³ = value / mu_scaling × slope + intercept

A Scanco scanner writes the calibration into every ISQ's extended header;
`tools/scanco_calibration.py` reads it, and `convert_ISQ_NRRD.py` now copies
it into `<stem>_meta.json`. `--calibration` takes either of those, or a JSON
with `slope` and `intercept` (and optionally `mu_scaling`); several paths are
tried in order. The converter's metadata also says whether the volume was
rescaled to 1/cm (then the divisor is 1) and whether a noise floor in shift
mode offset it (added back). `--ha-slope`/`--ha-intercept`/`--mu-scaling` give
a calibration by hand. Without one, the density is reported raw only and the
summary says why. For f0004028 (70 kVp, 0.5 mm Al) the calibration is slope
375.507, intercept −197.288, at mu_scaling 4096, which puts water at
−19 mg HA/cm³ and P22_WT_1's enamel at a mean of 1938 (5th–95th percentile
1662–2244).

Two things limit the number. The lower percentile leans on partial-volume
voxels at the enamel surface and the dentin junction, which read darker than
the enamel they belong to. And the converter's denoising and sharpening
filters move voxel values a little, so compare densities between scans
converted with one recipe. A converted mean outside 500–3500 mg HA/cm³ is
flagged as a likely units mistake.

**Thickness (`enamel_surfaces.py`).** The enamel's outer surface is placed
halfway between enamel and the background around the tooth, its junction with
the dentin halfway between enamel and dentin, each where a blurred edge truly
lies. (A single threshold cannot do both: the blurred outer rim reads like
dentin, so a mask cut at the enamel threshold stops a voxel short of the
surface, and the mask-based estimate this replaces read 5-13 µm low on
phantoms.) Each surface lies at its level where it runs, read as the enamel
mask reads it: the enamel's local peak, and the dentin under the enamel as the
mask found it. `levels.placed` gives the levels each surface was placed at.
Where no enamel stands (a local peak under the mask's bar) neither surface has
a level. Read off the dentin's own texture, a junction level there ran all
through the dentin near the enamel: 35% of P22_TGM2_KO_F_1's junction points,
which no line through enamel could leave from.
Surface points are told apart by what lies four voxels beyond them:
background (outer surface) or dentin (junction). Both need enamel two voxels
inside them, and the enamel mask says whether it is there. Asking whether the
scan there is brighter than the junction's level fails where the outer edge
is broad and the enamel stands little above its dentin. P42_C67B6_WT_1's outer
edge is blurred over σ 2.4 voxels (19 µm), and two voxels in, the scan had
risen only 80% of the way to the enamel. That refused the outer surface along
every crown wall: 37,000 points against 146,000 at the junction. The thickness
read 78 µm (middle half 48-105), with 16% of the junction unmeasured.
Answered by the mask, it found 115,000 points and read 69 µm, 92% measured. The thickness at each
junction point is the distance to the nearest outer-surface point that a
straight line through enamel reaches. A line runs through enamel while it stays
in the enamel mask away from its ends. Pairings that cut through dentin or across
a groove are refused and the next nearest tried. Testing brightness against the
junction's level instead refused 32% of P82_Odam-Cre_WT_2's pairings: every line
crosses the blurred outer rim, which reads below that level over 0.85 σ of blur. Where two teeth touch
there is no outer surface between them, so the cut between the teeth stands in
for each one's: the enamel under a contact is measured to the contact, not
along the fused enamel to wherever it next meets the air. It is reported per
tooth and for the row, measured from the junction, with the same measured
from the outer surface alongside (the two differ on a curved crown). On
blurred phantom shells of 40-80 µm it reads 1.4-2.5 µm high.

**Junction area.** The area of the enamel-dentin junction, per tooth and in
all, measured on the scan smoothed by one voxel: on the raw scan, noise
roughens the surface and inflates its area with the scan's noise. Smoothed, it
reads 0-6% low (the most on tight curves) and holds steady across a sevenfold
range of noise, so scans of differing quality compare. Like the thickness, it
covers the junction where enamel lies over dentin, stopping a couple of voxels
short of the enamel's edge near the roots. On WT-M-1 the enamel volume over
the junction area (56 µm) agrees with the median thickness (63 µm).

The background is read from the unmasked scan the molar volume was cut from,
since the molar volume is zero there: `--scan`, or the segmentation's
`<sample>_reoriented_volume.nrrd` found in the sample folder (the studio passes
it). Without it the background is taken as zero, with a warning. On WT-M-1 and
P22_TGM2_WT_M_4 (6 µm voxels) the thickness adds about 25 s and 2 GB of memory,
and the map is about 8 MB. `--no-thickness` skips thickness, area and map;
`--no-thickness-map` keeps the numbers but draws no map; `--thickness-max`
sets the top of the map's colour scale (default 120 µm; keep one value across a
cohort).

## Volume and density in a comparison

Between its halfway faces, the enamel is found by its shape: a layer brighter
than the dentin it covers. A knockout whose enamel is dimmer keeps its
volume, and the difference shows in its density, which is reported apart (and
in mg HA/cm³ with a calibration). Its enamel is held to the same bar as the
wildtype's: a local peak 15% of the dentin's contrast above the dentin.

A per-sample threshold instead mixes the two: it follows dim enamel down, or
misses it, and a dimmer enamel reads as less of it. To measure the volume
above one density on calibrated scans, give one absolute `--threshold` for the
whole cohort and report it. The studio exposes the choice as its **Enamel**
option (and **Fixed threshold**), and says on the Results page which rule
each sample was measured with.

The density ratio is meaningful only when intensity zero means air
(calibrated, or mu-scaled). With raw counts that carry an offset, compare
`enamel_minus_dentin` instead.

## Patches of thinner or less dense enamel

`enamel_patches.py`, run from the thickness. Thin enamel is not patchy enamel:
a tooth whose enamel is a tenth thinner everywhere has no patches, and one
whose enamel is full depth but for a few spots half as deep has. The median
barely moves for the second, and the spread of the thickness sees neither,
because on a molar crown the spread is anatomy. On every P22 TGM2 scan,
knockout and wildtype alike, the 5th percentile is the taper to the cervical
margin (13-15 µm) and the median 56-59 µm. So each point of the enamel-dentin
junction is held up against the enamel around it, on the same tooth.

**Two values per junction point.** The thickness measured from it, and the
density of the enamel's core above it: the scan along the line the thickness
was measured on, read only where it lies at least 2.5 blur sigmas from both
faces (checked against every face point, not only along the line). Blur pulls
a thin layer's brightness down, so without that clearance thin enamel reads as
less mineralized. The blur is read off the scan's own outer edge
(`edge_blur_um`). A column too thin to have such a core is not judged for
density. The density is taken above the background, so its deficit is the
share of the mineral that is missing, as the thickness's is the share of the
depth; not above the dentin, whose own mineral a genotype may change too.

**The enamel around a point** is a Gaussian-weighted, local-linear fit of the
value over the points within about 0.1 mm (sigma 60 µm, `--patch-around-um`),
read at the point. Local-linear, so the low side of a steady slope (a cusp's
flank, the taper toward the margin) is not taken for a patch; and after the
first fit each point counts by how little it falls short of the fit, so a
patch does not pull its own reference down. Thicker spots keep their say: the
weight is one-sided. On a sheet of points, disks 30-40% thinner and 40-90 µm
in radius are found at their full depth and 80-98% of their area; a plain
local average missed the 60-90 µm ones entirely.

**Judged points.** Not those measured to a contact between teeth, not those
within 100 µm of the enamel's edge (`--patch-margin-um`), and only where
"around" was read on enough enamel. The enamel's edge is where it stops on
bare dentin: the cervical edge, found from the bare root surface beside it,
and the rim of any bare area of the crown of at least 0.002 mm². At P22 most
cusp tips (and the middle of M3's crown) carry no enamel the mask counts, and
judged up to them, nearly every wildtype "thin patch" was the ring of enamel
tapering to one. `enamel.patches.margin` gives how many root pieces and bare
crown areas there were, and the bare crown's area (counted in surface voxels:
about its area).

**A patch** is judged points at least 45% thinner than around them and at
least 1.5 voxels thinner (`--thinner-by`), or missing at least 12% of the
mineral around them (`--less-dense-by`), joined within 1.5 voxels and covering
at least 0.002 mm² (`--patch-min-area-mm2`). Each kind shares the junction's
area among the points it can judge: where dim enamel brings the junction's
level near the noise, the raw junction doubles back into the enamel with
points that have no core to read, and sharing among all points halved a dim
phantom column's patch.

**The bars were set on the wildtypes, before any knockout was looked at.**
On the six usable P22 TGM2 wildtypes (never WT_M_5), 5.2% of the judged
junction fell past a 25% thickness bar (fissure floors are thinner than their
walls) and 37.8% past a 5% density bar (the enamel's own density varies about
5%, root-mean-square, around its surroundings; point noise is 1.2-1.4%). By a
rule fixed beforehand, each bar was raised until 1% of the pooled judged
wildtype junction fell past it: 43.8% thinner and 11.2% less mineral, each
rounded up to the map's bands, 45% and 12%. Leaving any one wildtype out moved
them to 42.8-44.5% and 11.1-11.2%, which round the same way. A patch is thus
enamel falling short of the enamel around it as only about 1% of P22 wildtype
enamel does. The wildtypes' own shares at those bars: 0.6-1.2% (thin) and
0.2-0.5% (less dense) of the judged junction, in patches between the cusps.
Two checks set beforehand did not settle anything: the wildtypes' ranking
changed with the scale (40, 60, 90 µm) -- they lie within a percent of one
another, too close to rank -- and the margin's band showed no elbow (wildtype
thin share 1.6, 0.8, 0.5% at 60, 100, 150 µm, against 78, 58, 35% of the
junction judged), so both stayed at their defaults, 60 µm and 100 µm. For other scanners, ages or strains, set
the bars again the same way on their own wildtypes. Enamel thinner than about two
voxels has no junction points, so a severe patch is a ring around a hole: the
hole it encloses counts with it (`hole_mm2`). A pit's patch reaches beyond its
rim: beside a pit, the nearest way out of the enamel is the pit's wall, so the
enamel there really is thinner by the measure the thickness uses.

**What to report.** Per animal, M1's `thin_patch_share` and
`low_density_patch_share`, knockout against wildtype; then M2, the count per
mm², the mean depth, and the threshold-free `shortfall` as a check that the
result does not hang on the threshold. Compare `share_judged`, `noise` and
`edge_blur_um` between the groups too: a group whose scans are noisier or
blurrier is judged on less, or more coarsely.

**What it cannot see.**
- Thinning wider than about 0.3 mm: "around" follows it, and the thickness
  median and percentiles carry it.
- Fissure floors really are thinner than their walls, so wildtype teeth have
  patches too. Read a knockout against its wildtype, never against zero.
- The density can be judged only where the enamel is deep enough to have a
  core: on the 6 µm P22 scans about half the junction, biased to thick enamel,
  and little on 8 µm scans blurred over 13-15 µm (P82, P42), where only the
  thickness's patches mean much.
- At P22 M3 is still maturing; at P42 and later worn cusps read as thin.
- Compare within one scanner and protocol: both kinds of patch depend on the
  blur and the noise.

The defaults were set on phantoms (`test_enamel_patches.py`, the patchy cap in
`test_molar_enamel.py`). `--save-points` keeps every junction point's
measurements, and `enamel_patches.py sweep` runs the patch stage again on them
under other settings in seconds, without measuring: the way to check a
cohort's result against the threshold and the scale.

## Tests

```bash
python -m pytest quantify_molar/test_molar_enamel.py
```

A three-tooth phantom with enamel caps of known thickness checks the threshold
position, the enamel fraction against construction, tooth naming in both scan
orientations, thickness, the fixed threshold, the enamel density and its
percentiles against the noise that was built in, the mg HA/cm³ conversion, the
batch CLI, and that the JSON parses under a strict parser (no NaN). A blurred
molar row fused enamel to enamel, with a dim bridge beneath each contact, must
come apart into the three teeth that were built; two teeth pressed together
check that the cut falls midway between the junctions without a seam and
within a voxel of one with it, that each tooth's enamel under the contact is
measured to the contact, and that a contact's area and a gap read as built.
A molar row blurred, quantized and dimmed toward M1 as P82_Odam-Cre_WT_2 is,
with a worn cusp, must keep each tooth's enamel whole between its halfway
faces (one threshold loses the dim tooth's), and a stretch of root dentin with
no enamel must keep its dentin. The blurred shell's outer rim must count as
enamel. Two shells dimmed apart along the jaw must each read their own dentin,
enamel and thickness. A shell blurred as P42_C67B6_WT_1's outer edge is (2.4
voxels), its enamel little above its dentin, must keep its outer surface and
read its thickness within a voxel (it reads about a voxel thick, as a layer
only 3.3 σ deep does). The histogram's threshold must say whether it came from
the valley or the knee. `tools/tests/test_scanco_calibration.py` pins the
calibration reader to the bytes of a real Scanco header.

`test_enamel_patches.py` paints thicknesses and densities on a tilted sheet of
junction points: uniform enamel and steady slopes have no patches, disks
thinner than the rest are found once at their size and depth, wide thinning
only in part, shallow, small or barely-thinner ones not at all, a taper to the
margin and a contact are not judged, a fragment alone is not judged, twice the
point density counts the same area, a ring around a hole counts the hole, and
the grid the fits run on matches the sums it stands for. A blurred, noisy tooth
at 6 µm with two pits, a column missing a tenth of its mineral, and decoys
below each bar must give exactly those patches, each kind only its own; the
pits' area and depth must match the nearest-point thickness of what was built;
the blur and the cervical edge must be read off the scan; and the saved points
must give the same patches again.
