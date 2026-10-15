# Using the studio from an iPad (Tailscale)

The studio can run on your Mac and be used from an iPad (or phone, or another
laptop) anywhere, over [Tailscale](https://tailscale.com). The Mac does all the
work and holds all the data; the iPad is just the screen. This is the way to
demo it when the Mac has to stay behind.

```bash
conda activate microct-analysis
python -m studio --last --tailscale
```

It prints the address to open on the iPad, for example:

```
MicroCT Studio over Tailscale. On the iPad (Tailscale on), open:
               http://100.101.102.103:8765/
               http://ethans-macbook-pro:8765/
               http://ethans-macbook-pro.tail1234.ts.net:8765/
```

The first address (the numbers) always works. The names work when MagicDNS is
on in your tailnet, which it is by default.

## What `--tailscale` does

- **Listens on the Mac's Tailscale address only** (the `100.x.y.z` one). Your
  own devices signed in to your tailnet can reach it; nobody else on the
  conference or café Wi-Fi can, because the studio isn't listening on that
  network at all. Tailscale encrypts the connection end to end.
- **Answers to the Mac's tailnet names.** The server refuses requests that
  don't name it (a guard against malicious web pages), so it's told its
  MagicDNS names as well as its address.
- **Keeps the Mac from idle-sleeping** while the studio runs (macOS
  `caffeinate`). A sleeping Mac drops off the tailnet. `--allow-sleep` turns
  this off.
- **Hides "Show in Finder" and "napari"** in the interface, since they would
  open windows on the Mac's screen, where nobody is looking.

In this mode `http://localhost:8765` on the Mac itself does **not** work; use
the printed address there too (the Mac can reach its own tailnet address).

## The night before: checklist

Do all of this at home, then test from the iPad **on cellular or a phone
hotspot, not your home Wi-Fi**, so you're testing what will happen at the
venue.

**On the Mac**

1. **Tailscale is on and signed in.** Menu-bar icon → *Connected*. Check
   with `tailscale status`; it should list both the Mac and the iPad.
   - If `tailscale` is "command not found": Tailscale menu-bar app →
     *Settings* → *Install CLI*. (The studio also looks inside
     `/Applications/Tailscale.app` on its own, so this is only needed for
     checking by hand.)
2. **Plugged into power.**
3. **Lid stays open.** `caffeinate` stops idle sleep but **not** lid-close
   sleep. Leave the lid open with the screen dimmed. A locked screen is fine:
   lock with ⌃⌘Q and the studio keeps running.
   - Optional, belt and braces: *System Settings → Battery → Options →
     "Prevent automatic sleeping on power adapter when the display is off"*.
   - If the lid *must* close: `sudo pmset -a disablesleep 1` before you leave,
     and **`sudo pmset -a disablesleep 0` when you're back** (it stays in force
     until you undo it, even across restarts, and a closed, unventilated
     laptop running heavy jobs gets hot).
4. **The data drive is connected and mounted** (e.g. the T7) and the
   workspace opens on the Mac. `--last` reopens the last workspace only if
   its drive is connected; otherwise you'll be asked to pick one on the iPad.
5. **The firewall prompt is answered.** The first time you run with
   `--tailscale`, macOS may ask *"Allow python to accept incoming network
   connections?"*. Click **Allow**. If nobody answers it, the iPad can't
   connect. If you missed it: *System Settings → Network → Firewall →
   Options* and set python to *Allow incoming connections*.
6. **The repository is available offline.** The project lives in Box. Make
   sure the folder is downloaded (right-click → *Make Available Offline*) so
   Box can't pull files out from under the running server.
7. **No other studio is running** on the same workspace. Quit any other
   `python -m studio` first: a second server on the same workspace marks the
   first one's running jobs as failed.
8. Start it in a Terminal window you leave open:
   `python -m studio --last --tailscale`.
9. Make sure macOS won't restart itself for an update overnight
   (*System Settings → General → Software Update → Automatic Updates*: off
   for tonight).

**On the iPad**

1. Tailscale app installed, signed in **with the same account** as the Mac,
   switch on (the VPN icon appears in the status bar).
2. Safari → the address the Mac printed. The workspace should appear.
3. Optional, for a cleaner demo: Share → **Add to Home Screen**. Opened from
   the Home Screen icon, the studio fills the screen with no Safari address
   bar.
4. Walk through the exact demo: open the specimens, the viewer, the 3-D view,
   Results and Compare. Each 3-D view streams its volume from the Mac the
   first time it opens, so this also shows how fast it'll be.
5. Set the iPad's Auto-Lock to *Never* (Settings → Display & Brightness)
   for the talk.

## At the venue

- Connect the iPad to the venue Wi-Fi (or your phone's hotspot), then switch
  Tailscale on. Open the studio from the bookmark or Home Screen icon.
- If the venue Wi-Fi is locked down, Tailscale falls back to its relay
  servers: it still works, just slower for opening 3-D volumes. A phone
  hotspot is usually faster than venue Wi-Fi.
- Open the heavy views (3-D, Compare) once before you go on stage so they're
  already loaded.
- Have a fallback: a short screen recording of the demo on the iPad, in case
  the network fails entirely.

## What works from the iPad

Everything in the browser works: the library, the slice viewer, 3-D views
(drag to turn, pinch to zoom, two fingers to move), landmarks, Results charts,
Compare, figure export (saved to the iPad's Files) and queueing jobs. Jobs run
on the Mac, not the iPad.

Things to avoid for the demo:

- **Importing new scans from the iPad.** Uploading multi-gigabyte scans over
  the internet is slow, and iPad Safari can't pick a whole folder. Import at
  the Mac beforehand.
- **Chart tooltips** follow a mouse pointer; on the iPad, tap a point or
  drag a finger along a chart to read values. Rehearse it so it doesn't
  surprise you.
- **Heavy jobs during the talk.** Segmentation and conversion run on the Mac
  and can take minutes; queue one only if you want to show the queue.

## Troubleshooting

| What you see on the iPad | Cause and fix |
|---|---|
| Page never loads / "Safari can't connect" | Tailscale off on one side, the Mac asleep, the studio stopped, or the firewall prompt unanswered. On the Mac: `tailscale status` and check the Terminal window is still running. |
| `Unrecognized Host header` | You used a name the studio wasn't told about (e.g. a custom alias). Use the numeric `100.x.y.z` address. |
| `Cannot serve over Tailscale: ... Stopped` (in the Mac Terminal) | Tailscale is off on the Mac. Switch it on in the menu bar and run the command again. |
| Workspace shows "Not connected" | The data drive isn't mounted on the Mac. Plug it in, then open the workspace again. |
| 3-D views very slow to open | Tailscale is relaying. `tailscale ping <ipad-name>` on the Mac says `via DERP` when relayed and `via <ip>` when direct. Try a phone hotspot. |
| "Copy for Prism" won't copy | Use *Save CSV* instead. (A fallback copy is built in, but some iPadOS versions refuse it.) |

## Security

Anyone who can open the page can run jobs and browse files on the Mac. With
`--tailscale` that means **every device in your tailnet**. That's fine for a
personal tailnet (just your own devices). If you've shared the Mac with other
people in Tailscale, or your tailnet belongs to an organization, restrict who
can reach the Mac with
[Tailscale access controls](https://tailscale.com/kb/1018/acls), or don't use
this mode. The studio has no password of its own.

Stop it with Ctrl-C in the Terminal when the demo is over. That also lets the
Mac sleep normally again.

Don't use `--host 0.0.0.0` instead: that listens on *every* network the Mac
is on, including the venue Wi-Fi.
