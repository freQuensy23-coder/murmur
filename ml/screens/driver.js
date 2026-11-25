// Headless renderer: one Chrome, one page per device size, JSON lines in / JSON lines out.
//   node driver.js            (reads {"id","html","w","h","css_w","css_h","out"} per line on stdin)
// Uses the system Chrome through playwright-core (never downloads browsers). Deterministic: fixed
// viewport and device scale, animations off, fonts awaited, no network.
'use strict';
const path = require('path');
const fs = require('fs');
const readline = require('readline');
const NODE_MODULES = process.env.MURMUR_SCREENS_NODE || path.join(__dirname, '..', '..', '.cache', 'ml', 'screens', 'node_modules');
const { chromium } = require(path.join(NODE_MODULES, 'playwright-core'));
const CHROME = process.env.MURMUR_CHROME || '/usr/local/bin/google-chrome';

// Collect the text that is really visible: per character, inside every clipping ancestor and the
// viewport, and not covered by another element (elementFromPoint at the glyph centre).
function collectVisible() {
  const vw = window.innerWidth, vh = window.innerHeight;
  const out = [];
  const clipCache = new Map();
  function clipOf(el) {
    if (!el || el === document.documentElement) return [0, 0, vw, vh];
    if (clipCache.has(el)) return clipCache.get(el);
    let c = clipOf(el.parentElement);
    const cs = getComputedStyle(el);
    if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible' || cs.clipPath !== 'none') {
      const r = el.getBoundingClientRect();
      c = [Math.max(c[0], r.left), Math.max(c[1], r.top), Math.min(c[2], r.right), Math.min(c[3], r.bottom)];
    }
    if (cs.visibility === 'hidden' || cs.opacity === '0' || cs.display === 'none') c = [0, 0, 0, 0];
    clipCache.set(el, c);
    return c;
  }
  function roleOf(el) {
    const r = el.closest('[data-role]');
    return r ? r.getAttribute('data-role') : 'chrome';
  }
  const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  let node;
  while ((node = tw.nextNode())) {
    const s = node.nodeValue;
    if (!s || !s.trim()) continue;
    const el = node.parentElement;
    if (!el || el.closest('[data-ocr-ignore]')) continue;
    const clip = clipOf(el);
    if (clip[2] <= clip[0] || clip[3] <= clip[1]) continue;
    const role = roleOf(el);
    const ell = getComputedStyle(el).textOverflow === 'ellipsis';
    let cur = null, lead = false;
    const flush = () => { if (cur && cur.text.trim()) { cur.trail = !!cur.pendingSpace; out.push(cur); } cur = null; };
    // iterate by code point (keeps surrogate pairs / emoji together)
    let i = 0;
    while (i < s.length) {
      const cp = s.codePointAt(i);
      const n = cp > 0xffff ? 2 : 1;
      range.setStart(node, i); range.setEnd(node, i + n);
      const rects = range.getClientRects();
      const r = rects.length ? rects[0] : null;
      const ch = s.substr(i, n);
      i += n;
      if (!r || r.width === 0 || r.height === 0) {           // collapsed whitespace, line-break space
        if (cur && /\s/.test(ch)) cur.pendingSpace = true;
        continue;
      }
      const cx = (r.left + r.right) / 2, cy = (r.top + r.bottom) / 2;
      let vis = cx >= clip[0] && cx < clip[2] && cy >= clip[1] && cy < clip[3];
      if (vis) {
        const hit = document.elementFromPoint(cx, cy);
        vis = !!hit && (hit === el || el.contains(hit) || hit.contains(el));
      }
      if (!vis) { if (cur) cur.cut = true; continue; }
      if (cur && (Math.abs(r.top - cur.top) > r.height * 0.5 || cur.cut)) flush();
      if (/\s/.test(ch)) { if (cur) cur.pendingSpace = true; else lead = true; continue; }
      if (!cur) { cur = { role, lead, text: '', x0: r.left, y0: r.top, x1: r.right, y1: r.bottom, top: r.top, ell }; lead = false; }
      if (cur.pendingSpace && cur.text) cur.text += ' ';
      cur.pendingSpace = false;
      cur.text += ch;
      cur.x0 = Math.min(cur.x0, r.left); cur.y0 = Math.min(cur.y0, r.top);
      cur.x1 = Math.max(cur.x1, r.right); cur.y1 = Math.max(cur.y1, r.bottom);
    }
    if (cur && cur.ell && cur.cut) cur.text += '…';
    flush();
  }
  // exclusion rectangles declared by the template
  const ex = [];
  document.querySelectorAll('[data-exclude]').forEach(e => {
    const r = e.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) ex.push({ kind: e.getAttribute('data-exclude'), box: [r.left, r.top, r.right, r.bottom] });
  });
  // the focused editable node: its bounds clipped to its ancestors' clip and the viewport (what
  // AccessibilityNodeInfo.getBoundsInScreen reports for a partly scrolled-off view)
  const focus = [];
  document.querySelectorAll('[data-focus]').forEach(e => {
    const r = e.getBoundingClientRect(), c = clipOf(e.parentElement);
    const y = e.getAttribute('data-focus') === 'y-parent' ? e.parentElement.getBoundingClientRect() : r;
    const b = [Math.max(r.left, c[0]), Math.max(y.top, c[1]), Math.min(r.right, c[2]), Math.min(y.bottom, c[3])];
    focus.push(b[2] > b[0] && b[3] > b[1] ? b : null);
  });
  return { items: out.map(o => ({ role: o.role, text: o.text, lead: o.lead, trail: o.trail, box: [o.x0, o.y0, o.x1, o.y1] })),
           exclude: ex, focus, vw, vh };
}

async function main() {
  const browser = await chromium.launch({
    executablePath: CHROME, headless: true,
    args: ['--font-render-hinting=none', '--disable-gpu', '--disable-lcd-text', '--force-color-profile=srgb',
           '--disable-font-subpixel-positioning', '--hide-scrollbars', '--no-sandbox'],
  });
  const ctxs = new Map();
  async function pageFor(cw, ch, dsf, loc) {
    const bcp = (loc === 'he' || loc === 'he-IL') ? 'he-IL' : 'ru-RU';
    const k = `${cw}x${ch}@${dsf}@${bcp}`;
    if (!ctxs.has(k)) {
      const c = await browser.newContext({ viewport: { width: cw, height: ch }, deviceScaleFactor: dsf,
        reducedMotion: 'reduce', javaScriptEnabled: true, offline: true, locale: bcp, timezoneId: 'UTC' });
      const p = await c.newPage();
      ctxs.set(k, p);
    }
    return ctxs.get(k);
  }
  const rl = readline.createInterface({ input: process.stdin });
  for await (const line of rl) {
    if (!line.trim()) continue;
    let req;
    try {
      req = JSON.parse(line);
      const dsf = req.w / req.css_w;
      const page = await pageFor(req.css_w, req.css_h, dsf, req.locale);
      await page.goto('file://' + req.html, { waitUntil: 'load' });
      await page.evaluate(() => document.fonts.ready);
      const vis = await page.evaluate(collectVisible);
      const buf = await page.screenshot({ type: 'png', animations: 'disabled', caret: 'hide', scale: 'device' });
      fs.writeFileSync(req.out, buf);
      process.stdout.write(JSON.stringify({ id: req.id, ok: true, dsf, visible: vis }) + '\n');
    } catch (e) {
      process.stdout.write(JSON.stringify({ id: req && req.id, ok: false, error: String(e && e.stack || e) }) + '\n');
    }
  }
  await browser.close();
}
main().catch(e => { console.error(e); process.exit(1); });
