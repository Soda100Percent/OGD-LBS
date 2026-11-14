// Shared Right-Side Multi-Card Render Inspector Dock with Bottom Unified Controls
(function() {
  const E = window.OGDChartEngine;

  const state = {
    mountId: 'render-dock',
    rendersByDataDir: {},
    summaryByDataDir: {},
    getScaleMode: () => 'scaled',
    currentDataDir: null,
    historyCards: [],
    nextSeq: 1,
    lockHover: false,
    // Global Unified Settings applied synchronously to Current Card & all History Cards
    settings: {
      baseMode: 'l2_heatmap', // 'l2_heatmap' | 'clay' | 'wireframe' | 'bone'
      camera: 'front',
      frameToken: 'f0000',
      overlayBone: true,
      overlayWire: false,
      overlayOpacity: 0.72,
      bgClass: 'bg-checker',
    },
    availableCameras: ['front'],
    availableFrames: ['f0000'],
  };

  function collectGlobalCameraAndFrameOptions() {
    const cams = new Set();
    const frames = new Set();
    Object.values(state.rendersByDataDir || {}).forEach(rec => {
      (rec.cameras || []).forEach(c => cams.add(c));
      (rec.frames || []).forEach(f => frames.add(f));
    });
    state.availableCameras = cams.size ? Array.from(cams).sort() : ['front'];
    state.availableFrames = frames.size ? Array.from(frames).sort() : ['f0000'];
    if (!state.availableCameras.includes(state.settings.camera)) {
      state.settings.camera = state.availableCameras[0];
    }
    if (!state.availableFrames.includes(state.settings.frameToken)) {
      state.settings.frameToken = state.availableFrames[0];
    }
  }

  function findInitialDataDir() {
    const keys = Object.keys(state.rendersByDataDir || {}).filter(
      dd => ((state.rendersByDataDir[dd] || {}).items || []).length > 0
    );
    return keys[0] || null;
  }

  function resolveLayersForDataDir(dataDir) {
    const rec = state.rendersByDataDir[dataDir];
    if (!rec || !rec.items || rec.items.length === 0) {
      return { available: false, baseItem: null, overlayItems: [], fallbackNotice: '' };
    }
    const wantCam = state.settings.camera;
    const wantFrame = state.settings.frameToken;

    let pool = rec.items.filter(it => it.camera_short === wantCam && it.frame_token === wantFrame);
    let fallbackNotice = '';
    if (pool.length === 0) {
      pool = rec.items.filter(it => it.camera_short === wantCam);
      if (pool.length > 0) fallbackNotice = `Fallback frame: ${pool[0].frame_token}`;
    }
    if (pool.length === 0) {
      pool = rec.items.filter(it => it.frame_token === wantFrame);
      if (pool.length > 0) fallbackNotice = `Fallback view: ${pool[0].camera_short}`;
    }
    if (pool.length === 0) {
      pool = rec.items;
      fallbackNotice = `Fallback: ${pool[0].camera_short} · ${pool[0].frame_token}`;
    }

    const byMode = {};
    pool.forEach(it => { byMode[it.render_mode] = it; });

    const baseItem = byMode[state.settings.baseMode] || byMode['l2_heatmap'] || byMode['clay'] || pool[0];
    const overlayItems = [];
    if (state.settings.overlayBone && byMode['bone'] && baseItem.render_mode !== 'bone') {
      overlayItems.push(byMode['bone']);
    }
    if (state.settings.overlayWire && byMode['wireframe'] && baseItem.render_mode !== 'wireframe') {
      overlayItems.push(byMode['wireframe']);
    }

    return {
      available: true,
      baseItem,
      overlayItems,
      fallbackNotice,
    };
  }

  function buildSingleCardHtml(cardObj) {
    const cid = cardObj.cardId;
    const dd = cardObj.dataDir;
    const summary = state.summaryByDataDir[dd] || {};
    const comp = summary.complexity || {};
    const mField = state.getScaleMode() === 'scaled' ? 'metrics_scaled' : 'metrics_raw';
    const mets = summary[mField] || {};
    const layers = resolveLayersForDataDir(dd);
    const entryShort = dd ? dd.split('/').pop() : '-';

    const badgeHtml = cardObj.isHistory
      ? `<span class="rc-badge rc-badge-history">📌 HISTORY #${cardObj.seq}</span>`
      : `<span class="rc-badge rc-badge-current">🎯 CURRENT (HOVER)</span>`;

    const headerBtnHtml = cardObj.isHistory
      ? `<button class="rc-icon-btn danger" onclick="OGDRenderDock.removeHistory('${cid}')" title="Delete this history card">✖ Delete</button>`
      : `<button class="rc-icon-btn" onclick="OGDRenderDock.pinCurrent()" title="Pin current card to History">📌 Pin</button>`;

    if (!layers.available) {
      return `
        <div class="render-card ${cardObj.isHistory ? 'history-card' : 'current-card'}" id="rc-${cid}">
          <div class="rc-header">
            <div>
              <div class="rc-title">${badgeHtml} ${E.esc(summary.model || '')} · ${E.esc(entryShort)}</div>
              <div class="rc-sub">${E.esc(summary.stage_label || '')} — ${E.esc(summary.method_label || 'No render available')}</div>
            </div>
            ${headerBtnHtml}
          </div>
          <div style="padding:20px 12px;text-align:center;color:#94a3b8;font-size:0.78rem;">
            No rendered image in pool for <code>${E.esc(dd || '-')}</code>.<br>
            Hover any table row or chart bar with renders to preview.
          </div>
        </div>
      `;
    }

    const bItem = layers.baseItem;
    const imgsHtml = [
      `<img class="layer-img base-layer" src="${E.esc(bItem.rel_png)}" alt="base">`,
      ...layers.overlayItems.map(ov =>
        `<img class="layer-img ov-layer" src="${E.esc(ov.rel_png)}" style="opacity:${state.settings.overlayOpacity}" alt="${E.esc(ov.render_mode)}">`
      ),
      bItem.rel_colorbar
        ? `<img class="colorbar-img" src="${E.esc(bItem.rel_colorbar)}" alt="colorbar">`
        : '',
    ].join('');

    const subNote = layers.fallbackNotice
      ? ` <span style="color:#fbbf24;">(${E.esc(layers.fallbackNotice)})</span>`
      : ` <span style="color:#38bdf8;">[${E.esc(bItem.render_mode)} · ${E.esc(bItem.camera_short)} · ${E.esc(bItem.frame_token)}]</span>`;

    return `
      <div class="render-card ${cardObj.isHistory ? 'history-card' : 'current-card'}" id="rc-${cid}">
        <div class="rc-header">
          <div>
            <div class="rc-title">${badgeHtml} ${E.esc(summary.model || '')} · <span style="color:#38bdf8">${E.esc(entryShort)}</span></div>
            <div class="rc-sub">${E.esc(summary.stage_label || '')} | <b>${E.esc(summary.method_label || '')}</b>${subNote}</div>
          </div>
          ${headerBtnHtml}
        </div>
        <div class="rc-viewport ${state.settings.bgClass}">
          ${imgsHtml}
        </div>
        <div class="rc-metrics-bar">
          <div>V/F: <b>${E.fmtInt(comp.num_vertices)}/${E.fmtInt(comp.num_triangles)}</b></div>
          <div>Bones: <b>${E.fmtInt(comp.num_bones_total)} (K=${E.fmtInt(comp.max_bones_per_vertex)})</b></div>
          <div>Loose: <b>${E.fmtInt(comp.num_loose_parts)}/${E.fmtInt(comp.num_loose_parts_welded)}</b></div>
          <div>VD (L2): <b>${E.fmt(mets.l2_err)}</b></div>
          <div>HD Mean: <b>${E.fmt(mets.hausdorff_mean)}</b></div>
          <div>ARAP: <b>${E.fmt(mets.arap_ratio)}</b></div>
        </div>
        <div class="rc-footer">
          <a href="${E.esc(bItem.rel_png)}" download="${E.esc(entryShort)}_${E.esc(summary.model || 'model')}_${E.esc(bItem.render_type)}_${E.esc(bItem.frame_token)}.png">
            📥 Raw PNG
          </a>
          <button onclick="OGDRenderDock.exportCompositedPng('${cid}')">
            🖼️ Export Composited PNG
          </button>
        </div>
      </div>
    `;
  }

  function renderCardsOnly() {
    const scrollEl = document.getElementById('dock-cards-scroll');
    if (!scrollEl) return;
    if (!state.currentDataDir) {
      state.currentDataDir = findInitialDataDir();
    }
    const curCardHtml = buildSingleCardHtml({
      cardId: 'current',
      isHistory: false,
      dataDir: state.currentDataDir,
    });
    const histHtml = state.historyCards.map(hc => buildSingleCardHtml(hc)).join('');
    scrollEl.innerHTML = curCardHtml + histHtml;

    const cntEl = document.getElementById('dock-hist-count');
    if (cntEl) cntEl.textContent = String(state.historyCards.length);
  }

  function renderFullDock() {
    const mount = document.getElementById(state.mountId);
    if (!mount) return;
    collectGlobalCameraAndFrameOptions();

    const baseOpts = [
      ['l2_heatmap', '🔥 L2 Heatmap'],
      ['clay', '🎨 Clay Shaded'],
      ['wireframe', '🕸️ Wireframe'],
      ['bone', '🦴 Skeleton Only'],
    ].map(([v, lbl]) => `<option value="${v}" ${state.settings.baseMode === v ? 'selected' : ''}>${lbl}</option>`).join('');

    const camOpts = state.availableCameras.map(c =>
      `<option value="${E.esc(c)}" ${state.settings.camera === c ? 'selected' : ''}>${E.esc(c)}</option>`
    ).join('');

    const frameOpts = state.availableFrames.map(f =>
      `<option value="${E.esc(f)}" ${state.settings.frameToken === f ? 'selected' : ''}>${E.esc(f)}</option>`
    ).join('');

    const bgOpts = [
      ['bg-checker', 'Checkerboard'],
      ['bg-dark', 'Dark'],
      ['bg-white', 'White (Paper)'],
    ].map(([v, lbl]) => `<option value="${v}" ${state.settings.bgClass === v ? 'selected' : ''}>${lbl}</option>`).join('');

    mount.innerHTML = `
      <div class="dock-top-header">
        <h3>🖼️ Render Inspector Dock</h3>
        <div style="display:flex;gap:5px;align-items:center;">
          <label style="font-size:0.73rem;display:flex;align-items:center;gap:3px;cursor:pointer;" title="Pause hover switching">
            <input type="checkbox" id="dock-lock-hover" ${state.lockHover ? 'checked' : ''}
              onchange="OGDRenderDock.setLockHover(this.checked)"> 🔒 Lock
          </label>
          <button class="rc-icon-btn" onclick="OGDRenderDock.pinCurrent()" title="Pin current card to History">📌 Pin</button>
          <button class="rc-icon-btn danger" onclick="OGDRenderDock.clearHistory()" title="Clear all pinned history cards">
            🗑️ Clear (<span id="dock-hist-count">${state.historyCards.length}</span>)
          </button>
        </div>
      </div>

      <div class="dock-cards-scroll" id="dock-cards-scroll"></div>

      <div class="dock-bottom-controls">
        <div class="dock-ctrl-title">
          <span>⚙️ UNIFIED RENDER SETTINGS (Syncs All Cards)</span>
        </div>
        <div class="dock-ctrl-grid">
          <div class="dock-ctrl-item">
            <span>Type</span>
            <select onchange="OGDRenderDock.setGlobalSetting('baseMode', this.value)">${baseOpts}</select>
          </div>
          <div class="dock-ctrl-item">
            <span>Angle</span>
            <select onchange="OGDRenderDock.setGlobalSetting('camera', this.value)">${camOpts}</select>
          </div>
          <div class="dock-ctrl-item">
            <span>Frame</span>
            <select onchange="OGDRenderDock.setGlobalSetting('frameToken', this.value)">${frameOpts}</select>
          </div>
          <div class="dock-ctrl-item">
            <span>BG</span>
            <select onchange="OGDRenderDock.setGlobalSetting('bgClass', this.value)">${bgOpts}</select>
          </div>
        </div>
        <div class="dock-overlay-row">
          <div style="display:flex;gap:8px;align-items:center;">
            <span style="color:#94a3b8;font-weight:600;">Overlay:</span>
            <label>
              <input type="checkbox" ${state.settings.overlayBone ? 'checked' : ''}
                onchange="OGDRenderDock.setGlobalSetting('overlayBone', this.checked)"> +🦴 Bone
            </label>
            <label>
              <input type="checkbox" ${state.settings.overlayWire ? 'checked' : ''}
                onchange="OGDRenderDock.setGlobalSetting('overlayWire', this.checked)"> +🕸️ Wire
            </label>
          </div>
          <label title="Overlay opacity">
            <span style="color:#94a3b8;">Alpha:</span>
            <input type="range" min="15" max="100" value="${Math.round(state.settings.overlayOpacity * 100)}"
              style="width:65px;" oninput="OGDRenderDock.setGlobalSetting('overlayOpacity', Number(this.value)/100)">
          </label>
        </div>
      </div>
    `;

    renderCardsOnly();
  }

  function init(config) {
    state.mountId = config.mountId || 'render-dock';
    state.rendersByDataDir = config.rendersByDataDir || {};
    state.summaryByDataDir = config.summaryByDataDir || {};
    if (config.getScaleMode) state.getScaleMode = config.getScaleMode;
    state.currentDataDir = config.initialDataDir || findInitialDataDir();
    renderFullDock();
  }

  function setGlobalSetting(key, value) {
    state.settings[key] = value;
    renderCardsOnly();
  }

  function setLockHover(locked) {
    state.lockHover = Boolean(locked);
  }

  function hover(dataDir) {
    if (!dataDir || state.lockHover) return;
    if (state.currentDataDir === dataDir) return;
    state.currentDataDir = dataDir;
    renderCardsOnly();
    document.querySelectorAll('tr.hoverable-row').forEach(tr => {
      tr.classList.toggle('row-active-hover', tr.getAttribute('data-dir') === dataDir);
    });
  }

  function pin(dataDir) {
    const target = dataDir || state.currentDataDir;
    if (!target) return;
    const seq = state.nextSeq++;
    state.historyCards.unshift({
      cardId: `hist_${seq}`,
      seq: seq,
      isHistory: true,
      dataDir: target,
    });
    renderCardsOnly();
  }

  function pinCurrent() {
    pin(state.currentDataDir);
  }

  function removeHistory(cardId) {
    state.historyCards = state.historyCards.filter(c => c.cardId !== cardId);
    renderCardsOnly();
  }

  function clearHistory() {
    state.historyCards = [];
    renderCardsOnly();
  }

  function toggleDock() {
    const wl = document.getElementById('workspace-layout');
    if (wl) wl.classList.toggle('dock-collapsed');
    setTimeout(() => {
      if (window.OGDChartEngine) window.OGDChartEngine.resizeAll();
    }, 120);
  }

  function loadImg(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  async function exportCompositedPng(cardId) {
    const cardObj = cardId === 'current'
      ? { dataDir: state.currentDataDir }
      : state.historyCards.find(c => c.cardId === cardId);
    if (!cardObj || !cardObj.dataDir) return;
    const layers = resolveLayersForDataDir(cardObj.dataDir);
    if (!layers.available || !layers.baseItem) return;

    const w = (layers.baseItem.resolution && layers.baseItem.resolution[0]) || 1024;
    const h = (layers.baseItem.resolution && layers.baseItem.resolution[1]) || 1024;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    if (state.settings.bgClass === 'bg-white') {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, w, h);
    } else if (state.settings.bgClass === 'bg-dark') {
      ctx.fillStyle = '#090d16';
      ctx.fillRect(0, 0, w, h);
    }

    const baseImg = await loadImg(layers.baseItem.rel_png);
    ctx.globalAlpha = 1.0;
    ctx.drawImage(baseImg, 0, 0, w, h);

    for (const ov of layers.overlayItems) {
      const ovImg = await loadImg(ov.rel_png);
      ctx.globalAlpha = state.settings.overlayOpacity;
      ctx.drawImage(ovImg, 0, 0, w, h);
    }
    ctx.globalAlpha = 1.0;

    if (layers.baseItem.rel_colorbar) {
      try {
        const cbImg = await loadImg(layers.baseItem.rel_colorbar);
        const cbH = Math.round(h * 0.52);
        const cbW = Math.round((cbImg.width / Math.max(1, cbImg.height)) * cbH);
        const x0 = w - cbW - 18;
        const y0 = h - cbH - 18;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.78)';
        ctx.fillRect(x0 - 6, y0 - 6, cbW + 12, cbH + 12);
        ctx.drawImage(cbImg, x0, y0, cbW, cbH);
      } catch (_) {}
    }

    const summary = state.summaryByDataDir[cardObj.dataDir] || {};
    const entryShort = cardObj.dataDir.split('/').pop();
    canvas.toBlob(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${entryShort}_${summary.model || 'model'}_${layers.baseItem.render_mode}_${layers.baseItem.camera_short}_${layers.baseItem.frame_token}_composited.png`;
      a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  }

  window.OGDRenderDock = {
    init,
    refresh: renderCardsOnly,
    setGlobalSetting,
    setLockHover,
    hover,
    pin,
    pinCurrent,
    removeHistory,
    clearHistory,
    toggleDock,
    exportCompositedPng,
  };
})();
