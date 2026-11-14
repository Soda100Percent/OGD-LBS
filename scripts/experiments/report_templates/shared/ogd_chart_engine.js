// Shared OGD-LBS ECharts Engine with ResizeObserver, Pseudo-Log Axis, Whiskers & Export
(function() {
  const instancesById = {};

  const resizeObserver = typeof ResizeObserver !== 'undefined'
    ? new ResizeObserver((entries) => {
        for (const entry of entries) {
          const dom = entry.target;
          if (!dom || !dom.id) continue;
          const inst = instancesById[dom.id] || (typeof echarts !== 'undefined' && echarts.getInstanceByDom(dom));
          if (inst) {
            try { inst.resize(); } catch (_) {}
          }
        }
      })
    : null;

  function esc(v) {
    return String(v ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function num(v) {
    if (v === null || v === undefined || v === '' || typeof v === 'boolean') return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function fmt(v, digits = 4) {
    const n = num(v);
    if (n === null) return '-';
    if (n === 0) return '0';
    if (Math.abs(n) >= 1e5 || (Math.abs(n) > 0 && Math.abs(n) < 1e-3)) {
      return n.toExponential(2).replace('e+', 'e');
    }
    return Number(n.toPrecision(digits)).toString();
  }

  function fmtInt(v) {
    const n = num(v);
    if (n === null) return '-';
    return Math.round(n).toLocaleString();
  }

  function fmtP(v) {
    const n = num(v);
    if (n === null) return '-';
    if (n < 1e-4) return n.toExponential(2);
    return n.toFixed(4);
  }

  function choosePseudoLogC(vals, axisMode = 'auto', ratioThreshold = 40.0) {
    if (axisMode === 'linear') return null;
    const pos = (vals || []).map(num).filter(v => v !== null && v > 0).sort((a, b) => a - b);
    if (pos.length < 2) return null;
    const ratio = pos[pos.length - 1] / pos[0];
    if (axisMode === 'auto' && ratio < ratioThreshold) return null;
    const idx = Math.floor((pos.length - 1) * 0.15);
    return Math.max(pos[idx], 1e-12);
  }

  function txY(v, c) {
    const n = num(v);
    if (n === null) return null;
    if (c === null || c === undefined) return n;
    return Math.log10(1 + Math.max(0, n) / c);
  }

  function invTxY(v, c) {
    const n = num(v);
    if (n === null) return null;
    if (c === null || c === undefined) return n;
    return c * (Math.pow(10, n) - 1);
  }

  function makeWhiskerSeries(whiskerData, halfWidth = 4.5, stroke = '#0f172a') {
    return {
      name: '[Min, Max] Range',
      type: 'custom',
      z: 15,
      tooltip: { show: false },
      data: whiskerData,
      renderItem: (params, api) => {
        const xIdx = api.value(0);
        const lowPoint = api.coord([xIdx, api.value(1)]);
        const highPoint = api.coord([xIdx, api.value(2)]);
        return {
          type: 'group',
          children: [
            { type: 'line', shape: { x1: lowPoint[0], y1: lowPoint[1], x2: highPoint[0], y2: highPoint[1] }, style: { stroke: stroke, lineWidth: 1.3 } },
            { type: 'line', shape: { x1: highPoint[0] - halfWidth, y1: highPoint[1], x2: highPoint[0] + halfWidth, y2: highPoint[1] }, style: { stroke: stroke, lineWidth: 1.3 } },
            { type: 'line', shape: { x1: lowPoint[0] - halfWidth, y1: lowPoint[1], x2: lowPoint[0] + halfWidth, y2: lowPoint[1] }, style: { stroke: stroke, lineWidth: 1.3 } }
          ]
        };
      }
    };
  }

  function renderLegendBanner(containerOrId, items) {
    const el = typeof containerOrId === 'string' ? document.getElementById(containerOrId) : containerOrId;
    if (!el) return;
    el.innerHTML = (items || []).map(it => {
      const style = it.dashed
        ? 'background:' + it.color + ';border:1.5px dashed ' + (it.borderColor || '#0f172a') + ';'
        : 'background:' + it.color + ';';
      const sym = it.symbolText ? ('<b style="color:' + it.color + ';font-size:1.05rem;">' + esc(it.symbolText) + '</b> ') : '';
      return '<span class="legend-item"><span class="legend-swatch" style="' + style + '"></span>' + sym + esc(it.label) + '</span>';
    }).join('');
  }

  function init(domOrId) {
    const dom = typeof domOrId === 'string' ? document.getElementById(domOrId) : domOrId;
    if (!dom || typeof echarts === 'undefined') return null;
    let chart = echarts.getInstanceByDom(dom);
    if (!chart) {
      chart = echarts.init(dom, null, { renderer: 'svg' });
    }
    if (dom.id) {
      instancesById[dom.id] = chart;
    }
    if (resizeObserver) {
      try { resizeObserver.observe(dom); } catch (_) {}
    }
    return chart;
  }

  function bindRenderDock(chart, resolveDataDirFn) {
    if (!chart) return;
    chart.off('mouseover');
    chart.off('click');
    chart.on('mouseover', (params) => {
      const dd = resolveDataDirFn ? resolveDataDirFn(params) : (params.data && params.data.dataDir);
      if (dd && window.OGDRenderDock) {
        window.OGDRenderDock.hover(dd);
      }
    });
    chart.on('click', (params) => {
      const dd = resolveDataDirFn ? resolveDataDirFn(params) : (params.data && params.data.dataDir);
      if (dd && window.OGDRenderDock) {
        window.OGDRenderDock.pin(dd);
      }
    });
  }

  function resizeAll() {
    Object.values(instancesById).forEach(c => {
      try { c.resize(); } catch (_) {}
    });
  }

  function exportChart(domId, format = 'svg', filename = null) {
    const dom = document.getElementById(domId);
    const svg = dom ? dom.querySelector('svg') : null;
    if (!svg) return;
    const outName = filename || domId;
    let content = svg.outerHTML;
    if (!content.includes('xmlns="http://www.w3.org/2000/svg"')) {
      content = content.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ');
    }
    if (format === 'svg') {
      const blob = new Blob([content], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${outName}.svg`; a.click();
      URL.revokeObjectURL(url);
      return;
    }
    const bbox = svg.getBoundingClientRect();
    const w = parseFloat(svg.getAttribute('width')) || bbox.width || 900;
    const h = parseFloat(svg.getAttribute('height')) || bbox.height || 500;
    const canvas = document.createElement('canvas');
    canvas.width = w * 2;
    canvas.height = h * 2;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const img = new Image();
    const url = URL.createObjectURL(new Blob([content], { type: 'image/svg+xml;charset=utf-8' }));
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      if (format === 'png') {
        canvas.toBlob(b => {
          const u = URL.createObjectURL(b);
          const a = document.createElement('a');
          a.href = u; a.download = `${outName}.png`; a.click();
          URL.revokeObjectURL(u);
        }, 'image/png');
      } else if (format === 'pdf' && window.jspdf) {
        const doc = new window.jspdf.jsPDF({ orientation: w >= h ? 'l' : 'p', unit: 'pt', format: [w, h] });
        doc.addImage(canvas.toDataURL('image/jpeg', 0.98), 'JPEG', 0, 0, w, h);
        doc.save(`${outName}.pdf`);
      }
    };
    img.src = url;
  }

  window.addEventListener('resize', resizeAll);
  window.exportChart = exportChart;
  window.OGDChartEngine = {
    esc,
    num,
    fmt,
    fmtInt,
    fmtP,
    choosePseudoLogC,
    txY,
    invTxY,
    makeWhiskerSeries,
    renderLegendBanner,
    init,
    bindRenderDock,
    resizeAll,
    exportChart,
  };
})();
