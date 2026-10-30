// OGD-LBS Minimal Experiment Dashboard

const defaultPalette = [
    '#1C328A',
    '#D35400',
    '#556B2F',
    '#8B0000',
    '#2C3E50',
    '#8E44AD',
    '#4D4D4D',
    '#9FA8FF',
    '#16A085',
    '#B7791F'
];

let colorPalette = (() => {
    try {
        const saved = localStorage.getItem('ogd_lbs_palette');
        if (!saved) return [...defaultPalette];
        const parsed = JSON.parse(saved);
        if (!Array.isArray(parsed) || parsed.length === 0) return [...defaultPalette];
        const result = [...defaultPalette];
        parsed.forEach((v, i) => {
            if (typeof v === 'string' && /^#[0-9a-fA-F]{6}$/.test(v)) result[i] = v;
        });
        return result;
    } catch (_) {
        return [...defaultPalette];
    }
})();

let customFontFamily = (() => {
    try {
        return localStorage.getItem('ogd_lbs_font_family') || 'Segoe UI';
    } catch (_) {
        return 'Segoe UI';
    }
})();

window.CUSTOM_FONT_BASE64 = null;
const allChartInstances = [];

if (customFontFamily) {
    document.documentElement.style.setProperty('--custom-font', customFontFamily);
}

function escapeHtml(v) {
    return String(v)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function finiteNumber(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}

function formatValue(v) {
    const n = finiteNumber(v);
    if (n === null) return '';
    if (Math.abs(n) >= 1000000 || (Math.abs(n) > 0 && Math.abs(n) < 0.001)) {
        return n.toExponential(3).replace('e+', 'e');
    }
    return Number(n.toPrecision(6)).toString();
}

function getColor(key) {
    const rows = (window.DASHBOARD_DATA && window.DASHBOARD_DATA.rows) || [];
    const keys = Array.from(new Set(rows.map(r => r.groupKey || r.param || r.split || r.model || r.display_id)));
    const idx = Math.max(0, keys.indexOf(key));
    return colorPalette[idx % colorPalette.length];
}

function initUI() {
    const settingsBtn = document.getElementById('settings-btn');
    const settingsPanel = document.getElementById('settings-panel');

    if (settingsBtn && settingsPanel) {
        settingsBtn.addEventListener('click', () => {
            settingsPanel.classList.toggle('hidden');
        });
    }

    const colorContainer = document.getElementById('color-pickers-container');
    if (colorContainer) {
        colorContainer.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const wrapper = document.createElement('div');
            wrapper.className = 'color-picker-wrapper';

            const input = document.createElement('input');
            input.type = 'color';
            input.value = colorPalette[i % colorPalette.length];

            const label = document.createElement('span');
            label.innerText = `C${i + 1}`;

            input.addEventListener('input', (e) => {
                colorPalette[i % colorPalette.length] = e.target.value;
                try {
                    localStorage.setItem('ogd_lbs_palette', JSON.stringify(colorPalette));
                } catch (_) {}
                renderAllCharts();
            });

            wrapper.appendChild(input);
            wrapper.appendChild(label);
            colorContainer.appendChild(wrapper);
        }
    }

    const fontUpload = document.getElementById('font-upload');
    if (fontUpload) {
        fontUpload.addEventListener('change', (e) => {
            const file = e.target.files && e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (evt) => {
                window.CUSTOM_FONT_BASE64 = evt.target.result;
                const fontName = 'OGDUserFont_' + Date.now();
                const newFont = new FontFace(fontName, `url(${window.CUSTOM_FONT_BASE64})`);

                newFont.load().then((loadedFont) => {
                    document.fonts.add(loadedFont);
                    customFontFamily = fontName;
                    try {
                        localStorage.setItem('ogd_lbs_font_family', fontName);
                    } catch (_) {}
                    document.documentElement.style.setProperty('--custom-font', fontName);
                    renderAllCharts();
                }).catch((err) => {
                    console.warn('Failed to load font:', err);
                });
            };
            reader.readAsDataURL(file);
        });
    }

    const resetBtn = document.getElementById('btn-reset-appearance');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            try {
                localStorage.removeItem('ogd_lbs_palette');
                localStorage.removeItem('ogd_lbs_font_family');
            } catch (_) {}
            location.reload();
        });
    }

    window.addEventListener('resize', () => {
        allChartInstances.forEach(c => {
            try { c.resize(); } catch (_) {}
        });
    });
}

window.exportChart = function(domId, format, filename) {
    const dom = document.getElementById(domId);
    const svgNode = dom ? dom.querySelector('svg') : null;

    if (!svgNode) {
        window.alert('SVG node not found. Ensure ECharts renderer is SVG.');
        return;
    }

    let svgContent = svgNode.outerHTML;

    if (!svgContent.includes('xmlns="http://www.w3.org/2000/svg"')) {
        svgContent = svgContent.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ');
    }

    if (window.CUSTOM_FONT_BASE64 && customFontFamily) {
        const fontStyle = `<defs><style>@font-face { font-family: "${customFontFamily}"; src: url(${window.CUSTOM_FONT_BASE64}); }</style></defs>`;
        svgContent = svgContent.replace(/(<svg[^>]*>)/i, '$1' + fontStyle);
    }

    if (format === 'svg') {
        const blob = new Blob([svgContent], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename + '.svg';
        a.click();
        URL.revokeObjectURL(url);
        return;
    }

    const bbox = svgNode.getBoundingClientRect();
    const width = parseFloat(svgNode.getAttribute('width')) || bbox.width || 1000;
    const height = parseFloat(svgNode.getAttribute('height')) || bbox.height || 600;

    if (format === 'png') {
        rasterizeSvg(svgContent, width, height, 2, (canvas) => {
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename + '.png';
                a.click();
                URL.revokeObjectURL(url);
            }, 'image/png');
        });
        return;
    }

    if (format === 'pdf') {
        generateImagePdf(svgContent, width, height, filename);
    }
};

function rasterizeSvg(svgContent, width, height, pixelRatio, callback) {
    const canvas = document.createElement('canvas');
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const img = new Image();
    const url = URL.createObjectURL(new Blob([svgContent], { type: 'image/svg+xml;charset=utf-8' }));

    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        callback(canvas);
    };

    img.onerror = () => {
        URL.revokeObjectURL(url);
        window.alert('Failed to rasterize SVG.');
    };

    img.src = url;
}

function generateImagePdf(svgContent, width, height, filename) {
    if (typeof jspdf === 'undefined' || !jspdf.jsPDF) {
        window.alert('PDF export library is unavailable.');
        return;
    }

    rasterizeSvg(svgContent, width, height, 3, (canvas) => {
        const imgData = canvas.toDataURL('image/jpeg', 0.98);
        const orientation = width >= height ? 'l' : 'p';
        const doc = new jspdf.jsPDF({
            orientation: orientation,
            unit: 'pt',
            format: [width, height]
        });
        doc.addImage(imgData, 'JPEG', 0, 0, width, height);
        doc.save(filename + '.pdf');
    });
}

function createChartBox(domId, title, parentNode, extraClass) {
    const box = document.createElement('div');
    box.className = 'img-box';

    const chartClass = extraClass || 'chart-wrapper';

    box.innerHTML = `
        <h3>${escapeHtml(title)}</h3>
        <div class="export-btn-container">
            <button class="btn-sm btn-svg" onclick="exportChart('${domId}', 'svg', '${domId}')">📥 SVG</button>
            <button class="btn-sm btn-png" onclick="exportChart('${domId}', 'png', '${domId}')">🖼️ PNG</button>
            <button class="btn-sm btn-pdf" onclick="exportChart('${domId}', 'pdf', '${domId}')">📄 PDF</button>
        </div>
        <div id="${domId}" class="${chartClass}"></div>
    `;

    parentNode.appendChild(box);
    return document.getElementById(domId);
}

function renderBars() {
    const D = window.DASHBOARD_DATA || {};
    const charts = D.charts || {};
    const bars = charts.bars || [];
    const container = document.getElementById('dynamic-charts-container');

    if (!container) return;

    if (!Array.isArray(bars) || bars.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'empty-section';
        empty.innerText = 'No bar-chart metrics found in aggregated_metrics.csv.';
        container.appendChild(empty);
        return;
    }

    bars.forEach((bar) => {
        if (!bar.items || bar.items.length === 0) return;

        const domId = 'chart-bar-' + bar.key;
        const dom = createChartBox(domId, bar.title, container);
        const chart = echarts.init(dom, null, { renderer: 'svg' });
        allChartInstances.push(chart);

        const labels = bar.items.map(i => i.display_id);
        const data = bar.items.map(i => {
            return {
                value: i.value,
                itemStyle: { color: getColor(i.groupKey || i.display_id) }
            };
        });

        chart.setOption({
            animation: false,
            textStyle: { fontFamily: customFontFamily },
            tooltip: {
                trigger: 'item',
                formatter: (p) => {
                    const item = bar.items[p.dataIndex];
                    return `<b>${escapeHtml(item.label)}</b><br/>${escapeHtml(bar.metric)}: <b>${formatValue(item.value)}</b>`;
                }
            },
            grid: { top: 30, bottom: 100, left: 90, right: 30 },
            xAxis: {
                type: 'category',
                data: labels,
                axisLabel: { rotate: 35, fontSize: 11, interval: 0 }
            },
            yAxis: {
                type: 'value',
                scale: true,
                name: bar.yName || bar.metric,
                nameLocation: 'middle',
                nameGap: 65
            },
            series: [
                {
                    type: 'bar',
                    barMaxWidth: 42,
                    data: data
                }
            ]
        });
    });
}

function buildParetoFilters() {
    const D = window.DASHBOARD_DATA || {};
    const pareto = (D.charts && D.charts.pareto) || {};
    const panel = document.getElementById('pareto-filters');

    if (!panel || !pareto.items || pareto.items.length === 0) return;

    let html = '';
    const filterKeys = pareto.filterKeys || [];

    filterKeys.forEach((key) => {
        const vals = Array.from(new Set(pareto.items.map(i => String((i.params || {})[key] ?? 'N/A')))).sort();
        if (vals.length < 2) return;

        html += `<div class="filter-group"><strong>${escapeHtml(key)}:</strong>`;
        vals.forEach((v) => {
            html += `<label><input type="checkbox" class="p-filter" data-key="${escapeHtml(key)}" value="${escapeHtml(v)}" checked> ${escapeHtml(v)}</label>`;
        });
        html += '</div>';
    });

    panel.innerHTML = html;
    document.querySelectorAll('.p-filter').forEach(el => el.addEventListener('change', renderPareto));
}

function renderPareto() {
    const D = window.DASHBOARD_DATA || {};
    const pareto = (D.charts && D.charts.pareto) || {};
    const dom = document.getElementById('chart-pareto');

    if (!dom) return;

    if (!pareto.items || pareto.items.length === 0) {
        const parent = dom.closest('.img-box');
        if (parent) parent.style.display = 'none';
        return;
    }

    let chart = echarts.getInstanceByDom(dom);

    if (!chart) {
        chart = echarts.init(dom, null, { renderer: 'svg' });
        allChartInstances.push(chart);
    }

    const selected = {};
    document.querySelectorAll('.p-filter').forEach((cb) => {
        const k = cb.getAttribute('data-key');
        if (!selected[k]) selected[k] = new Set();
        if (cb.checked) selected[k].add(cb.value);
    });

    const items = pareto.items.filter((it) => {
        for (const k in selected) {
            const val = String((it.params || {})[k] ?? 'N/A');
            if (!selected[k].has(val)) return false;
        }
        return true;
    });

    const series = items.map((it) => {
        return {
            name: it.display_id,
            type: 'scatter',
            symbolSize: 18,
            itemStyle: {
                color: getColor(it.groupKey || it.display_id),
                opacity: 0.85,
                borderColor: '#fff',
                borderWidth: 1
            },
            data: [[it.x, it.y]],
            tooltip: {
                formatter: () => {
                    return `<div style="max-width:360px;white-space:normal;">
                        <b>${escapeHtml(it.label)}</b><br/>
                        ${escapeHtml(pareto.xName)}: <b>${formatValue(it.x)}</b><br/>
                        ${escapeHtml(pareto.yName)}: <b>${formatValue(it.y)}</b>
                    </div>`;
                }
            }
        };
    });

    chart.setOption({
        animation: false,
        textStyle: { fontFamily: customFontFamily },
        tooltip: { trigger: 'item' },
        legend: { type: 'scroll', orient: 'vertical', right: 10, top: 40 },
        grid: { top: 40, left: 95, right: 210, bottom: 70 },
        xAxis: {
            type: 'value',
            name: pareto.xName,
            nameLocation: 'middle',
            nameGap: 40,
            scale: true
        },
        yAxis: {
            type: 'value',
            name: pareto.yName,
            nameLocation: 'middle',
            nameGap: 70,
            scale: true
        },
        series: series
    }, true);
}

function renderAllCharts() {
    allChartInstances.splice(0, allChartInstances.length);
    const container = document.getElementById('dynamic-charts-container');
    if (container) container.innerHTML = '';

    renderBars();
    renderPareto();
}

document.addEventListener('DOMContentLoaded', () => {
    initUI();
    buildParetoFilters();
    renderAllCharts();
});
