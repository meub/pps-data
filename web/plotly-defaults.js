// Site-wide Plotly defaults, loaded right after plotly.js on every page with charts.
// Horizontal bar charts: Plotly draws category labels almost touching the bars, so give
// the y axis of any chart with an orientation "h" trace a gap. ticklabelstandoff needs
// Plotly 2.30+; automargin lets the left margin grow if the gap pushes a long label out.
(function () {
  const LABEL_GAP = 10;  // px
  const wrap = (fn) => function (gd, data, layout, config) {
    if (Array.isArray(data)) {
      layout = layout || {};
      data.filter(t => t && t.orientation === "h").forEach(t => {
        const ax = "yaxis" + (t.yaxis ? t.yaxis.slice(1) : "");
        layout[ax] = Object.assign({ ticklabelstandoff: LABEL_GAP, automargin: true }, layout[ax]);
      });
    }
    return fn.call(this, gd, data, layout, config);
  };
  Plotly.newPlot = wrap(Plotly.newPlot);
  Plotly.react = wrap(Plotly.react);
})();
