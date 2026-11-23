CSS = r"""
:root{
  --paper:#f2eee3; --panel:#fbf9f2; --ink:#14315c; --ink2:#2d4b78; --text:#1d2533; --dim:#5e6a7c; --hair:#c9cfd9;
  --grid:rgba(20,49,92,.055); --red:#c2513a; --red-soft:rgba(194,81,58,.12); --teal:#1d7d6b; --teal-soft:rgba(29,125,107,.12);
  --amber:#a67112; --amber-soft:rgba(166,113,18,.13); --violet:#6b4fa0; --violet-soft:rgba(107,79,160,.12);
  --display:"Epilogue","Noto Serif SC",sans-serif;
  --body:"IBM Plex Serif","Noto Serif SC",serif;
  --mono:"Victor Mono","Noto Serif SC",ui-monospace,monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;color:var(--text);font:17px/1.72 var(--body);background:var(--paper);
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:24px 24px;background-attachment:fixed;-webkit-font-smoothing:antialiased}
a{color:var(--ink2);text-decoration-thickness:1px;text-underline-offset:3px} a:hover{color:var(--red)}
.wrap{display:grid;grid-template-columns:230px minmax(0,1fr);gap:44px;max-width:1220px;margin:0 auto;padding:34px 24px 80px}
.rail{position:sticky;top:24px;align-self:start;font:12px/1.55 var(--mono);color:var(--dim)}
.rail .back{display:inline-block;margin-bottom:18px;color:var(--red);text-decoration:none}
.rail ol{list-style:none;padding:0;margin:0;border-left:1px solid var(--ink)}
.rail li a{display:block;padding:4px 0 4px 13px;color:var(--dim);text-decoration:none}
.rail li a:hover{color:var(--ink)}
.rail .num{color:var(--ink);margin-right:6px}
.masthead{border:1.5px solid var(--ink);background:var(--panel);padding:26px 30px 22px;margin-bottom:38px;position:relative}
.masthead:after{content:"";position:absolute;inset:6px;border:1px solid var(--hair);pointer-events:none}
.kicker{font:600 12px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--red)}
h1{font:800 60px/1.02 var(--display);letter-spacing:-.01em;margin:10px 0 12px;color:var(--ink)}
h1 .slash{color:var(--red)}
.subtitle{font-size:18px;max-width:780px;margin:0 0 18px;color:var(--text)}
.spec{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-top:1px solid var(--ink);font:12px/1.5 var(--mono);position:relative;z-index:1}
.spec div{padding:9px 12px 9px 0;border-bottom:1px solid var(--hair)}
.spec b{display:block;color:var(--dim);font-weight:400;text-transform:uppercase;letter-spacing:.1em;font-size:10px}
main.article section{max-width:740px;margin:0 0 58px}
h2{font:800 32px/1.12 var(--display);margin:0 0 14px;color:var(--ink)}
h2 .n{font:600 12px var(--mono);color:var(--red);display:block;letter-spacing:.14em;margin-bottom:6px}
h3{font:700 18px/1.3 var(--display);color:var(--ink2);margin:26px 0 8px}
p{margin:0 0 14px}
ul,ol.list{margin:0 0 14px;padding-left:22px} li{margin:4px 0}
code{font:14px var(--mono);color:var(--ink);background:rgba(20,49,92,.07);padding:1px 4px;border-radius:2px;overflow-wrap:anywhere}
a code{color:inherit}
.callout{border-left:3px solid var(--teal);background:var(--teal-soft);padding:12px 16px;margin:18px 0}
.callout.warn{border-left-color:var(--red);background:var(--red-soft)}
.callout.note{border-left-color:var(--amber);background:var(--amber-soft)}
.callout b{color:var(--teal)} .callout.warn b{color:var(--red)} .callout.note b{color:var(--amber)}
.callout p:last-child{margin-bottom:0}
table{width:100%;border-collapse:collapse;margin:12px 0 18px;font-size:14.5px}
th,td{border-bottom:1px solid var(--hair);padding:7px 8px;text-align:left;vertical-align:top}
th{font:600 10.5px var(--mono);text-transform:uppercase;letter-spacing:.08em;color:var(--dim);border-bottom:1.5px solid var(--ink)}
td.v{font:13px var(--mono);text-align:right;white-space:nowrap}
table.terms td:first-child{width:32%;color:var(--ink);font-weight:600}
td.bad{color:var(--red);font:600 13px var(--mono);white-space:nowrap}
td.good{color:var(--teal);font:600 13px var(--mono);white-space:nowrap}
.plate{border:1.5px solid var(--ink);background:var(--panel);margin:24px 0 28px;padding:14px 18px 12px}
.plate.wide{width:min(900px,calc(100vw - 330px))}
.plate-meta{font:600 11px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--red);margin-bottom:8px}
.caption{font-style:italic;color:var(--dim);font-size:14.5px;margin:8px 0 0}
svg{display:block;width:100%;height:auto}
svg text{font-family:var(--mono)}
.nd{fill:var(--panel);stroke:var(--ink);stroke-width:1.3}
.nd.teal{fill:var(--teal-soft);stroke:var(--teal)}
.nd.red{fill:var(--red-soft);stroke:var(--red)}
.nd.amber{fill:var(--amber-soft);stroke:var(--amber)}
.nd.violet{fill:var(--violet-soft);stroke:var(--violet)}
.tx{font-size:12.5px;fill:var(--text)}
.tx.b{font-weight:600;fill:var(--ink)}
.tx.dim{fill:var(--dim);font-size:11.5px}
.tx.red{fill:var(--red)} .tx.teal{fill:var(--teal)} .tx.amber{fill:var(--amber)}
.edge{stroke:var(--ink);stroke-width:1.3;fill:none}
.edge.teal{stroke:var(--teal)} .edge.red{stroke:var(--red);stroke-dasharray:5 4} .edge.dash{stroke-dasharray:4 4}
.lane{font:600 11px var(--mono);letter-spacing:.12em;fill:var(--red)}
.ax-grid{stroke:var(--hair);stroke-width:1}
.ax-one{stroke:var(--ink);stroke-width:1.4}
.ax-lab{font-size:11px;fill:var(--dim)}
.row-lab{font-size:12px;fill:var(--text)}
.row-bad{font-size:12px;fill:var(--red);font-weight:600}
.val{font-size:12px;fill:var(--ink)} .val.small{font-size:10.5px;fill:var(--dim)}
.bar{stroke:none} .bar.d5{fill:var(--ink2)} .bar.prio{fill:var(--violet)} .bar.d4{fill:var(--amber)} .bar.pass{fill:var(--teal)}
.bar.ctrl{fill:var(--hair)} .bar.head{fill:var(--teal)} .bar.off{fill:var(--red)}
.whisk{stroke:var(--text);stroke-width:1.6}
.dot{stroke:none;opacity:.85} .dot.s1{fill:var(--ink2)} .dot.s2{fill:var(--red)} .dot.s3{fill:var(--teal)} .dot.s4{fill:var(--amber)}
.seg-trace{fill:var(--amber)} .seg-mlir{fill:var(--violet)} .seg-llvm{fill:var(--ink2)} .seg-warm{fill:var(--red)}
.legend{font:12px var(--mono);color:var(--dim);margin:6px 0 0}
.legend span{display:inline-block;margin-right:14px}
.legend i{display:inline-block;width:10px;height:10px;margin-right:5px;vertical-align:-1px}
.colophon{border-top:1.5px solid var(--ink);margin-top:44px;padding-top:16px;font:12px/1.7 var(--mono);color:var(--dim);max-width:760px}
.lang-toggle{position:fixed;top:16px;right:16px;z-index:10;display:flex;border:1.5px solid var(--ink);background:var(--panel)}
.lang-toggle button{font:600 12px var(--mono);background:none;border:0;color:var(--dim);padding:6px 12px;cursor:pointer}
body[data-lang="en"] .lang-toggle button[data-set="en"],body[data-lang="zh"] .lang-toggle button[data-set="zh"]{color:var(--panel);background:var(--ink)}
@media (max-width:980px){.wrap{grid-template-columns:minmax(0,1fr)}.rail{position:static}.plate.wide{width:auto}h1{font-size:44px}.spec{grid-template-columns:1fr 1fr}table{display:block;overflow-x:auto}}
@media (max-width:560px){.wrap{padding:20px 14px 60px;gap:24px}h1{font-size:34px}.spec{grid-template-columns:minmax(0,1fr)}}
body[data-lang="en"] [lang="zh"]:not(html){display:none !important}
body[data-lang="zh"] [lang="en"]:not(html){display:none !important}
"""

JS = r"""
(function() {
  var KEY = 'flydsl-compiler-redesign-source-lang', body = document.body, stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) {}
  if (stored === 'en' || stored === 'zh') { body.setAttribute('data-lang', stored); }
  else { var nav = (navigator.language || 'en').toLowerCase(); body.setAttribute('data-lang', nav.indexOf('zh') === 0 ? 'zh' : 'en'); }
  document.querySelectorAll('.lang-toggle button[data-set]').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var v = btn.getAttribute('data-set'); body.setAttribute('data-lang', v);
      try { localStorage.setItem(KEY, v); } catch (e) {}
    });
  });
})();
"""
