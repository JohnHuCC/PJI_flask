// Create a new directed graph
console.log('from reactive_diagram_dp_draw.js')
let g_rd_dp = new dagreD3.graphlib.Graph().setGraph({});
g_rd_dp.setNode("start", { label: "start" });
g_rd_dp.setNode("infected", { label: "infected" });

function insertEdge(g, na, nb) {
    if (!g.hasEdge(na, nb)) {
        g.setEdge(na, nb, {})
    }
}
function insertNode(g, name) {
    if (!g.hasNode(name)) {
        g.setNode(name, { label: reactive_decision_list_map_json[name] })
    }
}

function processL(g, ns) {
  ns.forEach((n) => insertNode(g, n));
  insertEdge(g, "start", ns[0])
  for (var i = 0; i < (ns.length-1); i += 1) {
        insertEdge(g, ns[i], ns[i+1])
  }
  insertEdge(g, ns[ns.length-1], "infected")
}

function processLL(g, nss) {
    Object.keys(nss).forEach(k => { 
        var ns = nss[k]
        processL(g, ns)
    })
}


processLL(g_rd_dp, reactive_decision_list_json)

var svg_rd_dp = d3.select("#diagnosis_box_reactive").append("svg").attr("width", 1200).attr("height", 400);
var inner_rd_dp = svg_rd_dp.append("g");

// Set the rankdir
g_rd_dp.graph().rankdir = "LR";
g_rd_dp.graph().nodesep = 15;

// Set up zoom support
var zoom_rd_dp = d3.behavior.zoom().on("zoom", function () {
    inner_rd_dp.attr("transform", "translate(" + d3.event.translate + ")" +
        "scale(" + d3.event.scale + ")");
});

svg_rd_dp.call(zoom_rd_dp);

// Create the renderer
var render_rd_dp = new dagreD3.render();

// Run the renderer. This is what draws the final graph.
render_rd_dp(inner_rd_dp, g_rd_dp);
// // Create the renderer
// var render = new dagreD3.render();
// // Draw graph
// render(inner, g);

// Center the graph
var initialScale_rd_dp = 0.1;
zoom
    .translate([(svg_rd_dp.attr("width") - g_rd_dp.graph().width * initialScale_rd_dp) , 20])
    .scale(initialScale_rd_dp)
    .event(svg_rd_dp);
svg_rd_dp.attr('height', g_rd_dp.graph().height * initialScale_rd_dp + 40);

