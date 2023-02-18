var g = new dagreD3.graphlib.Graph().setGraph({});
g.setNode("Output", { label: "Output" });

function insertEdge(g, na, nb) {
    if (!g.hasEdge(na, nb)) {
        g.setEdge(na, nb, {})
    }
}
function insertNode(g, name) {
    if (!g.hasNode(name)) {
        g.setNode(name, { label: reactive_diagram_json[name] })
    }
}

function processL(g, ns) {
  ns.forEach((n) => insertNode(g, n));
//   insertEdge(g, "start", ns[0])
  for (let i = 0; i < (ns.length-1); i += 1) {
        insertEdge(g, ns[i], ns[i+1])
  }
  insertEdge(g, ns[ns.length-1], "Output")
}

function processLL(g, nss) {
    Object.keys(nss).forEach(k => { 
        let ns = nss[k]
        processL(g, ns)
    })
}


processLL(g, reactive_diagram_json)

var svg = d3.select("#reactive_diagram_box").append("svg").attr("width", 1000).attr("height", 50);
var inner = svg.append("g");


// Set the rankdir
g.graph().rankdir = "LR";
g.graph().nodesep = 60;

// Set up zoom support
var zoom = d3.behavior.zoom().on("zoom", function () {
    inner.attr("transform", "translate(" + d3.event.translate + ")" +
        "scale(" + d3.event.scale + ")");
});
svg.call(zoom);

// Create the renderer
var render = new dagreD3.render();

// Run the renderer. This is what draws the final graph.
render(inner, g);

// // Create the renderer
// let render = new dagreD3.render();
// // Draw graph
// render(inner, g);

// Center the graph
var initialScale = 0.6;
zoom
    .translate([(svg.attr("width") - g.graph().width * initialScale) , 20])
    .scale(initialScale)
    .event(svg);
svg.attr('height', g.graph().height * initialScale + 40);