// Create a new directed graph
console.log('from decisionpath.js')
var g = new dagreD3.graphlib.Graph().setGraph({});
let range_inputs = document.querySelectorAll(".range_input");
g.setNode("start", { label: "start" });
g.setNode("infected", { label: "infected" });

range_inputs.forEach(element => {
    let range_value = document.querySelector(`#${element.id}_value`);
    range_value.innerHTML = element.value;
    element.addEventListener("change", event => {
        console.log(`${element.id}: ${event.target.value}`);
        document.getElementById("diagnosis_box").innerHTML = "";
        range_value.innerHTML = element.value;
        var svg = d3.select("#diagnosis_box").append("svg").attr("width", 1300).attr("height", 500);
        var inner = svg.append("g");
        var g = new dagreD3.graphlib.Graph().setGraph({});
        g.setNode("start", { label: "start" });
        g.setNode("infected", { label: "infected" });
        g.graph().rankdir = "LR";
        g.graph().nodesep = 60;
        processLL(g, decision_list_json, event.target.value);
        // Set up zoom support
        var zoom = d3.behavior.zoom().on("zoom", function () {
            inner.attr("transform", "translate(" + d3.event.translate + ")" +
                "scale(" + d3.event.scale + ")");
        });
        svg.call(zoom);
        var render = new dagreD3.render();
        render(inner, g);
        // var initialScale = 0.95;
        var initialScale = 0.1;
        zoom
            .translate([(svg_rd_dp.attr("width") - g_rd_dp.graph().width * initialScale_rd_dp) , 20])
            // .translate([(svg.attr("width") - g.graph().width * initialScale) , 20])
            .scale(initialScale)
            .event(svg);
        svg.attr('height', g.graph().height * initialScale + 40);
            });
});

function insertEdge(g, na, nb) {
    if (!g.hasEdge(na, nb)) {
        g.setEdge(na, nb, {})
    }
}
function insertNode(g, name) {
    if (!g.hasNode(name)) {
        g.setNode(name, { label: rule_map_json[name] })
    }
}

function processL(g, ns) {
  ns.forEach((n) => insertNode(g, n));
  insertEdge(g, "start", ns[0])
  for (let i = 0; i < (ns.length-1); i += 1) {
        insertEdge(g, ns[i], ns[i+1])
  }
  insertEdge(g, ns[ns.length-1], "infected")
}

function processLL(g, nss, max_num) {
    Object.keys(nss).forEach(k => {
        if (Number(k) < max_num || max_num == -1) {
            let ns = nss[k]
            processL(g, ns)
        }
    })
}

processLL(g, decision_list_json, 5)

var svg = d3.select("#diagnosis_box").append("svg").attr("width", 1300).attr("height", 700);
var inner = svg.append("g");


// Set the rankdir
g.graph().rankdir = "LR";
g.graph().nodesep = 30;

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
var initialScale = 0.95;
// var initialScale = 2;
zoom     
    .translate([(svg.attr("width")/1.2 - g.graph().width * initialScale) , 100])
    .scale(initialScale)
    .event(svg);
svg.attr('height', g.graph().height * initialScale + 200);



/*

json:
    nss = { "key": [ "value1", "value2",  ], "key": [ "value1", "value2",  ] }
    Object.keys(decision_list_json).forEach(k => { 
        ns = decision_list_json[k]
    })

array:
    nss = [ [ "value1", "value2",  ], [ "value1", "value2",  ] ]
    nss.forEach(ns => {
        
    })


*/