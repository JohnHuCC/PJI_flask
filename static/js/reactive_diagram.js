console.log("Reactive Diagram js loaded");

let range_inputs = document.querySelectorAll(".range_input");
let inputs_args = document.querySelectorAll(".input_arg");


range_inputs.forEach(element => {
    let range_value = document.querySelector(`#${element.id}_value`);
    range_value.innerHTML = element.value;
    element.addEventListener("change", event => {
        console.log(`${element.id}: ${event.target.value}`);
        range_value.innerHTML = event.target.value;
    });
});

var g = new dagreD3.graphlib.Graph({compound:true}).setGraph({});
// g.setNode("Output", { label: "Output", style: "fill: #53FF53" , labelStyle: "font-weight: bold"});
g.setNode("Output", { label: `Output: ${result}`, style: "fill: #5eaa5f" , labelStyle: "font-weight: bold", width: 200});
g.setNode("MetaLearner", { label: `Meta Learner`, style: "fill: #5eaa5f" , labelStyle: "font-weight: bold", width: 150, height:150});
g.setNode("Xgboost", { label: `Xgboost`, style: "fill: #f9e54e" , labelStyle: "font-weight: bold", width: 150, height:150});
g.setNode("RandomForest", { label: `Random Forest`, style: "fill: #f8981d" , labelStyle: "font-weight: bold", width: 150, height:150});
g.setNode("NaiveBayes", { label: `Naive Bayes`, style: "fill: #eeb3a3" , labelStyle: "font-weight: bold", width: 150, height:150});
g.setNode("LogisticRegression", { label: `LogisticRegression`, style: "fill: #5bbdc8" , labelStyle: "font-weight: bold", width: 150, height:150});
g.setNode("FeatureGroups", {label:`Feature Groups`, clusterLabelPos: 'top', style: 'fill: #d3d7e8', labelStyle: "font-weight: bold"});
g.setNode("BaseClassifier", {label:`Base Classifier`, clusterLabelPos: 'top', style: 'fill: #d3d7e8', labelStyle: "font-weight: bold"});
g.setNode("DataInput", {label:`Input Data`, clusterLabelPos: 'top', style: 'fill: #d3d7e8', labelStyle: "font-weight: bold"});
function insertEdge(g, na, nb) {
    if (!g.hasEdge(na, nb)) {
        g.setEdge(na, nb, {})
    }
}
var count = 0
color_arr = ["#ece4e2", "#fcd3d1", "#fcd3d1", "#fcd3d1", "#fcd3d1", "#fcd3d1", "#fcd3d1", "#fcdff3", "#fcdff3", "#fe929f", "#fe929f", "#fe7773", "#fe7773", "#fe7773", "#fe7773",
"#fe7773", "#fe7773", "#fe7773", "#fe7773", ]
function insertNode(g, name) {
    if (!g.hasNode(name)) {
        inputs_args.forEach(element => {
            let comp_A = element.name.toLowerCase().replace('_', ' ')
            let comp_B = reactive_rule_map_json[name].toLowerCase().replace('_', ' ')
            if (element.type == "radio" && element.checked) {
                if (comp_B.includes(comp_A)) {
                    console.log(comp_B, comp_A, element.value)
                    g.setNode(name, { label: `${reactive_rule_map_json[name]}: ${element.value}`, style: "fill:" + color_arr[count] , labelStyle: "font-weight: bold", width: 200, height:40})
                    // g.setNode(name, { label: reactive_rule_map_json[name], style: "fill: #afa" , labelStyle: "font-weight: bold"})
                    count += 1
                }
                g.setParent(name, 'FeatureGroups');
            }
            else if (element.type != "radio" && comp_B.includes(comp_A)) {
                console.log(comp_B, comp_A, element.value)
                g.setNode(name, { label: `${reactive_rule_map_json[name]}: ${element.value}`, style: "fill:" + color_arr[count] , labelStyle: "font-weight: bold", width: 200, height:40})
                // g.setNode(name, { label: reactive_rule_map_json[name], style: "fill: #afa" , labelStyle: "font-weight: bold"})
                count += 1
                g.setParent(name, 'FeatureGroups');
            }
        })
        // g.setNode(name, { label: reactive_rule_map_json[name], style: "fill: #afa" , labelStyle: "font-weight: bold"})
    }
}

function processL(g, ns) {
  ns.forEach((n) => insertNode(g, n));
  for (let i = 0; i < (ns.length-1); i += 1) {
        insertEdge(g, ns[i], ns[i+1])
  }
  insertEdge(g, ns[ns.length-1], "DataInput")
}
insertEdge(g, "DataInput", "Xgboost")
insertEdge(g, "DataInput", "RandomForest")
insertEdge(g, "DataInput", "NaiveBayes")
insertEdge(g, "DataInput", "LogisticRegression")

function processLL(g, nss) {
    Object.keys(nss).forEach(k => { 
        let ns = nss[k]
        processL(g, ns)
    })
}


processLL(g, reactive_rule_json)
g.setEdge("Xgboost", "MetaLearner", {label: `${result_xgb}` , labelStyle: "font-weight: bold"});
g.setEdge("RandomForest", "MetaLearner", {label: `${result_rf}`, labelStyle: "font-weight: bold"});
g.setEdge("NaiveBayes", "MetaLearner", {label: `${result_nb}`, labelStyle: "font-weight: bold"});
g.setEdge("LogisticRegression", "MetaLearner", {label: `${result_lr}`, labelStyle: "font-weight: bold"});

g.setParent("Xgboost", 'BaseClassifier');
g.setParent("RandomForest", 'BaseClassifier');
g.setParent("NaiveBayes", 'BaseClassifier');
g.setParent("LogisticRegression", 'BaseClassifier');

g.setEdge("MetaLearner", "Output", {});

var svg = d3.select("#reactive_diagram_box").append("svg").attr("width", 1150).attr("height", 1500);
var inner = svg.append("g");


// Set the rankdir
g.graph().rankdir = "LR";
g.graph().nodesep = 60;
g.graph().ranksep = 150;
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
    .translate([(svg.attr("width") - 1100) , 0])
    .scale(initialScale)
    .event(svg);
// svg.attr('height', g.graph().height * initialScale + 200);
// svg.attr('width', g.graph().width * initialScale + 100);
