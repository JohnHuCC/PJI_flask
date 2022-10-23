  // Create a new directed graph
  console.log(decision_list,'from decisionpath.js')
  var g = new dagreD3.graphlib.Graph().setGraph({});

  var nodes = [ "Start", "Histology", "Serum CRP", "PT",
                 "Neutrophil Segment", "APTT", "Serum ESR", "Synovial WBC", "Infected"
              ];
  
  // Automatically label each of the nodes
  nodes.forEach(function(node) {
      g.setNode(node, { label: node }); 
  });
  
  //LV1
  g.setEdge("Start", "Histology", { label: "" });
  g.setEdge("Start", "APTT", { label: "" });
  //LV2
  g.setEdge("Histology", "Serum CRP", { label: "positive" });
  g.setEdge("Histology", "PT", { label: "positive" });
  g.setEdge("APTT", "Serum ESR", { label: "≥ 28.0" });
  //LV3 
  g.setEdge("Serum CRP", "Infected", { label: "≥ 16.0" });
  g.setEdge("PT", "Neutrophil Segment", { label: "≥ 9.8" });
  g.setEdge("Serum ESR", "Synovial WBC", { label: "≥ 38.0" });
  //LV4
  g.setEdge("Neutrophil Segment", "Infected", { label: "≥ 62.0" });
  g.setEdge("Synovial WBC", "Infected", { label: "≥ 3780.0" });
//   g.setEdge("148570025_1100", "148570010_1100", { label: "" });
//   g.setEdge("148570025_1100", "148570026_1100", { label: "" });
//   g.setEdge("148570021_1100", "148570022_1100", { label: "" });
//   g.setEdge("148570010_1100", "148570011_1100", { label: "" });
//   g.setEdge("148570010_1100", "148570010_1200", { label: "" });
//   g.setEdge("148570020_1100", "148570020_1200", { label: "" });
//   g.setEdge("148570026_1100", "148570026_1200", { label: "" });
//   g.setEdge("148570026_1200", "148570011_1200", { label: "" });
//   g.setEdge("148570010_1200", "148570011_1200", { label: "" });
//   g.setEdge("148570022_1100", "148570023_1100", { label: "" });
//   g.setEdge("148570023_1100", "148570023_1200", { label: "" });
  
  var svg = d3.select("svg"),
      inner = svg.select("g");
  
  // Set the rankdir
  g.graph().rankdir = "LR";
  g.graph().nodesep = 60;
  
  // Set up zoom support
  var zoom = d3.behavior.zoom().on("zoom", function() {
        inner.attr("transform", "translate(" + d3.event.translate + ")" +
                                    "scale(" + d3.event.scale + ")");
      });
  svg.call(zoom);
  
  // Create the renderer
  var render = new dagreD3.render();
  
  
  // Run the renderer. This is what draws the final graph.
  render(inner, g);
  
  // Center the graph
  var initialScale = 0.75;
  zoom
    .translate([(svg.attr("width") - g.graph().width * initialScale) / 2, 20])
    .scale(initialScale)
    .event(svg);
  svg.attr('height', g.graph().height * initialScale + 40);