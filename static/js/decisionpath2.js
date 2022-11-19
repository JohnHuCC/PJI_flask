  // Create a new directed graph
  console.log('from decisionpath2.js')

  Object.keys(decision_list_json).forEach(k => {
    path = decision_list_json[k]
    var g = new dagreD3.graphlib.Graph().setGraph({});
    path.forEach( node => {
      g.setNode(node, { label: rule_map_json[node] });
    })
    
    for (var i = 0; i < path.length - 1; i++) {
      g.setEdge(path[i], path[i+1], { label: "" });
    }
    var svg = d3.select("#diagnosis_box").append("svg").attr("width", 1000).attr("height", 50);
    var inner = svg.append("g");
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
  })
