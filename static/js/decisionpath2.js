// Create a new directed graph
console.log('from decisionpath2.js')

var g = new dagreD3.graphlib.Graph().setGraph({});


g.setNode("start", { label: "start" });
g.setNode("infected", { label: "infected" });

Object.keys(decision_list_json).forEach(k => {
    path = decision_list_json[k]
    let hasCommon = true;
    path.forEach((node, index) => {
        // node == path[index]
        if (k == 0) {
            g.setNode([k, path[index]], { label: rule_map_json[node] });
            switch (index) {
                case 0: // 開始
                    g.setEdge("start", [k, path[index]], {});
                    break;
                case path.length - 1: // 最後
                    g.setEdge([k, path[index - 1]], [k, path[index]], {});
                    g.setEdge([k, path[index]], "infected", {});
                    break;
                default: // 中間
                    g.setEdge([k, path[index - 1]], [k, path[index]], {});
                    break;
            }
        } else {
            let lastCommon = null;
            let currCommon = null;
            if (hasCommon) {
                for (let i = k - 1; i >= 0; i--) { // all previous path
                    for (let j = 0; j <= index; j++) { // all previous node
                        if (decision_list_json[i][j] == path[j]) {
                            currCommon = [i, path[j]];
                        }
                    }
                    for (let j = 0; j < index; j++) { // all previous node
                        if (decision_list_json[i][j] != path[j]) {
                            hasCommon = false;
                        } else {
                            lastCommon = [i, path[j]];
                        }
                    }
                }
            }
            if (lastCommon != null && currCommon[1] != path[index]) {
                g.setNode([k, path[index]], { label: rule_map_json[node] });
                switch (index) {
                    case 0: // 開始
                        g.setEdge("start", [k, path[index]], {});
                        break;
                    case path.length - 1: // 最後
                        g.setEdge(lastCommon, [k, path[index]], {});
                        g.setEdge([k, path[index]], "infected", {});
                        break;
                    default: // 中間
                        g.setEdge(lastCommon, [k, path[index]], {});
                        break;
                }
            } else if (currCommon == null) {
                g.setNode([k, path[index]], { label: rule_map_json[node] });
                switch (index) {
                    case 0: // 開始
                        g.setEdge("start", [k, path[index]], {});
                        break;
                    case path.length - 1: // 最後
                        g.setEdge([k, path[index - 1]], [k, path[index]], {});
                        g.setEdge([k, path[index]], "infected", {});
                        break;
                    default: // 中間
                        g.setEdge([k, path[index - 1]], [k, path[index]], {});
                        break;
                }
            }
        }
    })
})


var svg = d3.select("#diagnosis_box").append("svg").attr("width", 1000).attr("height", 50);
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

// Center the graph
var initialScale = 0.75;
zoom
    .translate([(svg.attr("width") - g.graph().width * initialScale) / 2, 20])
    .scale(initialScale)
    .event(svg);
svg.attr('height', g.graph().height * initialScale + 40);
