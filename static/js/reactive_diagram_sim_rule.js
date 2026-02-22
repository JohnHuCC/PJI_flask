console.log('from reactive_diagram_sim_rule.js')
let simplified_temp = "";
let diagnosis_simplified_result = document.getElementById("diagnosis_simplified_result");
Object.keys(reactive_decision_list_json).forEach((k, index1) => {
    path = reactive_decision_list_json[k];
    if (index1 != 0) {
        simplified_temp += "<h4>| ( ";
    } else {
        simplified_temp += "<h4>if ( ";
    }
    path.forEach((node, index2) => {
        if (index2 != path.length - 1) {
            simplified_temp += `${reactive_decision_list_map_json[node]} & `;
        } else {
            simplified_temp += `${reactive_decision_list_map_json[node]}`;
        }
    })
    simplified_temp += " )</h4><br>";
});
diagnosis_simplified_result.innerHTML = simplified_temp + "<h4>then infected</h4>";


