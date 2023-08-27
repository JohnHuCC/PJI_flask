console.log('from model_diagnosis_sp.js')
let simplified_temp = "";
// let range_input = document.querySelectorAll(".range_input");
let diagnosis_simplified_result = document.getElementById("diagnosis_simplified_result");
Object.keys(simplified_rule_json).forEach((k, index1) => {
    path = simplified_rule_json[k];
    if (index1 != 0) {
        simplified_temp += "<h4>| ( ";
    } else {
        simplified_temp += "<h4>if ( ";
    }
    path.forEach((node, index2) => {
        if (index2 != path.length - 1) {
            simplified_temp += `${simplified_rule_map_json[node]} & `;
        } else {
            simplified_temp += `${simplified_rule_map_json[node]}`;
        }
    })
    simplified_temp += " )</h4><br>";
});
diagnosis_simplified_result.innerHTML = simplified_temp + "<h4>then infected</h4>";


