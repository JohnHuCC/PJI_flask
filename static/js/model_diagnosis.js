let temp = "";
Object.keys(decision_list_json).forEach((k, index1) => {
    path = decision_list_json[k];
    if (index1 != 0) {
        temp += "<h4>| ( ";
    } else {
        temp += "<h4>if ( ";
    }
    path.forEach((node, index2) => {
        if (index2 != path.length - 1) {
            temp += `${rule_map_json[node]} & `;
        } else {
            temp += `${rule_map_json[node]}`;
        }
    })
    temp += " )</h4><br>";
});

let diagnosis_result = document.getElementById("diagnosis_result");
diagnosis_result.innerHTML = temp + "<h4>then infected</h4>";