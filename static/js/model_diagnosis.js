console.log('from model_diagnosis.js')
let temp = "";
let range_input = document.querySelectorAll(".range_input");
let diagnosis_result = document.getElementById("diagnosis_result");
// Object.keys(decision_list_json).forEach((k, index1) => {
//     path = decision_list_json[k];
//     if (index1 != 0) {
//         temp += "<h4>| ( ";
//     } else {
//         temp += "<h4>if ( ";
//     }
//     path.forEach((node, index2) => {
//         if (index2 != path.length - 1) {
//             temp += `${rule_map_json[node]} & `;
//         } else {
//             temp += `${rule_map_json[node]}`;
//         }
//     })
//     temp += " )</h4><br>";
// });
// diagnosis_result.innerHTML = temp + "<h4>then infected</h4>";

range_input.forEach(element => {
    let range_value = document.querySelector(`#${element.id}_value`);
    range_value.innerHTML = element.value;
    processDP(decision_list_json, range_value.innerHTML);
    element.addEventListener("change", event => {
        clear();
        console.log(`Range Value:${event.target.value}`);
        processDP(decision_list_json, event.target.value);
    });  
});

function clear(){
    temp = ""
    diagnosis_result.innerHTML = "";
}

function processDP(decision_list_json, max_num){
    Object.keys(decision_list_json).forEach((k, index1) => {
        if (Number(k) < max_num || max_num == -1) {
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
        }
    
    });
    
    diagnosis_result.innerHTML = temp + "<h4>then infected</h4>";
}


