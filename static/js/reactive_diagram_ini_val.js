console.log('rd_ini_val.js');
var feature_name = ["age", "segment", "HGB", "PLATELET", "Serum_WBC", "P_T", "APTT", "CCI", "Elixhauser"
,"Serum_CRP", "Serum_ESR", "Synovial_WBC", "Synovial_PMN"]
let age = document.getElementById("age");
let segment = document.getElementById("segment");
let HGB = document.getElementById("HGB");
let PLATELET = document.getElementById("PLATELET");
let Serum_WBC = document.getElementById("Serum_WBC");
let P_T = document.getElementById("P_T");
let APTT = document.getElementById("APTT");
let CCI = document.getElementById("CCI");
let Elixhauser = document.getElementById("Elixhauser");
let Serum_CRP = document.getElementById("Serum_CRP");
let Serum_ESR = document.getElementById("Serum_ESR");
let Synovial_WBC = document.getElementById("Synovial_WBC");
let Synovial_PMN = document.getElementById("Synovial_PMN");
let Revision_0 = document.getElementById("Revision_0");
let Revision_1 = document.getElementById("Revision_1");
let ASA_2_0 = document.getElementById("ASA_2_0");
let ASA_2_1 = document.getElementById("ASA_2_1");
let positive_culture_0 = document.getElementById("positive_culture_0");
let positive_culture_1 = document.getElementById("positive_culture_1");
let Single_Positive_culture_0 = document.getElementById("Single_Positive_culture_0");
let Single_Positive_culture_1 = document.getElementById("Single_Positive_culture_1");
let Positive_Histology_0 = document.getElementById("Positive_Histology_0");
let Positive_Histology_1 = document.getElementById("Positive_Histology_1");
let Purulence_0 = document.getElementById("Purulence_0");
let Purulence_1 = document.getElementById("Purulence_1");
// rangeInput.value = 1000; // 修改初始值
// rangeInput.min = 10; // 修改最小值
// rangeInput.max = 80; // 修改最大值
// feature_name.forEach((feature, columnIndex)=> {
//     let rangeInput = document.getElementById(feature);
//     rangeInput.value = 1000; // 修改初始值
// }); 

user_data_json.forEach((innerArray, rowIndex) => {
    age.value = innerArray[1];
    segment.value = innerArray[2];
    HGB.value = innerArray[3];
    PLATELET.value = innerArray[4];
    Serum_WBC.value = innerArray[5];
    P_T.value = innerArray[6];
    APTT.value = innerArray[7];
    CCI.value = innerArray[8];
    Elixhauser.value = innerArray[9];
    Serum_CRP.value = innerArray[13];
    Serum_ESR.value = innerArray[14];
    Synovial_WBC.value = innerArray[15];
    Synovial_PMN.value = innerArray[17];

    if (innerArray[10] == 0){
        Revision_0.checked = true;
        Revision_1.checked = false;
    } else {
        Revision_0.checked = false;
        Revision_1.checked = true;
    }
    
    if (innerArray[11] == 0){
        ASA_2_0.checked = true;
        ASA_2_1.checked = false;
    } else {
        ASA_2_0.checked = false;
        ASA_2_1.checked = true;
    }

    if (innerArray[12] == 0){
        positive_culture_0.checked = true;
        positive_culture_1.checked = false;
    } else {
        positive_culture_0.checked = false;
        positive_culture_1.checked = true;
    }

    if (innerArray[16] == 0){
        Single_Positive_culture_0.checked = true;
        Single_Positive_culture_1.checked = false;
    } else {
        Single_Positive_culture_0.checked = false;
        Single_Positive_culture_1.checked = true;
    }

    if (innerArray[18] == 0){
        Positive_Histology_0.checked = true;
        Positive_Histology_1.checked = false;
    } else {
        Positive_Histology_0.checked = false;
        Positive_Histology_1.checked = true;
    }

    if (innerArray[19] == 0){
        Purulence_0.checked = true;
        Purulence_1.checked = false;
    } else {
        Purulence_0.checked = false;
        Purulence_1.checked = true;
    }
});
 