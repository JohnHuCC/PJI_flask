console.log('progress_bar.js')
// 连接到 Socket.IO 服务器
var socket = io.connect('http://' + document.domain + ':' + location.port);

// 当从服务器接收到 'task_progress' 事件时
socket.on('task_progress', function(data) {
    // 更新进度条
    var progressBar = document.getElementById('progress-bar');
    progressBar.value = data.progress;
});

let Revision = null;
let ASA_2 = null;
let positive_culture = null;
let Single_Positive_culture = null;
let Positive_Histology = null;
let Purulence = null;
if (Revision_0.checked){
    Revision = 0
} else {
    Revision = 1
}

if (ASA_2_0.checked){
    ASA_2 = 0
} else {
    ASA_2 = 1
}

if (positive_culture_0.checked){
    positive_culture = 0
} else {
    positive_culture = 1
}

if (Single_Positive_culture_0.checked){
    Single_Positive_culture = 0
} else {
    Single_Positive_culture = 1
}

if (Positive_Histology_0.checked){
    Positive_Histology = 0
} else {
    Positive_Histology = 1
}

if (Purulence_0.checked){
    Purulence = 0
} else {
    Purulence = 1
}

// 触发后端任务（这里只是一个示例，您可能有自己的触发方式）
function startTask() {
    var arr = [age.value, segment.value, HGB.value, PLATELET.value, Serum_WBC.value, P_T.value, APTT.value, CCI.value, Elixhauser.value, Revision, ASA_2, positive_culture
        ,Serum_CRP.value, Serum_ESR.value, Synovial_WBC.value, Single_Positive_culture, Synovial_PMN.value, Positive_Histology, Purulence]
    console.log('arr:', arr)
    var name = document.getElementById("model_diagnosis_link").dataset.myValue;
    console.log('name:', name)
    socket.emit('run_task', { arr: arr, name: name});

    socket.on('update_frontend', function(data) {
        window.location.href = "/reactive_diagram?p_id="+name;
        // fetch('/reactive_diagram?p_id='+name, {
        //     method: 'GET',
        //     headers: {
        //         'Content-Type': 'application/json'
        //     },
        //     body: JSON.stringify(data)
        // })
        // console.log('update frontend:', data)
        // document.getElementById('result_text').textContent = 'Result: '+ data.result_text;
        // const result_xgb = data.result_xgb;
        // const result_rf = data.result_rf;
        // const result_nb = data.result_nb;
        // const result_lr = data.result_lr;
    });

    // socket.on('update_frontend_data', function(data) {
    //     console.log('update frontend data:', data)
    //     var reactive_rule_json = data.reactive_rule_json;
    //     var reactive_rule_map_json = data.reactive_rule_map_json;
    //     var reactive_decision_list_json = data.reactive_decision_list_json;
    //     var reactive_decision_list_map_json = data.reactive_decision_list_map_json;
    //     var reactived_data_json = data.reactived_data_json;
    // });
    
}

// 例如，您可以在一个按钮的点击事件中调用 startTask()
document.getElementById('reactived_submit').addEventListener('click', startTask);