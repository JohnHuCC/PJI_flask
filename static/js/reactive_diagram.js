console.log("Reactive Diagram js loaded");

let range_inputs = document.querySelectorAll(".range_input");

range_inputs.forEach(element => {
    let range_value = document.querySelector(`#${element.id}_value`);
    range_value.innerHTML = element.value;
    element.addEventListener("change", event => {
        console.log(`${element.id}: ${event.target.value}`);
        range_value.innerHTML = event.target.value;
    });
});