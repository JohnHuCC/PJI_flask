window.addEventListener('DOMContentLoaded', event => {
    // Simple-DataTables
    // https://github.com/fiduswriter/Simple-DataTables/wiki

    const datatablesSimple = document.getElementById('datatablesSimple');
    if (datatablesSimple) {
        new simpleDatatables.DataTable(datatablesSimple);
    }

    const datatablesSimple_1 = document.getElementById('datatablesSimple_1');
    if (datatablesSimple_1) {
        new simpleDatatables.DataTable(datatablesSimple_1);
    }
    const datatablesSimple_2 = document.getElementById('datatablesSimple_2');
    if (datatablesSimple_2) {
        new simpleDatatables.DataTable(datatablesSimple_2);
    }

    const datatablesSimple_3 = document.getElementById('datatablesSimple_3');
    if (datatablesSimple_3) {
        new simpleDatatables.DataTable(datatablesSimple_3);
    }
});
