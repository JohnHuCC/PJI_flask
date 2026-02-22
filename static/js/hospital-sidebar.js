window.addEventListener('DOMContentLoaded', function () {
    var STORAGE_KEY = 'hospital|sidenav-compact-v2';
    if (!document.body.classList.contains('hospital-ui')) {
        return;
    }

    var sidebarToggle = document.getElementById('sidebarToggle');
    var content = document.getElementById('layoutSidenav_content');

    if (!sidebarToggle) {
        return;
    }

    function isDesktop() {
        return window.innerWidth >= 992;
    }

    function applyDesktopState() {
        var saved = localStorage.getItem(STORAGE_KEY);
        var compact = saved === null ? true : saved === 'true';
        // Neutralize legacy SB class to avoid conflicting layout behavior.
        document.body.classList.remove('sidenav-toggled', 'sb-sidenav-toggled');
        document.body.classList.toggle('hospital-sidenav-compact', compact);
        document.body.classList.remove('hospital-sidenav-open');
    }

    function clearMobileState() {
        document.body.classList.remove('hospital-sidenav-open');
    }

    if (isDesktop()) {
        applyDesktopState();
    } else {
        clearMobileState();
    }

    // Capture phase to prevent the default SB listener in scripts.js from firing.
    sidebarToggle.addEventListener('click', function (event) {
        event.preventDefault();
        event.stopPropagation();
        if (event.stopImmediatePropagation) {
            event.stopImmediatePropagation();
        }
        document.body.classList.remove('sidenav-toggled', 'sb-sidenav-toggled');

        if (isDesktop()) {
            var next = !document.body.classList.contains('hospital-sidenav-compact');
            document.body.classList.toggle('hospital-sidenav-compact', next);
            localStorage.setItem(STORAGE_KEY, String(next));
        } else {
            document.body.classList.toggle('hospital-sidenav-open');
        }
    }, true);

    if (content) {
        content.addEventListener('click', function () {
            if (!isDesktop()) {
                clearMobileState();
            }
        });
    }

    window.addEventListener('resize', function () {
        if (isDesktop()) {
            applyDesktopState();
        } else {
            clearMobileState();
        }
    });
});
