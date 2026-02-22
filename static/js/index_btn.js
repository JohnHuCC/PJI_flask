$(() => {
    function load_circle(){
        // console.log("circle clicked")
        $('#example').shCircleLoader({
            // border radius
            // "auto" - calculate from selector's width and height
            radius: "auto",
            dotsRadius: "auto",

            // color
            // "auto" - get from selector's color CSS property
            color: "auto",

            // number of dots
            dots: 12,

            // animation speed
            duration: 1,

            // clockwise or counterclockwise
            clockwise: true,

            // true - don't apply CSS from the script
            externalCss: false, 

            // customize the animation
            keyframes: '0%{{prefix}transform:scale(1)}80%{{prefix}transform:scale(.3)}100%{{prefix}transform:scale(1)}',
        });
        let i = 0;
        setInterval(function(){
            if (++i > 100) $('#example').shCircleLoader('destroy');
            // if (++i > 50) $.ajax({
            //     type: "GET",
            //     url: "/progress", //localhost Flask
            //     success: function (data) {
            //         if (data == "1") {
            //             $('#example').shCircleLoader('progress', i + '%');
            //         }
            //     }
            // });
        }, 200)
    }
    $(document).on("click", ".editbtn", function () {
        console.log(this.getAttribute("data-id"));
        load_circle()
        $.ajax({
            type: "POST",
            url: "/", //localhost Flask
            data: JSON.stringify({ btnID: this.getAttribute("data-btn"), patientID: this.getAttribute("data-id") }),
            contentType: "application/json",
            success: function (data) {
                window.location.href = data;
            }
        });

        return false;
    });

});