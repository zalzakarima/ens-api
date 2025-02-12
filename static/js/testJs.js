$(document).on("click", ".btn-modal", function (e) {
    var values = e.currentTarget.getAttribute("id");
    console.log(values);

    $.ajax({
        type: "POST",
        url: "http://" + document.domain + ":" + location.port + "/admin/status-1",
        data: {
            clicked_data: values,
            submit_button: "",
        },
        dataType: "html",
        // success: function(res){
        //     console.log(res);
        // }
    });
    $("body").addClass("show-modal");
});

$(document).on("click", ".modal-popup .backdrop, .btn-pop-close", function () {
    $("body").removeClass("show-modal");
    var table = document.getElementById("dataBody");
    var rowCount = table.rows.length;
    console.log(rowCount);
    for (var i = 0; i < rowCount; i++) {
        table.deleteRow(0);
    }
    // var child_table_rec = document.getElementById("ul_rec");
    // var table_rec = document.getElementById("img_rec");
    // table_rec.removeChild(child_table_rec);

    // var child_table_proc = document.getElementById("ul_proc");
    // var table_proc = document.getElementById("img_proc");
    // table_proc.removeChild(child_table_proc);

    // document.getElementById("reception1").removeAttribute("class");
    // document.getElementById("reception2").removeAttribute("class");
    // document.getElementById("reception3").removeAttribute("class");
});
$(document).ready(function () {
    var socket = io.connect("http://" + document.domain + ":" + location.port);
    socket.on("status1", function (msg) {
        console.log(msg.kkk[2]);
        const table = document.getElementById("dataBody");
        for (var i = 0; i < msg.kkk.length; i++) {
            let row = table.insertRow();
            for (var j = 0; j < 17; j++) {
                if (j == 0) {
                    let aaa = row.insertCell(j);
                    aaa.innerHTML = msg.kkk[i];
                } else if (j != 0) {
                    let aaa = row.insertCell(j);
                    aaa.innerHTML = 0;
                }
            }
        }
    });
    socket.on("pcs_data", function (msg) {
        //   console.log(msg)
        const data = JSON.parse(msg.data);
        const dtime = msg.time;
        // console.log(msg.firstR)
        let regionArray = [
            "서울",
            "강원",
            "대전",
            "충남",
            "세종",
            "충북",
            "인천",
            "경기",
            "광주",
            "전남",
            "전북",
            "부산",
            "경남",
            "울산",
            "제주",
            "대구",
            "경북",
            "총계",
        ];
        let types = ["id", "acPowerFactor", "nowKw", "acAR", "acAS", "acAT", "acFreq", "acVR", "acVS", "acVT", "actKw", "battA"];

        regionArray.forEach((id) => {
            let element = document.getElementById(id).children;
            t = Array.from(element).map((node) => node);
            t.splice(0, 1);
            t.forEach((item, index) => {
                item.innerHTML = data[types[index % types.length]];
            });
        });

        let element2 = document.getElementById("first").children;
        t = Array.from(element2).map((node) => node);
        t.splice(0, 1);
        t.forEach((item, index) => {
            if (index > 3) {
                item.innerHTML = msg.firstR[index - 1];
                // console.log(msg.firstR[index])
            } else if (index < 3) {
                item.innerHTML = msg.firstR[index];
                // console.log(
            } else if (index == 3) {
                // item.innerHTML = msg.firstR[index-1];
                // console.log(msg.firstR[index])
            }
        });

        let element3 = document.getElementById("second").children;
        t = Array.from(element3).map((node) => node);
        t.splice(0, 1);
        t.forEach((item, index) => {
            if (index > 3) {
                item.innerHTML = msg.firstR[index - 1];
                // console.log(msg.firstR[index])
            } else if (index < 3) {
                item.innerHTML = msg.secondR[index];
                // console.log(
            } else if (index == 3) {
                // item.innerHTML = msg.firstR[index-1];
                // console.log(msg.firstR[index])
            }
        });

        let element4 = document.getElementById("third").children;
        t = Array.from(element4).map((node) => node);
        t.splice(0, 1);
        t.forEach((item, index) => {
            if (index > 3) {
                item.innerHTML = msg.firstR[index - 1];
                // console.log(msg.firstR[index])
            } else if (index < 3) {
                item.innerHTML = msg.thirdR[index];
                // console.log(
            } else if (index == 3) {
                // item.innerHTML = msg.firstR[index-1];
                // console.log(msg.firstR[index])
            }
        });

        let element5 = document.getElementById("normal").children;
        t = Array.from(element5).map((node) => node);
        t.splice(0, 1);
        t.forEach((item, index) => {
            item.innerHTML = msg.normalD[index];
        });

        let element6 = document.getElementById("commError").children;
        t = Array.from(element6).map((node) => node);
        t.splice(0, 1);
        t.forEach((item, index) => {
            item.innerHTML = msg.commED[index];
        });

        let element7 = document.getElementById("warning").children;
        t = Array.from(element7).map((node) => node);
        t.splice(0, 1);
        t.forEach((item, index) => {
            item.innerHTML = msg.warningD[index];
        });
        let element8 = document.getElementById("broken").children;
        t = Array.from(element8).map((node) => node);
        t.splice(0, 1);
        t.forEach((item, index) => {
            item.innerHTML = msg.brokenD[index];
        });

        $("#totPlaces").text(msg.totPlaces).html();
        $("#current_time").text(dtime).html();
    });
});
