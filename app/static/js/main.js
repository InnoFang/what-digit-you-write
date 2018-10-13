window.onload = () => {
    const clear = document.getElementById("clear");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const lineWidth = 10;
    const lineColor = "#000000";
    const background = "#e8fffa";
    const canvasWidth = 561; // 20 * 28 + 1
    const canvasHeight = 561; // 20 * 28 + 1

    let isDrawing = false;
    let curPos; // current position
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;


    function getPosition(clientX, clientY) {
        let box = canvas.getBoundingClientRect();
        return {x: clientX - box.x, y: clientY - box.y}
    }

    function draw(e) {
        if (isDrawing) {
            let pos = getPosition(e.clientX, e.clientY);
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = lineWidth;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.beginPath();
            ctx.moveTo(curPos.x, curPos.y);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke()
            ctx.closePath();
            curPos = pos;
        }
    }

    canvas.onmousedown = function (e) {
        isDrawing = true;
        curPos = getPosition(e.clientX, e.clientY);
        draw(e);
    };

    canvas.onmousemove = function (e) {
        draw(e);
    };

    canvas.onmouseup = function (e) {
        isDrawing = false;
    };

    clear.onclick = function () {
        console.log("click");
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    }
};