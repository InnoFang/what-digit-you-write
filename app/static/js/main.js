window.onload = () => {
    const reset = document.getElementById("reset");
    const picture = document.getElementById("picture");
    const canvas = document.getElementById("canvas");
    const picCtx = picture.getContext("2d");
    const ctx = canvas.getContext("2d");
    const lineWidth = 15;
    const lineColor = "#000000";
    const canvasWidth = 561; // 20 * 28 + 1
    const canvasHeight = 561; // 20 * 28 + 1
    const pictureWidth = 140; // 5 * 28
    const pictureHeight = 140; // 5 * 28

    let isDrawing = false;
    let curPos; // current position
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    picture.width = pictureWidth;
    picture.height = pictureHeight;

    clear();

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
            ctx.stroke();
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

        const img = new Image();
        img.src = canvas.toDataURL();
        img.onload = function () {
            let inputs = [];
            const c = document.createElement("canvas");
            const input = c.getContext('2d');
            input.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
            let data = input.getImageData(0, 0, 28, 28).data;
            console.log(c.toDataURL());
            for (let i = 0; i < 28; ++i) {
                for (let j = 0; j < 28; ++j) {
                    let px = 4 * (i * 28 + j);
                    let r = data[px];
                    let g = data[px + 1];
                    let b = data[px + 2];
                    inputs[i * 28 + j] = (r + g + b) / 3;
                    picCtx.fillStyle = 'rgb(' + [r, g, b].join(',') + ")";
                    picCtx.fillRect(j * 5, i * 5, 5, 5)
                }
            }

            $.ajax({
                url: "/predict",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(inputs),
                success: (ret) => {
                    console.log(ret)
                }
            });

        };
    };

    reset.onclick = clear;

    function clear() {
        ctx.fillStyle = "#ffffff";
        picCtx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
        picCtx.fillRect(0, 0, picture.width, picture.height);
    }
};