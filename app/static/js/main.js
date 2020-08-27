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

    /* The purpose of the first clear is to fill the canvas with a white background. */
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
            const input = document.createElement("canvas").getContext('2d');
            /* Map the original data size to 28*28 (28 * 28 = 784) */
            input.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
            let data = input.getImageData(0, 0, 28, 28).data;
            for (let i = 0; i < 28; ++i) {
                for (let j = 0; j < 28; ++j) {
                    let px = 4 * (i * 28 + j);
                    let r = data[px];
                    let g = data[px + 1];
                    let b = data[px + 2];
                    inputs[i * 28 + j] = (r + g + b) / 3;
                    /* Map the pixels of canvas `input` to canvas `picture` */
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
                    for (let i = 0; i < 2; ++i) {
                        let max = 0;
                        let max_idx = 0;
                        for (let j = 0; j < 10; ++j) {
                            let value = ret.data[i][j];
                            let tmp = value * 1000;
                            if (tmp > max) {
                                max = tmp;
                                max_idx = j;
                            }
                            /* Format the predicted value with three decimal */
                            value = value.toFixed(3);
                            $('.result tr').eq(j + 1).find('td').eq(i).text(value);
                        }

                        /* Add a style to the maximum */
                        for (let j = 0; j < 10; ++j) {
                            let node = $('.result tr').eq(j + 1).find('td').eq(i);
                            if (j === max_idx) {
                                node.addClass('answer');
                            } else {
                                node.removeClass('answer');
                            }
                        }
                    }
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

        $('.result td').text('').removeClass('answer');
    }
};