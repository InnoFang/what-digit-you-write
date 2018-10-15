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
        // let img = new Image();
        //
        // img.onload = function () {
        //
        //     picCtx.drawImage(img, 0, 0, picture.width, picture.height);
        //     const imgData = picCtx.getImageData(0, 0, picture.width, picture.height).data;
        //     let inputs = [];
        //     let total = 0;
        //     console.log(imgData);
        //     for (let i = 0; i < picture.width; ++i) {
        //         for (let j = 0; j < picture.height; ++j) {
        //             const idx = (i * picture.width + j) * 4;
        //             const r = imgData[idx];
        //             const g = imgData[idx + 1];
        //             const b = imgData[idx + 2];
        //             // const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        //             const gray = (r + g + b) / 3;
        //             inputs[i * 28 + j] = gray;
        //             total += gray;
        //             // imgData[idx] = imgData[idx + 1] = imgData[idx + 2] = gray;
        //             // imgData[idx + 3] = 255;
        //             // console.log("r: " + r + ", g: " + g + ", b: " + b + ", gray: " + gray)
        //             // picCtx.fillStyle = 'rgb(' + [0.299 * r, 0.587 * g, 0.114 * b].join(',') + ')';
        //             // picCtx.fillRect(i * 5, j * 5, 5, 5);
        //         }
        //     }
        //     // picCtx.putImageData(imgData, 0, 0);
        //     console.log(total);
        //
        // };
        const img = new Image();
        img.onload = function () {
            let inputs = [];
            let total = 0;
            const small = document.createElement("canvas").getContext('2d');
            picCtx.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
            let data = picCtx.getImageData(0, 0, 28, 28).data;
            for (let i = 0; i < 28; ++i) {
                for (let j = 0; j < 28; ++j) {
                    let px = 4 * (i * 28 + j);
                    let r = data[px];
                    let g = data[px + 1];
                    let b = data[px + 2];
                    inputs[i * 28 + j] = (r + g + b) / 3;
                    total += (r + g + b) / 3;
                    // picCtx.fillStyle = 'rgb(' + [r, g, b].join(',') + ")";
                    // picCtx.fillRect(i * 5, j * 5, 5, 5)
                }
            }
            console.log(total);
        };
        img.src = canvas.toDataURL();
    };

    reset.onclick = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        picCtx.clearRect(0, 0, picture.width, picture.height);
    }
};