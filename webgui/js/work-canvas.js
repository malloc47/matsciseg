var workcanvas = (function () {

    var ctx;
    var canvas;
    var mainimg;
    var width, height;

    function imgLoad() {
	ctx.globalAlpha = 1.0;
	drawImg(mainimg);
	ctx.redraw();
    }

    function init(w,h,c){
	width = w;
	height = h;
	canvas = c;
	ctx = canvas[0].getContext('2d');
	if(canvas.attr('width') != width*2)
	    canvas.attr('width',width*2);
	if(canvas.attr('height') != height*2)
	    canvas.attr('height',height*2);
	mainimg = new Image();
	mainimg.src = "/img/image0090.png"; // static img default
	mainimg.onload = imgLoad;
    };

    function setSrc(src) {mainimg.src = src;};

    function drawImg(img) {
	ctx.drawImage(img,0,0,width,height,0,0,width*2,height*2);
    }

    function loading() {
	ctx.fillStyle = '#FFFFFF';
	ctx.fillRect(0,0,width*2,height*2);
	ctx.globalAlpha = 0.25;
	drawImg(mainimg);
    }

    function fillCircle(x, y, radius, fillColor) {
        ctx.fillStyle = fillColor;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.arc(x, y, radius, 0, Math.PI * 2, false);
        ctx.fill();
    };

    function fillX(x, y, radius, fillColor) {
	fillCircle(x,y,radius,'#000000');
        ctx.strokeStyle = fillColor;
        ctx.beginPath();
        ctx.moveTo(x, y);
	ctx.lineTo(x+radius,y+radius);
        ctx.moveTo(x, y);
	ctx.lineTo(x-radius,y-radius);
        ctx.moveTo(x, y);
	ctx.lineTo(x+radius,y-radius);
        ctx.moveTo(x, y);
	ctx.lineTo(x-radius,y+radius);
        ctx.stroke();
    }

    function onredraw(f) {ctx.redraw = f;}

    return {
	init		: init,
	src		: setSrc,
	loading		: loading,
	fillCircle	: fillCircle,
	fillX		: fillX,
	onredraw	: onredraw,
	redraw		: imgLoad
    }
}());