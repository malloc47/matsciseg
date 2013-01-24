var workcanvas = (function () {
    "use strict"

    var ctx;
    var canvas;
    var mainimg;
    var img_width
    var img_height;
    var zoom = 2;

    function redraw() {
	if(canvas.attr('width') != img_width*zoom)
	    canvas.attr('width',img_width*zoom);
	if(canvas.attr('height') != img_height*zoom)
	    canvas.attr('height',img_height*zoom);
	ctx.globalAlpha = 1.0;
        drawImg(mainimg);
        ctx.redraw();
    }

    function imgLoad() {
        var w = this.width;
        var h = this.height;
        img_width = w;
        img_height = h;
        redraw();
    };

    function init(c,s){
	canvas = c;
	ctx = canvas[0].getContext('2d');
        mainimg = new Image();
	mainimg.onload = imgLoad;
        mainimg.src = s;
    };

    function getZoom() {return zoom;};
    function setZoom(newZoom) {zoom = newZoom;};

    function setSrc(src) {mainimg.src = src;};

    function drawImg(img) {
	ctx.drawImage(img,0,0,
                      img_width,img_height,
                      0,0,
                      img_width*zoom,img_height*zoom);
    };

    function loading() {
	ctx.fillStyle = '#FFFFFF';
	ctx.fillRect(0,0,img_width*zoom,img_height*zoom);
	ctx.globalAlpha = 0.25;
	drawImg(mainimg);
    };

    function fillCircle(x, y, radius, fillColor) {
        ctx.fillStyle = fillColor;
        ctx.beginPath();
        ctx.moveTo(x*zoom, y*zoom);
        ctx.arc(x*zoom, y*zoom, radius*zoom, 0, Math.PI * 2, false);
        ctx.fill();
    };

    function fillLine(val, radius, fillColor) {
	if(val.length == 2) {
	    fillCircle(val[0],val[1], radius, fillColor)
	    return;
	}
	ctx.strokeStyle = fillColor;
	ctx.lineWidth = radius*zoom*2; // convert radius to width
	ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(val[0]*zoom,val[1]*zoom);
	ctx.lineTo(val[2]*zoom,val[3]*zoom);
        ctx.stroke();
    };

    function fillX(x, y, radius, fillColor) {
	fillCircle(x,y,radius,'#000000');
        ctx.strokeStyle = fillColor;
	ctx.lineWidth = 1;
	ctx.lineCap = 'square';
        ctx.beginPath();
        ctx.moveTo(x*zoom, y*zoom);
	ctx.lineTo(x*zoom+radius*zoom,y*zoom+radius*zoom);
        ctx.moveTo(x*zoom, y*zoom);
	ctx.lineTo(x*zoom-radius*zoom,y*zoom-radius*zoom);
        ctx.moveTo(x*zoom, y*zoom);
	ctx.lineTo(x*zoom+radius*zoom,y*zoom-radius*zoom);
        ctx.moveTo(x*zoom, y*zoom);
	ctx.lineTo(x*zoom-radius*zoom,y*zoom+radius*zoom);
        ctx.stroke();
    };

    function onredraw(f) {ctx.redraw = f;};

    return {
	init		: init,
	src		: setSrc,
	loading		: loading,
	getZoom		: getZoom,
	setZoom		: setZoom,
	fillCircle	: fillCircle,
	fillLine	: fillLine,
	fillX		: fillX,
	onredraw	: onredraw,
	redraw		: redraw,
    }
}());
