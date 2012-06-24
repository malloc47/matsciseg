var workcanvas = (function () {

    var ctx;
    var canvas;
    var mainimg;
    var width, height;
    var zoom = 2;

    function imgLoad() {
	if(canvas.attr('width') != width*zoom)
	    canvas.attr('width',width*zoom);
	if(canvas.attr('height') != height*zoom)
	    canvas.attr('height',height*zoom);
	ctx.globalAlpha = 1.0;
	drawImg(mainimg);
	ctx.redraw();
    };

    function init(w,h,c){
	width = w;
	height = h;
	canvas = c;
	ctx = canvas[0].getContext('2d');
	if(canvas.attr('width') != width*zoom)
	    canvas.attr('width',width*zoom);
	if(canvas.attr('height') != height*zoom)
	    canvas.attr('height',height*zoom);
	mainimg = new Image();
	// mainimg.src = "/img/image0090.png"; // static img default
	mainimg.onload = imgLoad;

    };

    function getZoom() {return zoom;};
    function setZoom(newZoom) {zoom = newZoom;};

    function setSrc(src) {mainimg.src = src;};

    function drawImg(img) {
	ctx.drawImage(img,0,0,width,height,0,0,width*zoom,height*zoom);
    };

    function loading() {
	ctx.fillStyle = '#FFFFFF';
	ctx.fillRect(0,0,width*zoom,height*zoom);
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
	redraw		: imgLoad
    }
}());
