var tools = (function () {

    var toolMode;
    var imgMode;

    var imgPaths = {'img' : 'empty',
		     'seg' : 'output',
		     'edg' : 'edge'};

    var cursorMode = {'none':{'cursor':'default'},'addition':{'cursor':'crosshair'},'removal':{'cursor':'move'}};
    var currentCursor;

    function init() {
	toolMode = 'none';
	imgMode = 'seg';
	currentCursor = 'default';
    }

    function getTool() {return toolMode;}

    function setTool(newToolMode) {
	toolMode = toolMode==newToolMode ? 'none' : newToolMode;
	currentCursor = cursorMode[toolMode];
    }

    function getImgMode() {return imgMode;}

    function setImgMode(newImgMode) {imgMode = newImgMode;}

    function getImgPath() {return imgPaths[imgMode];}

    function cursor() {return currentCursor;}

    return {
	init		: init,
	getTool		: getTool,
	setTool		: setTool,
	getImgMode	: getImgMode,
	setImgMode	: setImgMode,
	getImgPath	: getImgPath,
	cursor		: cursor,
    }
}());