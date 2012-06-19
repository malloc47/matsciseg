var tools = (function () {

    var toolMode;
    var imgMode;

    var data = {'addition' : [],
		'removal'  : []}

    var imgPaths = {'img' : 'empty',
		     'seg' : 'output',
		     'edg' : 'edge'};

    var cursorMode = {'none':{'cursor':'default'},
		      'addition':{'cursor':'crosshair'},
		      'removal':{'cursor':'move'}};

    var currentCursor;

    function init() {
	toolMode = 'none';
	imgMode = 'seg';
	currentCursor = 'default';
	addition = [];
	removal = [];
    };

    function getTool() {return toolMode;};

    function setTool(newToolMode) {
	toolMode = toolMode==newToolMode ? 'none' : newToolMode;
	currentCursor = cursorMode[toolMode];
    };

    function getImgMode() {return imgMode;};

    function setImgMode(newImgMode) {imgMode = newImgMode;};

    function getImgPath() {return imgPaths[imgMode];};

    function cursor() {return currentCursor;};

    function push(val,type) {
	if(!type) type = toolMode;
	data[type].push(val);
    };

    function clear() {
	for(var d in data) {
	    data[d] = [];
	}
    };

    function get(type) {
	if(!type) type = toolMode;
	return data[type];
    };

    return {
	init		: init,
	getTool	: getTool,
	setTool	: setTool,
	getImgMode	: getImgMode,
	setImgMode	: setImgMode,
	getImgPath	: getImgPath,
	cursor		: cursor,
	push		: push,
	clear		: clear,
	get		: get
    }
}());