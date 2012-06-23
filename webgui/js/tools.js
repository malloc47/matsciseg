var tools = (function () {

    var toolMode;
    // var imgMode;

    var data = {'addition' : [],
		'removal'  : [],
		'line'     : []};

    var imgPaths = {'img'	: 'empty',
		     'seg'	: 'output',
		     'edg'	: 'edge'};

    var cursorMode = {'none'		:{'cursor':'default'},
		      'addition'	:{'cursor':'crosshair'},
		      'removal'		:{'cursor':'move'},
		      'line'		:{'cursor':'crosshair'}};

    var currentCursor;

    // var dilation;
    // var size;

    var props = {'imgMode'	: 'seg',
		 'dilation'	: 5,
		 'size'		: 5};
    
    function init() {
	toolMode = 'none';
	currentCursor = 'default';
	for (var key in data) {data[key] = [];}
	// imgMode = 'seg';
	// size = 5;
	// dilation = 5;
    };

    function getTool() {return toolMode;};

    function setTool(newToolMode) {
	toolMode = toolMode==newToolMode ? 'none' : newToolMode;
	currentCursor = cursorMode[toolMode];
    };

    function getProp(prop) {return props[prop];};

    function setProp(prop,newProp) {props[prop] = newProp;};

    // function getImgMode() {return imgMode;};
    // function setImgMode(newImgMode) {imgMode = newImgMode;};

    // function getSize() {return size;};
    // function setSize(newSize) {size = newSize;};

    // function getDilation() {return dilation;};
    // function setDilation(newDilation) {dilation = newDilation;};

    function getImgPath() {return imgPaths[props['imgMode']];};

    function cursor() {return currentCursor;};

    function push(val,type) {
	if(!type) type = toolMode;
	// handle the line case, where we want 4 points per line, but
	// add them two at a time
	if(type == 'line' && data['line'].length > 0) {
	    if(data['line'][data['line'].length-1].length == 2) {
		for (var i = 0; i < val.length; i++) {
		    data['line'][data['line'].length-1].push(val[i]);
		}
		return;
	    }
	}
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

    function tuplesToStr(l) {
	output = '';
	for (var i=0; i<l.length; i++) {
	    output += l[i][0].toString()+','+l[i][1].toString()+';';
	}
	// don't grab last ";"
	if(output[output.length-1] == ';') {
	    return output.substr(0,output.length-1);
	}
	else {
	    return output;
	}
    };

    function getStr(type) {
	if(!type) type = toolMode;
	return tuplesToStr(data[type]);
    };

    return {
	init		: init,
	getTool		: getTool,
	setTool		: setTool,
	// getImgMode	: getImgMode,
	// setImgMode	: setImgMode,
	// getSize		: getSize,
	// setSize 	: setSize,
	// getDilation	: getDilation,
	// setDilation 	: setDilation,
	getProp		: getProp,
	setProp		: setProp,
	getImgPath	: getImgPath,
	cursor		: cursor,
	push		: push,
	clear		: clear,
	get		: get,
	getStr		: getStr,
    }
}());
