var tools = (function () {

    var toolMode;
    // var imgMode;

    var data = {'addition' : [],
		'auto'     : [],
		'removal'  : [],
		'line'     : []};

    var imgPaths = {'img'	: 'empty',
		     'seg'	: 'output',
		     'edg'	: 'edge'};

    var cursorMode = {'none'		:{'cursor':'default'},
		      'addition'	:{'cursor':'crosshair'},
		      'auto'	        :{'cursor':'crosshair'},
		      'removal'		:{'cursor':'move'},
		      'line'		:{'cursor':'crosshair'}};

    var currentCursor;

    var props = {'imgMode'	: 'img',
		 'dilation'	: 15,
		 'size'		: 5};
    
    function init() {
	toolMode = 'none';
	currentCursor = 'default';
	for (var key in data) {data[key] = [];}
    };

    function getTool() {return toolMode;};

    function setTool(newToolMode) {
	// toolMode = toolMode==newToolMode ? 'none' : newToolMode;
	toolMode = newToolMode;
	currentCursor = cursorMode[toolMode];
    };

    function changeTool(newToolMode,cursorel,buttonset) {
	setTool(newToolMode);
	cursorel.css(cursor());
    }

    function getProp(prop) {return props[prop];};

    function setProp(prop,newProp) {props[prop] = newProp;};

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

    function dist(a,b) {
	return Math.sqrt(Math.pow(a[0]-b[0],2)+Math.pow(a[1]-b[1],2));
    }

    function removeClosest(p) {
	type = '';
	idx = -1;
	prevDist = 0;
	// brute-force nearest-neighbor
	for (var key in data) {
	    for (var i=0; i<data[key].length; i++) {
		// this will break if we add dilation/size to tuples
		for (var j=0; j<data[key][i].length; j+=2) {
		    x = data[key][i][j];
		    y = data[key][i][j+1];
		    if(idx < 0 || dist(p,[x,y])<prevDist) {
			type = key;
			idx = i;
			prevDist = dist(p,[x,y]);
		    }
		}
	    }
	}
	if(type in data && idx >= 0) {
	    data[type].splice(idx,1)
	    return true;
	}
	return false;
    }

    function tuplesToStr(l) {
	output = '';
	for (var i=0; i<l.length; i++) {
	    for (var j=0; j<l[i].length; j++) {
		output += l[i][j].toString()+',';
	    }
	    // switch comma to semicolon
	    output = output.slice(0,-1)+";";
	    // output += l[i][0].toString()+','+l[i][1].toString()+';';
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
	changeTool	: changeTool,
	getProp		: getProp,
	setProp		: setProp,
	getImgPath	: getImgPath,
	cursor		: cursor,
	push		: push,
	clear		: clear,
	get		: get,
	getStr		: getStr,
	remove		: removeClosest,
    }
}());
