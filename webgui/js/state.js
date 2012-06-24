var state = (function ($,log,workcanvas,tools) {

    var state = {};

    function pad(number, length) {    
	var str = '' + number;
	while (str.length < length) {
	    str = '0' + str;
	}
	return str;
    }

    function getstate() {
	$.getJSON("/state/",function(data) {
	    state = data;
    	    parsestate();
	});
    }

    function sendstate() {
	$.getJSON("/state/",state);
    }

    function syncstate() {
    	$.getJSON("/state/",state,function(data) {
    	    state = data;
    	    parsestate();
    	});
    }

    function parsestate() {
	tools.clear()
	sliceSelector.clear();
	sliceSelector.add(state['images'],state['image'],state['width']*0.15,
			  function(e) {
			      workcanvas.loading();
			      var c=$(this).attr('id');
			      state['image'] = parseInt(c);
			      log.append("opening slice "+c);
			      syncstate();
			  });

	workcanvas.init(state['width'],
			state['height'],
			$("#mainimg"));

	workcanvas.src('/' + tools.getImgPath() + '/'+pad(state['image'],4)+'/?'+new Date().getTime())

	if('response' in state) {
	    log.append(state['response']);
	}
    }

    $(document).ready(function() {

	// http://stackoverflow.com/questions/4098054/vertically-displayed-jquery-buttonset
	(function( $ ){
	    //plugin buttonset vertical
	    $.fn.buttonsetv = function() {
		$(':radio, :checkbox', this).wrap('<div style="margin: 1px"/>');
		$(this).buttonset();
		$('label:first', this).removeClass('ui-corner-left').addClass('ui-corner-top');
		$('label:last', this).removeClass('ui-corner-right').addClass('ui-corner-bottom');
		mw = 0; // max witdh
		$('label', this).each(function(index){
		    w = $(this).width();
		    if (w > mw) mw = w; 
		});
		$('label', this).each(function(index){
		    $(this).width(mw);
		});
	    };
	})( $ );

	log.init($('.output'),$('.overflow-wrap'));
	tools.init();
	sliceSelector.init($('.bottombar ul'));
	$('.workingarea').dragscrollable(); 

	sliderCallback = (function( event, ui ) {
	    var name = $(this).attr('id');
	    $('#'+name+'-value').val( ui.value + "px" );
	    tools.setProp(name,ui.value);
	    workcanvas.redraw();
	});

	$('#size').slider({
	    value:5, min: 1, max: 50, step: 1,
	    slide: sliderCallback
	});

	$('#dilation').slider({
	    value:15, min: 1, max: 50, step: 1,
	    slide: sliderCallback
	});

	zoomCallback = (function( event, ui ) {
	    var name = $(this).attr('id');
	    $('#'+name+'-value').val( ui.value + "X" );
	    workcanvas.setZoom(ui.value);
	    workcanvas.redraw();
	});

	$('.slider-zoom').slider({
	    value:2, min: 0.25, max: 5, step: 0.25,
	    slide: zoomCallback,
	    change: zoomCallback
	});

	// $('.slider').mouseup(function() {
	//     var name = $(this).attr('id');
	//     alert(tools.getProp(name));
	// });

	$('#dilation-value').val($('#dilation').slider('value')+'px');
	$('#size-value').val($('#size').slider('value')+'px');
	$('#zoom-value').val('2X');
	
	$('.button').button();
	// set icons
	$('#addition').button({icons: {primary: "ui-icon-plus"}});
	$('#removal').button({icons: {primary: "ui-icon-close"}});
	$('#line').button({icons: {primary: "ui-icon-minus"}});
	$('#img').button({icons: {primary: "ui-icon-image"}});
	$('#seg').button({icons: {primary: "ui-icon-video"}});
	$('#edg').button({icons: {primary: "ui-icon-script"}});
	$('#global').button({icons: {primary: "ui-icon-refresh"}});
	$('#local').button({icons: {primary: "ui-icon-arrowreturn-1-e"}});
	$('#copyr').button({icons: {primary: "ui-icon-triangle-1-e"}});
	$('#copyl').button({icons: {primary: "ui-icon-triangle-1-w"}});
	$('#reset').button({icons: {primary: "ui-icon-cancel"}});
	$('.dataset').button({icons: {primary: "ui-icon-document"}});

	$('#interactionset').buttonsetv();
	$('#imgtypeset').buttonsetv();
	$('#outputtabs').tabs();

	$('.serversend').click(function() {
	    var method = $(this).attr('id');
	    log.append("starting "+method);
	    workcanvas.loading();
	    state['command'] = method;
	    state['addition'] = tools.getStr('addition');
	    state['removal'] = tools.getStr('removal');
	    state['size'] = tools.getProp('size');
	    state['dilation'] = tools.getProp('dilation');
	    syncstate();
	});

	$('.dataset').click(function() {
	    var dataset = $(this).attr('id');
	    log.append("changing to "+dataset);
	    workcanvas.loading();
	    state['command'] = 'dataset';
	    state['dataset'] = dataset;
	    syncstate();
	});

	$('.imgtype').click(function() {
	    var m = $(this).attr('id');
	    log.append(m+" view");
	    workcanvas.loading();
	    tools.setProp('imgMode',m);
	    $(this).css({"background":"#AAAACC"});
	    workcanvas.src('/' + tools.getImgPath() + '/'+pad(state['image'],4)+'/?'+new Date().getTime());
	});

	$('.interaction').click(function() {
	    tools.changeTool($(this).attr('id'),
			     $('#mainimg'),
			     $('#interactionset input'));
	    log.append('mode changed to '+tools.getTool());
	});

	$('#reset').click(function() {
	    tools.clear();
	    workcanvas.redraw();
	    $('.output').children().remove();
	    log.append("reset successful")
	});

	workcanvas.init(state['width'],
			state['height'],
			$("#mainimg"));

	workcanvas.onredraw(function() {
	    addition = tools.get('addition');
	    removal = tools.get('removal');
	    line = tools.get('line');
	    dilation = tools.getProp('dilation');
	    size = tools.getProp('size');
	    for (var i=0; i<addition.length; i++) {
	  	workcanvas.fillCircle(addition[i][0], addition[i][1], size+dilation, 'rgba(255,255,255,0.5)');
	  	workcanvas.fillCircle(addition[i][0], addition[i][1], size, 'rgba(255,255,255,1.0)');
	    }
	    for (var i=0; i<removal.length; i++) {
	  	workcanvas.fillX(removal[i][0], removal[i][1], 3, 'rgba(255,255,255,1.0)');
	    }
	    for (var i=0; i<line.length; i++) {
		workcanvas.fillLine(line[i], size+dilation, 'rgba(255,255,255,0.5)');
		workcanvas.fillLine(line[i], size, 'rgba(255,255,255,1.0)');
	    }
	});

	$('#mainimg').bind('contextmenu', function(e){
	    e.preventDefault();
	    var x = e.pageX - this.offsetLeft + $('.workingarea').scrollLeft();
            var y = e.pageY - this.offsetTop + $('.workingarea').scrollTop();
	    x = Math.floor(x/workcanvas.getZoom());
	    y = Math.floor(y/workcanvas.getZoom());
	    if(tools.getTool() != 'none') {
		tools.changeTool('none',
				 $('#mainimg'),
				 $('#interactionset input'));
		log.append('mode changed to none');
	    }
	    else if(tools.remove([x,y])) {
		log.append("removing annotation");
	    }
	    else {
		log.append("error: no annotation");
	    }
	    workcanvas.redraw();
	    return false;
	});

	$('#mainimg').click(function(e) {
	    if(tools.getTool() == 'none') return;
	    var x = e.pageX - this.offsetLeft + $('.workingarea').scrollLeft();
            var y = e.pageY - this.offsetTop + $('.workingarea').scrollTop();
	    x = Math.floor(x/workcanvas.getZoom());
	    y = Math.floor(y/workcanvas.getZoom());
	    size = tools.getProp('size');
	    dilation = tools.getProp('dilation');
	    tools.push([x,y]);
	    workcanvas.redraw();	    
	    log.append(tools.getTool() + ' at '+x+','+y);
	});

	$('#mainimg').bind('mousewheel', function(event, delta) {
            var dir = delta > 0 ? 'Up' : 'Down',
                vel = Math.abs(delta);
            $(this).text(dir + ' at a velocity of ' + vel);
	    curZoom = workcanvas.getZoom();
	    newZoom = delta > 0 ? curZoom + 0.25 : curZoom - 0.25;
	    if (newZoom >= 0.25 && newZoom <= 5) {
		workcanvas.setZoom(newZoom);
		$('#zoom').slider('value',newZoom);
	    }
            return false;
        });

	// hotkeys

	// button aliases
	hotkeys = {'a' : '#addition',
		   'n' : '#line',
		   'd' : '#removal',
		   'r' : '#reset',
		   'i' : '#img',
		   's' : '#seg',
		   'e' : '#edg',
		   'l' : '#local',
		   'g' : '#global',
		   '.' : '#copyl',
		   ',' : '#copyr',
		  }

	for (var key in hotkeys) {
	    $(document).bind('keypress', key, 
			     (function (val){
				 return function(){
				     $(val).click();
				     if(tools.getTool() == 'none'){
					 $('#interactionset input').removeAttr('checked').button('refresh');
				     }
				 };}(hotkeys[key])));
	}

	



	// $(document).bind('keydown', 'l', function(){
	//     // tools.changeTool('line',
	//     // 		     $('#mainimg'),
	//     // 		     $('#interactionset input'),
	//     // 		     $('#line'));
	//     $('#line').click();
	// });

	// $(document).bind('keydown', 'd', function(){
	//     // tools.changeTool('removal',
	//     // 		     $('#mainimg'),
	//     // 		     $('#interactionset input'),
	//     // 		     $('#removal'));
	//     $('#removal').click();
	// });

	getstate();
    }); 

    return {}

}($,log,workcanvas,tools));
