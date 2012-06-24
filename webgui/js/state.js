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

	$('.slider-zoom').slider({
	    value:2, min: 0.25, max: 5, step: 0.25,
	    slide: function( event, ui ) {
		var name = $(this).attr('id');
		$('#'+name+'-value').val( ui.value + "X" );
		workcanvas.setZoom(ui.value);
		workcanvas.redraw();
	    }
	});

	// $('.slider').mouseup(function() {
	//     var name = $(this).attr('id');
	//     alert(tools.getProp(name));
	// });

	// $('.slider-text').val('5px');
	$('#dilation-value').val($('#dilation').slider('value')+'px');
	$('#size-value').val($('#size').slider('value')+'px');
	$('#zoom-value').val('2X');
	
	$('.button').button();
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
	    $('.imgtype').css({"background":"#CCCCCC"});
	    $(this).css({"background":"#AAAACC"});
	    workcanvas.src('/' + tools.getImgPath() + '/'+pad(state['image'],4)+'/?'+new Date().getTime());
	});

	$('.interaction').click(function() {
	    tools.setTool($(this).attr('id'));
	    $('#mainimg').css(tools.cursor());
	    log.append('mode changed to '+tools.getTool());
	    $('.interaction').css({"background":"#CCCCCC"});
	    if(tools.getTool() == 'none') {
		$('#interactionset input').removeAttr('checked');
		$('#interactionset input').button('refresh');
		// $('.interaction').prop('checked', false);
	    }
		// $(this).css({"background":"#CCAAAA"});
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
	    if(tools.remove([x,y])) {
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
/*	    if(tools.getTool() == 'addition') {
		// workcanvas.fillCircle(x, y, size, '#ffffff');
	  	workcanvas.fillCircle(x, y, size+dilation, 'rgba(255,255,255,0.5)');
	  	workcanvas.fillCircle(x, y, size, 'rgba(255,255,255,1.0)');
	    }
	    else if(tools.getTool() == 'removal') {
		workcanvas.fillX(x, y, size, '#ffffff');
	    }
	    else if(tools.getTool() == 'line') {
		prev = tools.get('line');
		prev = prev[prev.length-1].slice(0) // copy array
		// workcanvas.fillLine(prev, size, '#ffffff');
		workcanvas.fillLine(prev, size+dilation, 'rgba(255,255,255,0.5)');
		workcanvas.fillLine(prev, size, 'rgba(255,255,255,1.0)');
	    }
*/
	    log.append(tools.getTool() + ' at '+x+','+y);
	});
	getstate();
    }); 

    return {}

// (function poll(){
//     $.ajax({ url: "server", success: function(data){
//         //Update your dashboard gauge
//         salesGauge.setValue(data.value);

//     }, dataType: "json", complete: poll, timeout: 30000 });
// })();

}($,log,workcanvas,tools));
