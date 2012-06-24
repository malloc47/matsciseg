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
		})
		    $('label', this).each(function(index){
			$(this).width(mw);
		    })
			};
	})( $ );

	log.init($('.output'),$('.overflow-wrap'));
	tools.init();
	sliceSelector.init($('.bottombar ul'));
	$('.workingarea').dragscrollable(); 

	$('.slider').slider({
	    value:5,
	    min: 1,
	    max: 100,
	    step: 1,
	    slide: function( event, ui ) {
		var name = $(this).attr('id');
		$('#'+name+'-value').val( ui.value + "px" );
		tools.setProp(name,ui.value);
	    }
	});

	$('.slider-zoom').slider({
	    value:2,
	    min: 0.25,
	    max: 5,
	    step: 0.25,
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

	$('.slider-text').val('5px');
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
	    // tools.setImgMode(m)
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
	    for (var i=0; i<addition.length; i++) {
	  	workcanvas.fillCircle(addition[i][0]*2, addition[i][1]*2, 5, '#ffffff');
	    }
	    for (var i=0; i<removal.length; i++) {
	  	workcanvas.fillX(removal[i][0]*2, removal[i][1]*2, 5, '#ffffff');
	    }
	});

	$('#mainimg').bind('contextmenu', function(e){
	    e.preventDefault();
	    var x = e.pageX - this.offsetLeft + $('.workingarea').scrollLeft();
            var y = e.pageY - this.offsetTop + $('.workingarea').scrollTop();
	    x = Math.floor(x/workcanvas.getZoom());
	    y = Math.floor(y/workcanvas.getZoom());
	    // stuff here
	    return false;
	});

	$('#mainimg').click(function(e) {
	    if(tools.getTool() == 'none') return;
	    var x = e.pageX - this.offsetLeft + $('.workingarea').scrollLeft();
            var y = e.pageY - this.offsetTop + $('.workingarea').scrollTop();
	    x = Math.floor(x/workcanvas.getZoom());
	    y = Math.floor(y/workcanvas.getZoom());
	    tools.push([x,y]);
	    if(tools.getTool() == 'addition')
		workcanvas.fillCircle(x, y, 5, '#ffffff');
	    else if(tools.getTool() == 'removal')
		workcanvas.fillX(x, y, 5, '#ffffff');
	    else if(tools.getTool() == 'line') {
		prev = tools.get('line');
		prev = prev[prev.length-1].slice(0) // copy array
		workcanvas.fillLine(prev, 5, '#ffffff');
	    }
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
