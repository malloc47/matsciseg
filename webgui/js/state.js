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
	// update bottom bar
	$('.bottombar ul').children().remove();
	$('.bottombar ul').css({'width':(124*state['images'].length+20).toString()});
	for (var i = 0; i < state['images'].length; i++) {
	    $('.bottombar ul').append('<li'
				      + (state['images'][i]==state['image'] ? ' class="selected" ' : '')				      
				      + '><img class="thumb'
				      + state['images'][i]
				      // + (i==parseInt(state['image']) ? ' selected' : '')
				      + '" src="/thumb/'
				      + pad(state['images'][i],4)
				      + '/'
				      +'?'+new Date().getTime()
				      +'" '
				      + 'id="' + state['images'][i] + '" '
				      + '/><p>'
				      + state['images'][i]
				      + '</p></li>');
	}
	$('.bottombar ul li img').click(function(e) {
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

	log.init($('.output'),$('.rawoutput'));
	tools.init();

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
	    tools.setImgMode(m)
	    $('.imgtype').css({"background":"#CCCCCC"});
	    $(this).css({"background":"#AAAACC"});
	    workcanvas.src('/' + tools.getImgPath() + '/'+pad(state['image'],4)+'/?'+new Date().getTime());
	});

	$('.interaction').click(function() {
	    tools.setTool($(this).attr('id'));
	    $('#mainimg').css(tools.cursor());
	    log.append('mode changed to '+tools.getTool());
	    $('.interaction').css({"background":"#CCCCCC"});
	    if(tools.getTool() != 'none')
		$(this).css({"background":"#CCAAAA"});
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

	$('#mainimg').click(function(e) {
	    if(tools.getTool() == 'none') return;
	    var x = e.pageX - this.offsetLeft + $('.workingarea').scrollLeft();
            var y = e.pageY - this.offsetTop + $('.workingarea').scrollTop();
	    x_new = Math.floor(x/2);
	    y_new = Math.floor(y/2);
	    tools.push([x_new,y_new]);
	    if(tools.getTool() == 'addition')
		workcanvas.fillCircle(x, y, 5, '#ffffff');
	    else if(tools.getTool() == 'removal')
		workcanvas.fillX(x, y, 5, '#ffffff');
	    log.append(tools.getTool() + ' at '+x_new+','+y_new);
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