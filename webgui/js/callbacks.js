var callbacks = (function ($,log,workcanvas,tools,sliceSelector,remote) {

    function reload_slices() {
        sliceSelector.clear();
        sliceSelector.add(tools.getProp('images'),
                          tools.getProp('image'),
                          tools.getProp('dataset'),
                          128, // fixed thumbnail width
			  change_slice);
    }

    function update_slices(d) {
        remote.slices(d,
                      function(slices) {
                          tools.setProp('images',slices);
                          tools.setProp('image',slices[0]);
                          // only now to we setup the image
	                  workcanvas.init($("#mainimg"),tools.getDataset());
	                  workcanvas.onredraw(callbacks.canvas_redraw);
                          reload_slices();
                      });
    }

    function change_dataset() {
	var dataset = $(this).attr('id');
	log.append("changing to "+dataset);
	$("#accordion").accordion("activate", 0);
	workcanvas.loading();
        tools.setProp('dataset',dataset);
        workcanvas.src(tools.getDataset());
        workcanvas.redraw();
        reload_slices();
	// $.getJSON("/state/",function(data) {
	//     state = data;
    	//     parsestate();
	// });
    }

    function change_slice(e) {
	workcanvas.loading();
	var c=$(this).attr('id');
	tools.setProp('image',parseInt(c));
	log.append("opening slice "+c);
        workcanvas.src(tools.getDataset());
        workcanvas.redraw();
        sliceSelector.update(c);
    }

    function slider( event, ui ) {
	var name = $(this).attr('id');
	$('#'+name+'-value').val( ui.value + "px" );
	tools.setProp(name,ui.value);
	workcanvas.redraw();
    }

    function zoom( event, ui ) {
	var name = $(this).attr('id');
	$('#'+name+'-value').val( ui.value + "X" );
	workcanvas.setZoom(ui.value);
	workcanvas.redraw();
    }

    function global() {
	log.append("starting global");
	workcanvas.loading();
	remote.global(tools.getProp('dataset'),
		      tools.getProp('image'),
		      tools.getProp('dilation'),
		      function() {log.append("global successful");
				  workcanvas.src(tools.getDataset());
				  workcanvas.redraw();},
		      function() {log.append("error: no response");
				  workcanvas.redraw();});
    }

    function local() {
	if(!(tools.getStr('addition').length > 0 ||
	     tools.getStr('auto').length > 0 ||
	     tools.getStr('removal').length > 0 ||
	     tools.getStr('line').length > 0)) {
	    log.append("error: no annotations");
	    return;
	}
	log.append("starting local");
	workcanvas.loading();
	remote.local(tools.getProp('dataset'),
		      tools.getProp('image'),
		      tools.getProp('dilation'),
		      tools.getProp('size'),
		      tools.getStr('addition'),
		      tools.getStr('auto'),
		      tools.getStr('removal'),
		      tools.getStr('line'),
		      function() {log.append("local successful");
				  tools.clear();
				  workcanvas.src(tools.getDataset());
				  workcanvas.redraw();},
		      function() {log.append("error: no response");
				  workcanvas.redraw();});
    }

    function copyr() {copy('copyr',tools.getProp('image')-1);}
    function copyl() {copy('copyl',tools.getProp('image')+1);}

    function copy(dir,source) {
	log.append("starting "+dir);
	workcanvas.loading();
	remote.copy(tools.getProp('dataset'),
		      tools.getProp('image'),
		      source,
		      function() {log.append(dir+" successful");
				  workcanvas.src(tools.getDataset());
				  workcanvas.redraw();},
		      function() {log.append("error: no source slice");
				  workcanvas.redraw();});
    }

    function imgtype() {
        var m = $(this).attr('id');
	log.append(m+" view");
	workcanvas.loading();
	tools.setProp('imgMode',m);
	$(this).css({"background":"#AAAACC"});
	workcanvas.src(tools.getDataset());
        workcanvas.redraw();
    }

    function interaction() {
	tools.changeTool($(this).attr('id'),
			 $('#mainimg'),
			 $('#interactionset input'));
	log.append('mode changed to '+tools.getTool());
    }

    function reset() {
	tools.clear();
	workcanvas.redraw();
	$('.output').children().remove();
	log.append("reset successful")
    }

    function canvas_redraw() {
	addition = tools.get('addition');
	auto = tools.get('auto');
	removal = tools.get('removal');
	line = tools.get('line');
	dilation = tools.getProp('dilation');
	size = tools.getProp('size');
	for (var i=0; i<addition.length; i++) {
	    workcanvas.fillCircle(addition[i][0], addition[i][1], size+dilation, 'rgba(255,255,255,0.5)');
	    workcanvas.fillCircle(addition[i][0], addition[i][1], size, 'rgba(255,255,255,1.0)');
	}
	for (var i=0; i<auto.length; i++) {
	    workcanvas.fillCircle(auto[i][0], auto[i][1], 3, 'rgba(255,255,255,1.0)');
	}
	for (var i=0; i<removal.length; i++) {
	    workcanvas.fillX(removal[i][0], removal[i][1], 3, 'rgba(255,255,255,1.0)');
	}
	for (var i=0; i<line.length; i++) {
	    workcanvas.fillLine(line[i], size+dilation, 'rgba(255,255,255,0.5)');
	    workcanvas.fillLine(line[i], size, 'rgba(255,255,255,1.0)');
	}
    }

    function canvas_left(e) {
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
    }

    function canvas_right(e) {
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
    }

    function canvas_scroll(event, delta) {
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
    }

    function resize() {
        if($(window).height() < 710) return;
	var new_height = ($(window).height()) - $('.sliders').height() - $('.bottombar').height() - 50;
	$('.workingarea').css(
	    {'height': new_height+'px'});
    }

    return {
        reload_slices  : reload_slices,
        update_slices  : update_slices,
	change_dataset : change_dataset,
	change_slice   : change_slice,
	slider         : slider,
        zoom           : zoom,
        global         : global,
        local          : local,
        copyr          : copyr,
        copyl          : copyl,
        imgtype        : imgtype,
        interaction    : interaction,
        reset          : reset,
        canvas_redraw  : canvas_redraw,
        canvas_left    : canvas_left,
        canvas_right   : canvas_right,
        canvas_scroll  : canvas_scroll,
        resize         : resize,
    }
}($,log,workcanvas,tools,sliceSelector,remote));
