var init = (function ($,log,workcanvas,tools,callbacks,remote) {
    "use strict"


    function pad(number, length) {    
	var str = '' + number;
	while (str.length < length) {
	    str = '0' + str;
	}
	return str;
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
		var mw = 0; // max witdh
		$('label', this).each(function(index){
		    var w = $(this).width();
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

        remote.datasets(function(data) {
            // fetch first dataset's slices
            tools.setProp('dataset',data[0][0]);
            callbacks.update_slices(data[0][0]);
	    var parent = $('#datasets');
	    data.forEach(function (e) {
		parent.append('<button type="button" class="dataset button" id="'+e[0]+'">'+e[1]+'</button>')
                $('#'+e[0]).data('slices',e[2]);
	    });
	    $('.dataset').button({icons: {primary: "ui-icon-document"}})
                .click(callbacks.change_dataset);
            // console.log($('#datasets').first().attr('id'));
            // tools.setProp('dataset',$('#datasets').first().attr('id'));
	});

	$('#size').slider({
	    value:5, min: 1, max: 50, step: 1,
	    slide: callbacks.slider,
	    change: callbacks.slider,
	});

	$('#dilation').slider({
	    value:15, min: 1, max: 50, step: 1,
	    slide: callbacks.slider,
	    change: callbacks.slider,
	});

	$('.slider-zoom').slider({
	    value:2, min: 0.25, max: 5, step: 0.25,
	    slide: callbacks.zoom,
	    change: callbacks.zoom,
	});

	$('#dilation-value').val($('#dilation').slider('value')+'px');
	$('#size-value').val($('#size').slider('value')+'px');
	$('#zoom-value').val('2X');

	$('#accordion').accordion({fillSpace: true});
	
	$('.button').button();
	// set icons
	$('#none').button({icons: {primary: "ui-icon-arrow-4"}});
	$('#addition').button({icons: {primary: "ui-icon-plus"}});
	$('#auto').button({icons: {primary: "ui-icon-circle-plus"}});
	$('#removal').button({icons: {primary: "ui-icon-close"}});
	$('#line').button({icons: {primary: "ui-icon-minus"}});
	$('#img').button({icons: {primary: "ui-icon-image"}});
	$('#seg').button({icons: {primary: "ui-icon-video"}});
	$('#edg').button({icons: {primary: "ui-icon-script"}});
	$('#global').button({icons: {primary: "ui-icon-refresh"}});
	$('#local').button({icons: {primary: "ui-icon-arrowreturn-1-e"}});
	$('#prop').button({icons: {primary: "ui-icon-arrow-2-e-w"}});
	$('#copyr').button({icons: {primary: "ui-icon-triangle-1-e"}});
	$('#copyl').button({icons: {primary: "ui-icon-triangle-1-w"}});
	$('#reset').button({icons: {primary: "ui-icon-cancel"}});
	$('#reload').button({icons: {primary: "ui-icon-cancel"}});
	$('#save').button({icons: {primary: "ui-icon-document"}});
	$('.dataset').button({icons: {primary: "ui-icon-document"}});

	$('#interactionset').buttonsetv();
	$('#imgtypeset').buttonsetv();
	$('#outputtabs').tabs();

        $('#local').click(callbacks.local);
        $('#global').click(callbacks.global);

        $('#prop').click(callbacks.prop);

	$('#copyr').click(callbacks.copyr);
	$('#copyl').click(callbacks.copyl);

        $('.imgtype').click(callbacks.imgtype);

	$('.interaction').click(callbacks.interaction);

	$('#reset').click(callbacks.reset);
	$('#reload').click(callbacks.reload);
	// $('#save').click(callbacks.save);

	$('#mainimg').bind('contextmenu', callbacks.canvas_left);
	$('#mainimg').click(callbacks.canvas_right);
	$('#mainimg').bind('mousewheel', callbacks.canvas_scroll);

	// hotkeys

	// button aliases
	var hotkeys = {'m' : '#none',
		       'a' : '#addition',
		       't' : '#auto',
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
				 };}(hotkeys[key])));
	}

	$(document).bind('keypress', 'x', 
			 function(){
			     var curSize = tools.getProp('size');
			     if(curSize < 50) {
				 tools.setProp('size',curSize+1);
				 $('#size').slider('value',curSize+1);
				 workcanvas.redraw();
			     }
			 });

	$(document).bind('keypress', 'z', 
			 function(){
			     var curSize = tools.getProp('size');
			     if(curSize > 1) {
				 tools.setProp('size',curSize-1);
				 $('#size').slider('value',curSize-1);
				 workcanvas.redraw();
			     }
			 });

	$(document).bind('keypress', 'v', 
			 function(){
			     var curDilation = tools.getProp('dilation');
			     if(curDilation < 50) {
				 tools.setProp('dilation',curDilation+1);
				 $('#dilation').slider('value',curDilation+1);
				 workcanvas.redraw();
			     }
			 });

	$(document).bind('keypress', 'c', 
			 function(){
			     var curDilation = tools.getProp('dilation');
			     if(curDilation > 1) {
				 tools.setProp('dilation',curDilation-1);
				 $('#dilation').slider('value',curDilation-1);
				 workcanvas.redraw();
			     }
			 });

	$(window).resize(callbacks.resize);
	callbacks.resize();
    }); 

    return {}

}($,log,workcanvas,tools,callbacks,remote));
