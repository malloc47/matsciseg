var remote = (function ($,log) {
    function datasets(f) {
        $.getJSON("/datasets/",f);
    }

    function slices(d,f) {
        $.getJSON("/dataset/", {'dataset' : d}, f);
    }

    function global(dataset,slice,dilation,f,err) {
        $.getJSON("/global/", 
		  {'dataset'	: dataset,
		   'slice'	: slice,
		   'dilation'	: dilation,
		  }, f).error(err);
    }

    function local(dataset,slice,dilation,size,
		   addition,auto,removal,line,
		   f,err) {
        $.getJSON("/local/",
		  {'dataset'	: dataset,
		   'slice'	: slice,
		   'dilation'	: dilation,
		   'size'	: size,
		   'addition'	: addition,
		   'auto'	: auto,
		   'removal'	: removal,
		   'line'	: line,
		  }, f).error(err);
    }

    function copy(dataset,slice,source,f,err) {
        $.getJSON("/copy/", 
		  {'dataset'	: dataset,
		   'slice'	: slice,
		   'source'     : source
		  }, f).error(err);
    }

    // todo: remaining commands

    return {
        datasets : datasets,
        slices : slices,
        global : global,
        local : local,
        copy : copy,
    }

}($,log));
