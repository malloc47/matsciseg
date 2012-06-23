var log = (function () {

    var output;
    var output_box;

    function init(o,ob){
	output = o;
	output_box = ob;
    };

    function append(str){
	if(str.indexOf('error:') === 0)
	    output.append('<li class="errormsg">'+
				// '['+new Date().toTimeString()+'] '+
				str+'</li>');
	else
	    output.append('<li>'+
				// '['+new Date().toTimeString()+'] '+
				str+'</li>');
	output_box.prop({ scrollTop: output_box.prop("scrollHeight") }, 30);
    };

    function clear(){output.children().remove();};

    return {
	init	: init,
	append	: append,	
	clear	: clear
    }

}());