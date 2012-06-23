var sliceSelector = (function () {

    var ul;

    function pad(number, length) {    
	var str = '' + number;
	while (str.length < length) {
	    str = '0' + str;
	}
	return str;
    }

    function init(ul_in){
	ul = ul_in;
    };

    function add(imglst,curimg,w,f) {
	for (var i = 0; i < imglst.length; i++) {
	    ul.append('<li'
		      + (imglst[i]==curimg ? ' class="selected" ' : '')
		      + '><img class="thumb'
		      + imglst[i]
		      + '" src="/thumb/'
		      + pad(imglst[i],4)
		      + '/'
		      +'?'+new Date().getTime()
		      +'" '
		      + 'id="' + imglst[i] + '" '
		      + '/><p>'
		      + imglst[i]
		      + '</p></li>');
	}
	ul.css({'width':((w+10)*imglst.length+10).toString()});
	ul.find('img').click(f);
    };

    function clear() {
	ul.children().remove();
    };

    return {
	init	: init,
	add	: add,
	clear	: clear
    }

}());