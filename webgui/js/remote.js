var remote = (function ($) {
    function datasets(f) {
        $.getJSON("/datasets/",f);
    }

    function slices(d,f) {
        $.getJSON("/dataset/", {'dataset' : d}, f);
    }

    // todo: remaining commands

    return {
        datasets : datasets,
        slices : slices,
    }

}($));
