var client_ms_time = null;
$.cookie.raw = true;

var last_api_url = "";

function full_json(url, fnc, make_loading=false) {
	if($.cookie && client_ms_time !== null && last_api_url.length > 0) {
		
		$.cookie("client_ms_time", client_ms_time, {path: "/"});
		$.cookie("client_last_api_url", last_api_url, {path: "/"});
		// console.log("Setting the client times:", last_api_url, client_ms_time)
		// console.log($.cookie("client_ms_time"))
	}

	console.log(url);
	if(make_loading) {
		$('#loading').show();
	}
	var client_start_time = new Date();
	$.getJSON({url: url, success: function(data) {
		var client_end_time = new Date();
		client_ms_time = client_end_time.getTime() - client_start_time.getTime();
		last_api_url = url;

		console.log(data);
		$('#loading').hide();
		fnc(data);
	}, xhrFields: {withCredentials: true}}).fail(function(error) {
		console.log(error);
		alert("Error. Please try again.")
	});
}