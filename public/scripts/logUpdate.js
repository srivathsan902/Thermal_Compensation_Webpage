function sendLogToServer() {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:3500/log", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    var inputType = "formSubmission";
    var inputValue = 'Success';         
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                console.log(xhr.responseText);
            } else {
                console.error('Error:', xhr.status, xhr.statusText);
            }
        }
    };

    xhr.send(JSON.stringify({
        inputType: inputType,
        inputValue: inputValue,
        selectedOption: document.getElementById("Models").value,
        slider1Value: document.getElementById("slider1").value,
        slider2Value: document.getElementById("slider2").value
    }));
}