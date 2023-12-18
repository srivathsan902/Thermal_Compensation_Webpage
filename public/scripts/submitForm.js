document.getElementById("modelParameters").addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent the default form submission
    sendLogToServer();
});