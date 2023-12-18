function handleSliderChange(sliderId) {
    var sliderValue = document.getElementById(sliderId).value;
    document.getElementById(sliderId + "Value").innerText = sliderValue;
    // sendLogToServer();
}