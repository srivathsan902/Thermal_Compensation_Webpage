const express = require('express');
const fileUpload = require("express-fileupload");
const bodyParser = require('body-parser');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const filesPayloadExists = require('./middleware/filesPayloadExists');
const fileExtLimiter = require('./middleware/fileExtLimiter');
const fileSizeLimiter = require('./middleware/fileSizeLimiter');

const app = express();
const PORT = process.env.PORT || 3500;

app.use(bodyParser.json());

// Set headers to allow cross-origin requests
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*'); // You can replace '*' with the specific origin you want to allow
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
    res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');
    res.setHeader('Access-Control-Allow-Credentials', true);
    next();
});

app.use(express.static('public'));

app.post('/log', (req, res) => {
    console.log('Received log request');
    const { inputType, inputValue, selectedOption, slider1Value, slider2Value } = req.body;
    console.log(`Client Log - Type: ${inputType}, Value: ${inputValue}, Selected Option: ${selectedOption}, Slider 1 Value: ${slider1Value}, Slider 2 Value: ${slider2Value}`);

    // Call Python script with the selected value
    callPythonScript(inputType, inputValue, selectedOption, slider1Value, slider2Value, (modifiedValue) => {
        console.log('Output from Python Script:', modifiedValue);
        res.send('Log received on the server.');
    });
    
});

// Handle file upload
app.post('/upload',
    fileUpload({ createParentPath: true }),
    filesPayloadExists,
    fileExtLimiter(['.xlsx']),
    fileSizeLimiter,
    (req, res) => {
        console.log('Received Upload Request');
        const files = req.files
        console.log(files)

        Object.keys(files).forEach(key => {
            const filepath = path.join(__dirname, 'code/Datasets', files[key].name)
            files[key].mv(filepath, (err) => {
                if (err) return res.status(500).json({ status: "error", message: err })
            })
        })

        return res.json({ status: 'success', message: Object.keys(files).toString() })
    }
)



function callPythonScript(inputType, inputValue, selectedOption, slider1Value, slider2Value, callback) {
    const pythonScriptPath = path.join(__dirname, 'code', 'Run.py');
    const pythonProcess = spawn('python', [pythonScriptPath, inputType, inputValue, selectedOption, slider1Value, slider2Value]);

    let modifiedValue = '';

    pythonProcess.stdout.on('data', (data) => {
        modifiedValue += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error in Python Script: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            if (callback) {
                callback(modifiedValue.trim());
            }
        } else {
            console.error(`Python Script exited with code ${code}`);
            if (callback) {
                callback(null);
            }
        }
    });
}


// Serve your HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
