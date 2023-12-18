const form = document.getElementById('uploadForm')

        const sendFiles = async () => {
            // Object 
            const myFiles = document.getElementById('myFiles').files

            const formData = new FormData()

            Object.keys(myFiles).forEach(key => {
                formData.append(myFiles.item(key).name, myFiles.item(key))
            })

            const response = await fetch('http://localhost:3500/upload', {
                method: 'POST',
                body: formData
            })

            const json = await response.json()

            const upload_status = document.querySelector('p.upload-status')
            upload_status.textContent = `File Upload ${json?.status}`

            const uploaded_files = document.querySelector('p.uploaded-files')
            uploaded_files.textContent = json?.message

            console.log(json)
        }

        form.addEventListener('submit', (e) => {
            e.preventDefault()
            sendFiles()
        })