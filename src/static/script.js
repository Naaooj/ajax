
document.getElementById('file').addEventListener('change', function() {
    const uploadButton = document.getElementById('uploadButton');
    uploadButton.disabled = this.files.length === 0;
});

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const loaderContainer = document.getElementById('loader-container');
    const messageDiv = document.getElementById('message');
    const uploadButton = document.getElementById('uploadButton');
    const fileInput = document.getElementById('file');

    // Enable the upload button when a file is selected
    fileInput.addEventListener('change', function() {
        uploadButton.disabled = !fileInput.value;
        messageDiv.textContent = '';
    });

    // Show the spinner when the form is submitted
    uploadForm.addEventListener('submit', function(event) {
        // Prevent the form from submitting
        event.preventDefault();
        
        // Show the spinner
        loaderContainer.style.display = 'flex';

        // Send the form data to the server
        const formData = new FormData(uploadForm);
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loaderContainer.style.display = 'none';
            uploadButton.disabled = true;
            messageDiv.textContent = data.message;
        })
        .catch(error => {
            loaderContainer.style.display = 'none';
            messageDiv.textContent = 'An error occurred. Please try again.';
        });
    });
});