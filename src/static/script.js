(function() {
    
    let loaderContainer;
    let messageDiv;
    let uploadButton;
    let fileInput;
    let chooseFile;
    let fileName;

    function resetFormState(message) {
        loaderContainer.style.display = 'none';
        chooseFile.classList.remove('selected');
        uploadButton.disabled = true;
        fileInput.value = '';
        fileName.textContent = 'No file chosen';
        messageDiv.textContent = message;
    }

    function onFileInputChange() {
        uploadButton.disabled = !fileInput.value;
        chooseFile.classList.add('selected');
        messageDiv.textContent = '';
        fileName.textContent = fileInput.files.length > 0 ? fileInput.files[0].name : 'No file chosen';
    }

    function onSubmit(event) {
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
            resetFormState(data.message);
        })
        .catch(() => {
            resetFormState('An error occurred. Please try again.');
        });
    }

    document.addEventListener('DOMContentLoaded', function() {
        const loadingAppContainer = document.getElementById('loading-app-container');
        const uploadForm = document.getElementById('uploadForm');
        fileInput = document.getElementById('file');
        chooseFile = document.getElementById('chooseFile');
        fileName = document.getElementById('fileName');
        loaderContainer = document.getElementById('loader-container');
        messageDiv = document.getElementById('message');
        uploadButton = document.getElementById('uploadButton');

        // Hide the loading container and show the main content after 3 seconds
        setTimeout(function() {
            loadingAppContainer.classList.add('fade-out');

            // Wait for the transition to complete before hiding the container
            loadingAppContainer.addEventListener('transitionend', function() {
                loadingAppContainer.style.display = 'none';
            }, { once: true });
        }, 2000);

        // Trigger the file input when the custom button is clicked
        chooseFile.addEventListener('click', function() {
            fileInput.click();
        });

        // Enable the upload button when a file is selected
        fileInput.addEventListener('change', onFileInputChange);

        // Show the spinner when the form is submitted
        uploadForm.addEventListener('submit', onSubmit);
    });
})();