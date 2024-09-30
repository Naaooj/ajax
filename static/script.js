

document.getElementById('file').addEventListener('change', function() {
    var uploadButton = document.getElementById('uploadButton');
    if (this.files.length > 0) {
        uploadButton.disabled = false;
    } else {
        uploadButton.disabled = true;
    }
});
