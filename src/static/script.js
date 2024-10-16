
document.getElementById('file').addEventListener('change', function() {
    const uploadButton = document.getElementById('uploadButton');
    uploadButton.disabled = this.files.length === 0;
});
