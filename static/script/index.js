document.querySelector('#fileInput').addEventListener('change', function() {
    const label = document.querySelector('label[for="fileInput"]');
    if (this.files.length > 0) {
        label.textContent = this.files[0].name;
    } else {
        label.textContent = 'Выберите файл';
    }
});
