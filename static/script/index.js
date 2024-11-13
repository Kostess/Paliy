document.querySelector('#fileInput').addEventListener('change', (event) => {
    const label = document.querySelector('label[for="fileInput"]');
    const previewImage = document.querySelector('#previewImage');
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();

        reader.addEventListener('load', () => {
            previewImage.src = reader.result;
            previewImage.classList.remove('hidden');
        });

        reader.readAsDataURL(file);
        label.textContent = file.name;
    } else {
        previewImage.src = '#';
        previewImage.classList.add('hidden');
        label.textContent = 'Выберите файл';
    }
});