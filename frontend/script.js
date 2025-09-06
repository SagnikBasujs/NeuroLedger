document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    let fileInput = document.getElementById("csvFile");
    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        let response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        let result = await response.json();

        let outputDiv = document.getElementById("results");
        if (result.error) {
            outputDiv.innerHTML = `<p style="color:red;">Error: ${result.error}</p>`;
        } else {
            outputDiv.innerHTML = `<h3>Prediction Results:</h3><pre>${JSON.stringify(result.predictions, null, 2)}</pre>`;
        }
    } catch (err) {
        alert("Prediction failed: " + err.message);
    }
});
