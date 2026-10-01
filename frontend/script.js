const API_URL = "http://127.0.0.1:8000";
const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const removeButton = document.getElementById("removeButton");
const predictButton = document.getElementById("predictButton");
const predictText = document.getElementById("predictText");
const loader = document.getElementById("loader");

const apiStatus = document.getElementById("apiStatus");

const emptyResult = document.getElementById("emptyResult");
const results = document.getElementById("results");
const errorResult = document.getElementById("errorResult");
const errorMessage = document.getElementById("errorMessage");

const predictedClass = document.getElementById("predictedClass");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const predictionList = document.getElementById("predictionList");

let selectedFile = null;

const allowedTypes = [
    "image/jpeg",
    "image/png",
    "image/webp"
];

const MAX_FILE_SIZE = 10 * 1024 * 1024;


// ============================================================
// FILE VALIDATION
// ============================================================

function validateFile(file) {

    if (!file) {
        showError("No image selected.");
        return false;
    }

    if (!allowedTypes.includes(file.type)) {
        showError(
            "Please select a JPEG, PNG, or WEBP image."
        );
        return false;
    }

    if (file.size > MAX_FILE_SIZE) {
        showError(
            "Image is too large. Maximum size is 10 MB."
        );
        return false;
    }

    return true;
}


// ============================================================
// FILE HANDLING
// ============================================================

function handleFile(file) {

    if (!validateFile(file)) {
        return;
    }

    selectedFile = file;

    fileName.textContent = file.name;

    const reader = new FileReader();

    reader.onload = function(event) {

        imagePreview.src = event.target.result;

        dropZone.classList.add("hidden");
        previewContainer.classList.remove("hidden");

        predictButton.disabled = false;
        predictButton.style.pointerEvents = "auto";
        predictButton.style.cursor = "pointer";

        clearResults();
    };

    reader.onerror = function() {
        showError("Unable to read the selected image.");
    };

    reader.readAsDataURL(file);
}


// ============================================================
// FILE INPUT
// ============================================================

imageInput.addEventListener("change", function(event) {

    const file = event.target.files[0];

    if (file) {
        handleFile(file);
    }

});


// ============================================================
// DRAG AND DROP
// ============================================================

dropZone.addEventListener("dragover", function(event) {

    event.preventDefault();

    dropZone.classList.add("dragover");

});


dropZone.addEventListener("dragleave", function() {

    dropZone.classList.remove("dragover");

});


dropZone.addEventListener("drop", function(event) {

    event.preventDefault();

    dropZone.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (file) {
        handleFile(file);
    }

});


// ============================================================
// REMOVE IMAGE
// ============================================================

removeButton.addEventListener("click", function() {

    selectedFile = null;

    imageInput.value = "";

    imagePreview.src = "";

    fileName.textContent = "-";

    previewContainer.classList.add("hidden");

    dropZone.classList.remove("hidden");

    predictButton.disabled = true;

    clearResults();

});


// ============================================================
// PREDICT BUTTON
// ============================================================

predictButton.addEventListener("click", function(event) {

    event.preventDefault();

    predictWaste();

});


// ============================================================
// PREDICT
// ============================================================

async function predictWaste() {

    if (!selectedFile) {
        showError("Please select an image first.");
        return;
    }

    setLoading(true);
    clearResults();

    try {

        const formData = new FormData();

        formData.append("file", selectedFile);

        console.log("Sending image to:", `${API_URL}/predict`);
        console.log("File:", selectedFile.name);
        console.log("Type:", selectedFile.type);
        console.log("Size:", selectedFile.size);

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",
                body: formData
            }
        );

        console.log("API status:", response.status);

        const responseText = await response.text();

        console.log("API response:", responseText);

        let data;

        try {
            data = JSON.parse(responseText);
        } catch {
            throw new Error(
                `API returned an invalid response. HTTP ${response.status}`
            );
        }

        if (!response.ok) {
            throw new Error(
                data.detail ||
                `Prediction failed. HTTP ${response.status}`
            );
        }

        displayResults(data);

    } catch (error) {

        console.error("Prediction error:", error);

        if (error instanceof TypeError) {

            showError(
                "Cannot connect to the FastAPI server. " +
                "Make sure the Docker API is running on http://127.0.0.1:8000."
            );

        } else {

            showError(
                error.message ||
                "Prediction failed."
            );

        }

    } finally {

        setLoading(false);

    }
}

// ============================================================
// DISPLAY RESULTS
// ============================================================

function displayResults(data) {

    emptyResult.classList.add("hidden");
    errorResult.classList.add("hidden");
    results.classList.remove("hidden");

    predictedClass.textContent =
        formatClassName(data.predicted_class);

    const confidence =
        Number(data.confidence) * 100;

    confidenceValue.textContent =
        `${confidence.toFixed(2)}%`;

    confidenceBar.style.width =
        `${Math.min(confidence, 100)}%`;

    predictionList.innerHTML = "";

    if (
        Array.isArray(data.top_predictions)
    ) {

        data.top_predictions.forEach(
            function(prediction, index) {

                const row =
                    document.createElement("div");

                row.className =
                    "prediction-row";

                const rank =
                    document.createElement("span");

                rank.className =
                    "prediction-rank";

                rank.textContent =
                    String(index + 1).padStart(2, "0");

                const name =
                    document.createElement("span");

                name.className =
                    "prediction-name";

                name.textContent =
                    formatClassName(
                        prediction.class
                    );

                const score =
                    document.createElement("span");

                score.className =
                    "prediction-confidence";

                score.textContent =
                    `${(
                        Number(
                            prediction.confidence
                        ) * 100
                    ).toFixed(2)}%`;

                row.appendChild(rank);
                row.appendChild(name);
                row.appendChild(score);

                predictionList.appendChild(row);

            }
        );

    }

}


// ============================================================
// FORMAT CLASS NAME
// ============================================================

function formatClassName(value) {

    if (!value) {
        return "-";
    }

    return String(value)
        .replace(/-/g, " ")
        .replace(/\b\w/g, function(letter) {
            return letter.toUpperCase();
        });

}


// ============================================================
// LOADING
// ============================================================

function setLoading(isLoading) {

    predictButton.disabled =
        isLoading || !selectedFile;

    if (isLoading) {

        predictText.textContent =
            "Classifying...";

        loader.classList.remove("hidden");

    } else {

        predictText.textContent =
            "Classify Waste";

        loader.classList.add("hidden");

    }

}


// ============================================================
// CLEAR RESULTS
// ============================================================

function clearResults() {

    results.classList.add("hidden");

    errorResult.classList.add("hidden");

    emptyResult.classList.remove("hidden");

    predictionList.innerHTML = "";

    confidenceBar.style.width = "0%";

    confidenceValue.textContent = "0%";

}


// ============================================================
// ERROR
// ============================================================

function showError(message) {

    emptyResult.classList.add("hidden");

    results.classList.add("hidden");

    errorResult.classList.remove("hidden");

    errorMessage.textContent =
        message;

}


// ============================================================
// API STATUS
// ============================================================

async function checkApiStatus() {

    try {

        const response =
            await fetch(
                `${API_URL}/health`,
                {
                    method: "GET",
                    cache: "no-store"
                }
            );

        if (!response.ok) {
            throw new Error();
        }

        const data =
            await response.json();

        if (data.model_loaded) {

            apiStatus.className =
                "status online";

            apiStatus.innerHTML =
                `
                <span class="status-dot"></span>
                API Online
                `;

        } else {

            apiStatus.className =
                "status offline";

            apiStatus.innerHTML =
                `
                <span class="status-dot"></span>
                Model Not Loaded
                `;

        }

    } catch (error) {

        console.error(
            "API health check failed:",
            error
        );

        apiStatus.className =
            "status offline";

        apiStatus.innerHTML =
            `
            <span class="status-dot"></span>
            API Offline
            `;

    }

}


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function() {

        predictButton.disabled = true;

        checkApiStatus();

        setInterval(
            checkApiStatus,
            10000
        );

    }
);
