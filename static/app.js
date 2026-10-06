const form = document.getElementById("predictionForm");
const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const analyzeButton = document.getElementById("analyzeButton");
const buttonText = analyzeButton.querySelector(".button-text");
const errorMessage = document.getElementById("errorMessage");
const emptyResult = document.getElementById("emptyResult");
const predictionResult = document.getElementById("predictionResult");
const emotionIcon = document.getElementById("emotionIcon");
const emotionLabel = document.getElementById("emotionLabel");
const confidenceValue = document.getElementById("confidenceValue");
const probabilityList = document.getElementById("probabilityList");

const emotionMeta = {
  sadness: { emoji: "😔" },
  joy: { emoji: "😄" },
  love: { emoji: "🥰" },
  anger: { emoji: "😠" },
  fear: { emoji: "😨" },
  surprise: { emoji: "😲" },
};

function updateCharacterCount() {
  charCount.textContent = `${textInput.value.length} / 2000`;
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  buttonText.textContent = isLoading ? "Analyzing..." : "Analyze emotion";
}

function showError(message) {
  errorMessage.textContent = message;
}

function renderPrediction(data) {
  const emotion = data.predicted_emotion.toLowerCase();
  const confidence = Math.max(0, Math.min(1, data.confidence));

  emptyResult.classList.add("hidden");
  predictionResult.classList.remove("hidden");

  emotionIcon.textContent = emotionMeta[emotion]?.emoji ?? "✨";
  emotionLabel.textContent = emotion;
  confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;

  probabilityList.innerHTML = "";

  const sorted = Object.entries(data.all_probabilities)
    .sort(([, a], [, b]) => b - a);

  for (const [label, probability] of sorted) {
    const percent = Math.max(0, Math.min(100, probability * 100));
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `
      <div class="probability-name">
        <span>${emotionMeta[label]?.emoji ?? "•"}</span>
        <span>${label}</span>
      </div>
      <div class="probability-track" aria-label="${label} ${percent.toFixed(1)} percent">
        <div class="probability-fill" style="width: 0%"></div>
      </div>
      <div class="probability-value">${percent.toFixed(1)}%</div>
    `;
    probabilityList.appendChild(row);

    requestAnimationFrame(() => {
      row.querySelector(".probability-fill").style.width = `${percent}%`;
    });
  }
}

textInput.addEventListener("input", updateCharacterCount);

for (const chip of document.querySelectorAll(".example-chip")) {
  chip.addEventListener("click", () => {
    textInput.value = chip.dataset.text;
    updateCharacterCount();
    textInput.focus();
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");

  const text = textInput.value.trim();
  if (!text) {
    showError("Please enter a sentence first.");
    textInput.focus();
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      const detail = payload?.detail;
      throw new Error(
        typeof detail === "string"
          ? detail
          : "The server could not complete the prediction."
      );
    }

    renderPrediction(payload);
  } catch (error) {
    showError(
      error.message ||
      "Could not connect to the prediction API. Make sure FastAPI is running."
    );
  } finally {
    setLoading(false);
  }
});

updateCharacterCount();