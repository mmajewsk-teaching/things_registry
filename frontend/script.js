const API_BASE = "http://localhost:8000";

const output        = document.getElementById("output");
const spinnerWrap   = document.getElementById("spinner-wrap");
const spinnerText   = spinnerWrap.querySelector("p");
const resultPanel   = document.getElementById("result-panel");
const queryImg      = document.getElementById("query-img");
const matchesGrid   = document.getElementById("matches-grid");
const tabBtns       = document.querySelectorAll(".tab-btn");
const tabImages     = document.getElementById("tab-images");
const tabJson       = document.getElementById("tab-json");

function switchTab(name) {
    tabBtns.forEach(b => b.classList.toggle("active", b.dataset.tab === name));
    tabImages.classList.toggle("hidden", name !== "images");
    tabJson.classList.toggle("hidden",   name !== "json");
}

tabBtns.forEach(btn => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

function showSpinner(text = "Loading…") {
    spinnerText.textContent = text;
    spinnerWrap.classList.remove("hidden");
    resultPanel.classList.add("hidden");
    output.textContent = "";
}

function hideSpinner() {
    spinnerWrap.classList.add("hidden");
}

function renderResult(data, uploadedFile) {
    output.textContent = JSON.stringify(data, null, 2);

    const hasMatches = data.matches && data.matches.length > 0;

    if (hasMatches) {
        if (uploadedFile) {
            queryImg.src = URL.createObjectURL(uploadedFile);
            queryImg.style.display = "block";
        } else {
            queryImg.style.display = "none";
        }

        matchesGrid.innerHTML = "";
        data.matches.forEach((match, i) => {
            const sourcePath = match.metadata?.source_path;
            const pct        = ((match.similarity ?? 0) * 100).toFixed(1);
            const location   = match.metadata?.location || "unknown";

            const card = document.createElement("div");
            card.className = "match-card";
            card.innerHTML = `
                ${sourcePath
                    ? `<img src="${API_BASE}/static/${sourcePath}" alt="match ${i + 1}" loading="lazy">`
                    : `<div style="height:140px;display:flex;align-items:center;justify-content:center;color:#475569">no image</div>`
                }
                <div class="match-info">
                    <div class="match-similarity">${pct}% match</div>
                    <div class="match-location">📍 ${location}</div>
                </div>
            `;
            matchesGrid.appendChild(card);
        });

        switchTab("images");
    } else {
        switchTab("json");
    }

    resultPanel.classList.remove("hidden");
}

document.addEventListener("submit", async (e) => {
    e.preventDefault();

    const form     = e.target;
    const endpoint = API_BASE + form.dataset.endpoint;
    const method   = form.dataset.method || "GET";

    let options = { method };
    let uploadedFile = null;

    if (method !== "GET") {
        const formData = new FormData(form);
        const fileInput = form.querySelector('input[type="file"]');

        if (fileInput) {
            uploadedFile = fileInput.files[0] ?? null;
            options.body = formData;           // multipart - browser sets boundary
        } else {
            const body = Object.fromEntries(formData.entries());
            options.headers = { "Content-Type": "application/json" };
            options.body = JSON.stringify(body);
        }
    }

    showSpinner(form.dataset.loadingText);

    try {
        const res  = await fetch(endpoint, options);
        const text = await res.text();
        let data;
        try { data = JSON.parse(text); }
        catch { data = { status: res.status, raw: text }; }

        hideSpinner();
        renderResult(data, uploadedFile);

    } catch (err) {
        hideSpinner();
        renderResult({ error: err.message, type: err.constructor.name, endpoint }, null);
    }
});
