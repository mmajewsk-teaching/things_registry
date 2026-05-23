const API_BASE = "http://localhost:8000";

const output = document.getElementById("output");

function show(data) {
  output.textContent = JSON.stringify(data, null, 2);
}

document.addEventListener("submit", async (e) => {
    e.preventDefault();

    const form = e.target;

    const endpoint = API_BASE + form.dataset.endpoint;
    const method = form.dataset.method || "GET";

    let options = { method };

    if (method !== "GET") {
        const formData = new FormData(form);
        const body = Object.fromEntries(formData.entries());

        options.headers = {
        "Content-Type": "application/json",
        };
        options.body = JSON.stringify(body);
    }

    try {
        const res = await fetch(endpoint, options);
        const data = await res.json();
        show(data);

        
    } catch (err) {
        show({ error: err.message });
    }
});