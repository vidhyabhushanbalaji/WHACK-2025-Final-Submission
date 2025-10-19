const API_BASE = "http://127.0.0.1:8000/api";

let sessionId = null;
let currentHeadlineId = null;
let timer = null;
let timeLeft = 0;
var count = 0;
const explanations = ["","","","","","","","","",""];

// -----------------------------
// Start session
// -----------------------------
// Start session when Start Game button is clicked
document.getElementById("startBtn").addEventListener("click", async () => {
    // Get username from input (default to "Player")
    try {
        // Make POST request to start session
        const response = await fetch(`${API_BASE}/session`, {
            method: "POST",                 // MUST be POST
            headers: {
                "Content-Type": "application/json"  // send JSON
            },
            body: JSON.stringify({ persona: "Trader" }) // payload
        });

        // Check if backend returned OK
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse JSON response
        const data = await response.json();
        sessionId = data.session_id;
        console.log("Session started with ID:", sessionId);

        // Show game section (hidden by default)
        document.getElementById("game").style.display = "block";

        // Start first headline
        getNextHeadline();

    } catch (error) {
        console.error("Error starting session:", error);
        alert("Failed to start game. Check backend is running!");
    }
});

// -----------------------------
// Get next headline
// -----------------------------
async function getNextHeadline() {
    clearInterval(timer);  // stop previous timer

    const res = await fetch(`${API_BASE}/session/${sessionId}/next`);
    const data = await res.json();

    if (data.done) {
        alert("Game finished!");
        getSummary();
        return;
    }

    currentHeadlineId = data.headline_id;

    document.getElementById("headline").innerText = (count+1)+ ". "+ data.text;
    document.getElementById("newstext").innerText= data.article_preview;

    // Timer countdown
    timeLeft = data.timer_s;
    document.getElementById("timer").innerText = timeLeft;
    timer = setInterval(() => {
        timeLeft--;
        document.getElementById("timer").innerText = timeLeft;
        if (timeLeft <= 0) {
            clearInterval(timer);
            sendDecision("ignore");
        }
    }, 1000);
}

// -----------------------------
// Send decision
// -----------------------------
async function sendDecision(choice) {
    clearInterval(timer);

    if (!currentHeadlineId) return;

    const res = await fetch(`${API_BASE}/session/${sessionId}/decision`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            headline_id: currentHeadlineId,
            choice: choice
        })
    });
    const data = await res.json();

    document.getElementById("score").innerText = data.total_score;
    alert("Confidence: " + JSON.stringify(data.confidence)+"\nReasoning: "+data.ai_rationale.explanation);
    explanations[count] = data.ai_rationale.explanation;
    count++;
    getNextHeadline();
}

// -----------------------------
// Buttons for decisions
// -----------------------------
document.getElementById("endorse").addEventListener("click", () => sendDecision("endorse"));
document.getElementById("ignore").addEventListener("click", () => sendDecision("ignore"));

// -----------------------------
// Get summary
// -----------------------------
async function getSummary() {
    const res = await fetch(`${API_BASE}/session/${sessionId}/summary`);
    const data = await res.json();

    document.getElementById("game").style.display = "none";
    document.getElementById("summary").style.display = "block";
    if (data.total_score>=60){
        document.getElementById("summaryContent").innerText = "Your score is: "+ data.total_score + "\nGood Job";
    }
    if (data.total_score<60) {
        document.getElementById("summaryContent").innerText = "Your score is: "+data.total_score + " \nBetter luck next time";
    }
}
